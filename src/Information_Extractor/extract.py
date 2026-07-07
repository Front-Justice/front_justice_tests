###############
import math
## Script d'extraction à partir des segmentations. À lier avec le script "segmentation_kraken_yolo.

###############
import unicodedata
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
import src.Information_Extractor.semantic_search as semantic_search
import glob
import re
from src.utils.utils import OCRRecord, YOLOZone
from src.utils.utils import OCRLine
from src.utils.utils import YOLORecord
import src.utils.utils as utils
import copy
import PIL.Image as Image
import src.Information_Extractor.semantic_search as search
import src.Information_Extractor.geoextractor as geoextractor
import PIL
from collections import namedtuple
# import src.Vision.PARTY as PARTY
import src.Vision.KRAKEN as KRAKEN
import kraken.containers as containers
import src.date.parse_date as date
import fuzzysearch
import src.Information_Extractor.extraction_functions as extractions


class Extractor:
	"""
	Classe contenant un grand nombre de méthodes pour extraire les informations à partir:
	 	- d'un corpus d'annotations, objet YOLORecord
	 	- d'un ensemble de lignes, objet OCRRecord
	"""

	def __init__(self,
				 party_engine,
				 kraken_model_annotations: str,
				 kraken_model_transcription: str,
				 resize_factor: int = 1,
				 debug: bool = False,
				 use_party=False,
				 device="cpu",
				 minutier=None,
				 logger=None):
		"""
		Constructeur de la classe Extractor
		:param party_engine: le moteur party (instance de classe PARTY.PartyPredict)
		:param resize_factor: le facteur de redimension des images (accélère l'ocr)
		:param debug: Active le mode debug, ne charge pas le modèle party (va créer des erreurs)
		:param logger: Le logger utilisé pour avoir une trace des processus.
		"""

		# On initialise une pipeline de NER avec un modèle camembert adapté
		self.date_proces = ""
		self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates", local_files_only=False)
		self.ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates", local_files_only=False)
		utils.log_print(device)
		self.ner_model.to(device)
		self.logger = logger
		self.sentence_camembert = SentenceTransformer("dangvantuan/sentence-camembert-large", device=device, local_files_only=False)
		self.date_recognition_pipeline = pipeline('ner',
												  model=self.ner_model,
												  tokenizer=self.tokenizer,
												  aggregation_strategy="simple",
												  device=-1 if device == "cpu" else device)

		self.GeoExtractor = geoextractor.GeoExtractor()
		self.minute_courante = minutier
		entity_spotting_model = ("src/Information_Extractor/models/entity_spotting")
		self.kraken_model_annotations = kraken_model_annotations
		self.kraken_model_transcription = kraken_model_transcription
		tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base", local_files_only=False)
		self.entity_spotting_pipeline = pipeline('ner',
												 model=entity_spotting_model,
												 tokenizer=tokenizer,
												 aggregation_strategy="simple",
												 device=-1 if device == "cpu" else device)

		self.alto_namepaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		self.resize_factor: int = resize_factor
		self.use_party = use_party
		if debug is False:
			self.party = party_engine

		self.rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

		# On récupère les images en grande taille
		self.extracted_annotations = {}
		self.excluded_classes = ["Titre"]

	def filter_zones(self,
					 annotations: YOLORecord,
					 category: str) -> YOLORecord:
		"""
		Fonction permettant de filtrer les zones par catégorie
		:param annotations: un objet YOLORecord
		:param category: la catégorie à filtrer
		:return: un YOLORecord contenant uniquement les zones ciblées
		"""
		return [annotation for annotation in annotations if annotation.label == category]


	def extract_signature_greffier(self,
								   ocr_prediction:OCRRecord,
								   image: PIL.Image.Image) -> None:
		"""
		On extrait la ligne contenant la signature du greffier, pour clustering.
		:param ocr_prediction: un objet OCRRecord
		:param image: l'image chargée
		"""
		ligne_greffier, _ , index = utils.match_line_by_substring(corresponding_lines=ocr_prediction,
													   string_to_match="Le Greffier,",
													    exact_match=False,
																  return_index=True)

		# corresponding_idx = [idx for idx, line in enumerate(ocr_prediction) if line.prediction == "Le Greffier,"]
		#
		ligne_signature = ocr_prediction[index + 1]
		utils.log_print(ligne_signature.prediction)
		image = Image.open(image).convert("RGBA")
		height_rectangle = 150
		if "+" in ligne_signature.prediction:
			"Signature trouvée"
			# image = Image.open(self.minute_courante[3]["image_path"]).convert("RGBA")
			utils.extract_bbox_from_baseline(ligne_signature, height_rectangle=150, image=image)
		else:
			new_transcription = utils.extend_baseline_and_retranscribe(line=ligne_signature,
												   image_path=self.minute_courante[3]["image_path"],
												   ocr_model=self.kraken_model_transcription)
			utils.extract_bbox_from_baseline(new_transcription, height_rectangle=150, image=image)
			# utils.polygon_extraction(polygon, image)
		# return polygon

	def extract_lines_from_zone(self,
								annotations,
								target_zone: str | list,
								show_images: bool = False,
								loaded_image: PIL.Image.Image = None,
								ocr_prediction: OCRRecord = None,
								intersect_ratio: float | list[float] = 0.5,
								select_highest_prob_zone: bool = False) -> tuple[OCRRecord, list] | tuple[None, None]:
		"""
		Cette fonction extrait la ou les lignes correspondant à une zone
		:param annotations: L'ensemble des zones prédites
		:param target_zone: Le nom de la ou des zones à récupérer. Dans le cas de plusieurs zones, on récupèrera la ligne
		inclue dans l'intersection de ces zones
		:param show_images: Montrer l'image?
		:param loaded_image: L'image chargée par PIL (debug)
		:param ocr_prediction: OCRRecord: La liste de lignes transcrites (baseline, prediction, cuts)
		:param intersect_ratio: La proportion d'intersection minimale pour considérer la ligne dans la zone ciblée. Un flottant
		ou une liste de flottant si on vise plusieurs zones
		:param select_highest_prob_zone: Faut-il sélectionner la zone détecter la plus probable, en cas de zones multiples
		:return: La liste des lignes concernées par la zone et les zones.
		"""
		# On récupère la boîte correspondante
		if isinstance(intersect_ratio, float):
			intersect_ratio = [intersect_ratio]
		if isinstance(target_zone, str):
			all_zones = [target_zone]
		else:
			all_zones = target_zone
		all_filtered_zones = []
		all_lines = []
		zones_filtrees = []
		for target_zone, ratio in zip(all_zones, intersect_ratio):
			zones = self.filter_zones(annotations=annotations, category=target_zone)
			if len(zones) == 0:
				return None, None
			elif len(zones) > 1 and len(all_zones) == 1:
				utils.log_print(f"Erreur: plusieurs zones détectées. Target zone: {target_zone}")
				# Si on active l'option du choix de la zone la plus probable, il n'y a qu'à ordonner la liste
				if select_highest_prob_zone:
					zones.sort(key=lambda x: x.probs, reverse=True)
				else:
					return None, None

			# On va commencer par identifier le nom prédit par kraken
			coordonnees_zones_filtrees = zones[0].coordinates
			zones_filtrees.append(coordonnees_zones_filtrees)
			zones_filtrees_as_rectangle = self.rectangle(coordonnees_zones_filtrees[0][0],
														 coordonnees_zones_filtrees[0][1],
														 coordonnees_zones_filtrees[1][0],
														 coordonnees_zones_filtrees[1][1])
			all_filtered_zones.append(coordonnees_zones_filtrees)

			if show_images:
				# On doit adapter les dimensions à la taille de l'image chargée qui a été redimensionnée
				cropped = loaded_image.crop((coordonnees_zones_filtrees[0][0] * self.resize_factor,
											 coordonnees_zones_filtrees[0][1] * self.resize_factor,
											 coordonnees_zones_filtrees[1][0] * self.resize_factor,
											 coordonnees_zones_filtrees[1][1] * self.resize_factor))
				cropped.show()

			# On cherche la ligne qui entre dans la zone zonnée
			corresponding_lines = utils.match_lines_in_zones(ocr_prediction=ocr_prediction,
															 zone_as_rectangle=zones_filtrees_as_rectangle,
															 intersect_ratio=ratio)
			corresponding_lines = utils.vertical_order_lines(lines=corresponding_lines)
			all_lines.append([ocr_prediction.index(item) for item in corresponding_lines])

		if len(all_lines) == 1:
			corresponding_line_index = all_lines[0]
		else:
			# https://stackoverflow.com/a/3852806
			corresponding_line_index = set.intersection(*map(set, all_lines))
		corresponding_lines = OCRRecord()
		corresponding_lines.recreate_record([ocr_prediction[idx] for idx in corresponding_line_index])
		return corresponding_lines, zones_filtrees[0] if len(zones_filtrees) == 1 else zones_filtrees

	def extraire_lieu_jugement(self,
							   ocr_prediction: OCRRecord,
							   annotations: YOLORecord,
							   image: str = None,
							   loaded_image: PIL.Image.Image = None,
							   show_images: bool = True):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param annotations: un objet YOLORecord qui contient les coordonnées et labels de zone
		:param ocr_prediction: un objet OCRRecord qui contient les lignes prédites
		:param image: [Debug] le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
		  "Numéro": "501",
		  "baseline": [
			[2611, 555],
			[3203, 555]
		  ],
		  "bbox": [
			[2553, 178],
			[3249, 634]
		  ],
		  "certitude": 0.8,
		  "predictions": {
			"party": "rendu par le CONSEIL DE GUERRE permanent du Q. G. de la 2^e Armée séant aux Armées",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""
		corresponding_lines, lieu_jugement_zone = self.extract_lines_from_zone(annotations=annotations,
																			   target_zone="MainZone-judgementPlace",
																			   show_images=show_images,
																			   loaded_image=loaded_image,
																			   ocr_prediction=ocr_prediction,
																			   intersect_ratio=0.7)

		kraken_prediction = " ".join([line.prediction for line in corresponding_lines]).strip()
		corresponding_baselines = [line.baseline for line in corresponding_lines]

		# On transcrit avec party
		if self.use_party:
			party_segmentation = self.party.create_baseline(corresponding_baselines, image)
			party_prediction = self.party.timed_party_inference(segmentation=party_segmentation,
																image=loaded_image,
																objet_transcrit="lieu du jugement")
			party_prediction = " ".join([item.prediction for item in party_prediction]).strip()
		else:
			party_prediction = kraken_prediction

		chaine_seant = "séant"
		chaine_permanent = "permanent"
		clean_regexp_lieu = re.compile("^\s?[àa]\s?")
		clean_regexp_institution = re.compile("^\s?d[eu]?\s?")

		try:
			avant_seant_kraken = \
				utils.approximate_word_split(sentence=kraken_prediction, word=chaine_seant, sensibility=0.75)[
					0]
			institution_kraken = \
				utils.approximate_word_split(sentence=avant_seant_kraken, word=chaine_permanent)[1]
			institution_kraken = re.sub(clean_regexp_institution, "", institution_kraken)
		except IndexError:
			institution_kraken = None
		except TypeError:
			institution_kraken = None
		try:
			lieu_kraken, matching_word = utils.approximate_word_split(sentence=kraken_prediction, word=chaine_seant,
																	  return_word=True)
			lieu_kraken = lieu_kraken[1]
			lieu_kraken = re.sub(clean_regexp_lieu, "", lieu_kraken)
		except IndexError:
			lieu_kraken = None
		except TypeError:
			lieu_kraken = None

		try:
			avant_seant_party = \
				utils.approximate_word_split(sentence=party_prediction, word=chaine_seant, sensibility=0.75)[
					0]
			institution_party = \
				utils.approximate_word_split(sentence=avant_seant_party, word=chaine_permanent)[1]
			institution_party = re.sub(clean_regexp_institution, "", institution_party)
		except IndexError:
			institution_party = None
		except TypeError:
			institution_party = None
		try:
			lieu_party, matching_word = utils.approximate_word_split(sentence=party_prediction, word=chaine_seant,
																	 return_word=True)
			lieu_party = lieu_party[1]
			lieu_party = re.sub(clean_regexp_lieu, "", lieu_party)
		except IndexError:
			lieu_party = None
		except TypeError:
			lieu_party = None

		if party_prediction == kraken_prediction:
			certitude = 1
		else:
			certitude = 0.5
		if institution_party != "":
			institution = institution_party
		else:
			institution = institution_kraken
		if lieu_party != "":
			lieu = lieu_party
		else:
			lieu = lieu_kraken
		institution = utils.strip_punctuation(institution)
		lieu = utils.strip_punctuation(lieu)
		return {"institution": institution,
				"siège": lieu,
				"bbox": lieu_jugement_zone,
				"baseline": corresponding_baselines,
				"certitude": certitude,
				"predictions": {"party": party_prediction,
								"kraken": kraken_prediction}}

	def extraire_numero_jugement(self,
								 ocr_prediction: OCRRecord,
								 annotations: YOLORecord,
								 image: str = None,
								 loaded_image: PIL.Image.Image = None,
								 show_images: bool = True):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param image: [Debug] le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
		  "extracted": "501",
		  "baseline": [
			[2611, 555],
			[3203, 555]
		  ],
		  "bbox": [
			[2553, 178],
			[3249, 634]
		  ],
		  "certitude": 0.8,
		  "predictions": {
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""
		corresponding_lines, numero_jugement_zone = self.extract_lines_from_zone(annotations=annotations,
																				 target_zone="MainZone-judgementNumber",
																				 show_images=show_images,
																				 loaded_image=loaded_image,
																				 ocr_prediction=ocr_prediction,
																				 intersect_ratio=0.7)

		target_line = []
		for line in corresponding_lines:
			prediction = line.prediction
			# On va chercher une ligne avec un nombre uniquement ici
			expression_jugement = re.compile("\d+")
			is_jugement = re.match(expression_jugement, prediction)
			if is_jugement:
				target_line.append(line)
		if len(target_line) == 0:
			return {"Numéro": None,
					"baseline": None,
					"bbox": numero_jugement_zone,
					"certitude": None,
					"predictions": {"party": None,
									"kraken": None}}
		# assert len(target_line) == 1, "Erreur. Plusieurs lignes trouvées pour le numéro de jugement."

		numero_jugement_kraken = target_line[0].prediction
		# On transcrit avec party
		if self.use_party:
			party_segmentation = self.party.create_baseline([target_line[0].baseline], image)
			party_prediction = self.party.timed_party_inference(segmentation=party_segmentation,
																image=loaded_image,
																objet_transcrit="numéro de jugement")
			numero_jugement_party = party_prediction.prediction
		else:
			numero_jugement_party = numero_jugement_kraken

		target_line = target_line[0]

		if numero_jugement_party == numero_jugement_kraken:
			certitude = 0.8
			target_number = numero_jugement_kraken
		else:
			certitude = 0.5
			target_number = numero_jugement_party

		return {"extracted": target_number,
				"baseline": target_line.baseline,
				"bbox": numero_jugement_zone,
				"certitude": certitude,
				"predictions": {"party": numero_jugement_party,
								"kraken": numero_jugement_kraken}}



	def update_dict(self, nouveau_dictionnaire):
		self.minute_courante = nouveau_dictionnaire

	def extraire_description_soldat_NER_p2(self,
										   ocr_prediction: OCRRecord,
										   annotations: YOLORecord,
										   image_path:str,
										   loaded_image: PIL.Image.Image = None):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param annotations: l'objet YOLORecord avec les zones.
		:param ocr_prediction: un objet OCRRecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
		  "extracted": "501",
		  "baseline": [
			[2611, 555],
			[3203, 555]
		  ],
		  "bbox": [
			[2553, 178],
			[3249, 634]
		  ],
		  "certitude": 0.8,
		  "predictions": {
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""
		description_du_soldat = {}
		lignes_description_du_soldat, zone_identite_soldat = self.extract_lines_from_zone(annotations=annotations,
																					target_zone=["identite_soldat"],
																					show_images=False,
																					loaded_image=loaded_image,
																					ocr_prediction=ocr_prediction,
																					intersect_ratio=[.8],
																					select_highest_prob_zone=True)
		if not lignes_description_du_soldat:
			return None
		soldat: list[YOLOZone] = annotations.filter_zones("Nom du soldat")
		lignes_description_soldat_raw = lignes_description_du_soldat.join_transcription()
		# 1 correspond à la page 1, métadonnée qu'on envoie au modèle.
		lignes_description_soldat_raw = f"1 {utils.nfc_normalize(lignes_description_soldat_raw)}"
		result_spotting = self.entity_spotting_pipeline(lignes_description_soldat_raw)
		if len(soldat) == 1:
			bbox_nom_soldat = soldat[0].coordinates
			entite_et_baseline = extractions.extraire_entite_baseline(
				entities_list=result_spotting,
				nom_entite="nom_du_soldat",
				target_lines=lignes_description_du_soldat,
				image_path=image_path
			)
			if entite_et_baseline is None:
				description_du_soldat["identite"] = {
					"nom_1": {"extracted": None,
							"bbox": bbox_nom_soldat,
							"baseline": None}
				}
			elif len(entite_et_baseline) == 1:
				nom_du_soldat, baseline_nom_du_soldat = (entite_et_baseline[0]["extracted"],
														 entite_et_baseline[0]["baseline"])
				description_du_soldat["identite"] = {
					"nom_1": {"extracted": nom_du_soldat,
							"bbox": bbox_nom_soldat,
							"baseline": baseline_nom_du_soldat}
				}

		elif len(soldat) > 1:
			plusieurs_soldats = True
			bbox_nom_soldat = None
			utils.log_print("Plusieurs soldats.")
		else:
			bbox_nom_soldat = None
			utils.log_print("Aucun soldat identifié par YOLO.")

		description_du_soldat["prediction"] = lignes_description_soldat_raw

		description_du_soldat["identite"] = \
			{
				"prenom": extractions.extraire_feature(result_spotting,
													   lignes_description_du_soldat,
													   "prénom_du_soldat",
													   image_path=image_path),
				"nom_1": extractions.extraire_feature(result_spotting,
													lignes_description_du_soldat,
													"nom_du_soldat",
													   image_path=image_path)
			}

		description_du_soldat["identite"]["date_naissance"] = (
			extractions.extraire_date_naissance(entity_dict=result_spotting,
												lignes=lignes_description_du_soldat,
				image_path=image_path)
		)

		description_du_soldat["identite"]["lieu_naissance"] = (
			extractions.extraire_lieu_naissance(entity_dict=result_spotting,
												lignes=lignes_description_du_soldat,
												geoextractor=self.GeoExtractor,
													   image_path=image_path)
		)

		description_du_soldat["identite"]["lieu_residence"] = (
			extractions.extraire_lieu_residence(entity_dict=result_spotting,
												lignes=lignes_description_du_soldat,
												geoextractor=self.GeoExtractor,
												lieu_naissance = description_du_soldat["identite"]["lieu_naissance"],
													   image_path=image_path)
		)


		description_du_soldat["identite"]["situation_maritale"] = (
			extractions.extraire_sit_maritale(entity_dict=result_spotting,
											  lignes=lignes_description_du_soldat,
													   image_path=image_path)
		)

		description_du_soldat["identite"]["age"] = (
			extractions.extraire_feature(entities_list=result_spotting,
										 lignes=lignes_description_du_soldat,
										 feature="âge",
													   image_path=image_path)
		)
		description_du_soldat["identite"]["affectation"] = (
			extractions.extraire_feature(entities_list=result_spotting,
										 lignes=lignes_description_du_soldat,
										 feature="affectation_soldat",
													   image_path=image_path)
		)

		description_du_soldat["identite"]["rang"] = (
			extractions.extraire_feature(entities_list=result_spotting,
										 lignes=lignes_description_du_soldat,
										 feature="rang_actuel",
													   image_path=image_path)
		)

		# Profession
		profession = extractions.extraire_feature(
			result_spotting,
			lignes_description_du_soldat,
			"profession",
			image_path=image_path
		)

		description_du_soldat["profession"] = profession
		if description_du_soldat["profession"]:
			normalized, distance = utils.find_closest_word_in_list(target_word=profession["extracted"],
										word_list="src/resources/french_professions.txt",
										load_file=True)
			if distance > (len(normalized) / 2):
				description_du_soldat["profession"]["normalized"] = None
			else:
				description_du_soldat["profession"]["normalized"] = normalized

		else:
			description_du_soldat["profession"]["normalized"] = None

		return description_du_soldat

	def extraire_identite_defenseur(self,
									ocr_prediction: OCRRecord,
									annotations: YOLORecord,
									image_path: str = None,
									loaded_image: PIL.Image.Image = None,
									show_images: bool = False):
		"""
		Cette fonction extrait les informations sur la défenser à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: un objet OCRRecord
		:param image: [Debug] le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
		  "extracted": "501",
		  "baseline": [
			[2611, 555],
			[3203, 555]
		  ],
		  "bbox": [
			[2553, 178],
			[3249, 634]
		  ],
		  "certitude": 0.8,
		  "predictions": {
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""

		dictionnary = {}
		lignes_identite_defenseur, zone_identite_defenseur = self.extract_lines_from_zone(annotations=annotations,
																						  target_zone=[
																							  "seance_ouverte"],
																						  show_images=show_images,
																						  loaded_image=loaded_image,
																						  ocr_prediction=ocr_prediction,
																						  intersect_ratio=[.8])
		if lignes_identite_defenseur is None:
			return {"nom_du_defenseur": None}
		ligne_defenseur, *_ = utils.match_line_by_substring(corresponding_lines=lignes_identite_defenseur,
															string_to_match="défenseur")
		all_lines = lignes_identite_defenseur.join_transcription()
		try:
			apres_defenseur = utils.approximate_sentence_split(sentence=all_lines,
															   substring="accompagné de son défenseur",
															   max_dist=5)[-1]
		except TypeError:
			return {'nom_du_defenseur': {
				"extracted": None,
				"prediction": None}
			}
		entites_nommees = self.date_recognition_pipeline(apres_defenseur)
		try:
			nom_defenseur = [item["word"] for item in entites_nommees if item['entity_group'] == 'PER'][0]
		except IndexError:
			return {'nom_du_defenseur': {
				"extracted": "UNK",
				"prediction": apres_defenseur}
			}
		matching_line_defenseur = utils.match_line_by_substring(corresponding_lines=lignes_identite_defenseur,
																string_to_match=nom_defenseur,
																exact_match=True)
		baseline_nom_defenseur = utils.get_baseline_from_string(line=matching_line_defenseur,
																target_string=nom_defenseur,
																loaded_image=loaded_image,
																show_image=False,
																image_path=image_path)

		avocat = utils.check_substring_in_sentence(sentence=apres_defenseur, target_substring="avocat")
		notaire = utils.check_substring_in_sentence(sentence=apres_defenseur, target_substring="notaire")
		docteur = utils.check_substring_in_sentence(sentence=apres_defenseur, target_substring="docteur")
		designe_office = utils.check_substring_in_sentence(sentence=apres_defenseur,
														   target_substring="désigné d'office")

		dictionnary['nom_du_defenseur'] = {"extracted":
											   {"surname": {"persName": nom_defenseur},
												"avocat": avocat,
												"docteur": docteur,
												"notaire": notaire,
												"designe_office": designe_office,
												},
										   "bbox": zone_identite_defenseur,
										   "baseline": baseline_nom_defenseur,
										   "prediction": apres_defenseur
										   }

		return dictionnary

	def extraire_informations_procedure(self,
										ocr_prediction: OCRRecord,
										annotations: YOLORecord,
										image: str = None,
										loaded_image: PIL.Image.Image = None,
										show_images: bool = False):
		"""
		Cette fonction extrait les informations sur la défenser à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: un objet OCRRecord
		:param image: [Debug] le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		"""

		dictionnary = {}
		lignes_formalites, zone_formalites = self.extract_lines_from_zone(annotations=annotations,
																		  target_zone=["formalites"],
																		  show_images=show_images,
																		  loaded_image=loaded_image,
																		  ocr_prediction=ocr_prediction,
																		  intersect_ratio=[.8])
		try:
			lignes_formalites_as_string = lignes_formalites.join_transcription()
		except AttributeError:
			return {"prediction": None,
					"extracted": None,
					"bbox": None}
		similarite = utils.similarite_ratcliff(
			string_a="Et le Président ayant, en outre, rempli à leur égard les formalités prescrites par les articles 317 et 319 du Code d'instruction criminelle",
			string_b=lignes_formalites_as_string)
		if similarite > .8:
			precisions_jugement_temoins_defense = None
		else:
			try:
				precisions_jugement_temoins_defense = \
					utils.approximate_sentence_split(sentence=lignes_formalites_as_string,
													 substring="et 319 du Code d'instruction criminelle")[-1]
			except TypeError:
				precisions_jugement_temoins_defense = None

		return {"prediction": lignes_formalites_as_string,
				"extracted": precisions_jugement_temoins_defense,
				"bbox": zone_formalites}


	def extraire_questions_p2(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  image_path:str,
							  loaded_image: PIL.Image.Image = None) -> dict:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		# Si la zone n'est pas reconnue, on récupère tout de même le texte pour vérification
		if "questions" not in [item.label for item in annotations]:
			ocr_prediction = utils.vertical_order_lines(ocr_prediction)
			prediction_as_string = ocr_prediction.join_transcription()
			try:
				split, match = utils.approximate_sentence_split(sentence=prediction_as_string,
																substring=" Le Conseil délibérant à huis clos, le Président",
																return_match=True)
				target = f"{match} {split[-1]}"
			except TypeError:
				return None, None
			notes_bas_page_1 = "(1) et à décharge (s'il y en a)"
			notes_bas_page_2 = "(2) Indiquer si des témoins ont été entendus"
			split_bas_page_1 = utils.approximate_sentence_split(sentence=target, substring=notes_bas_page_1)
			if not split_bas_page_1:
				split_bas_page_2 = utils.approximate_sentence_split(sentence=target, substring=notes_bas_page_2)
				try:
					target = split_bas_page_2[0]
					return {"prediction": target,
							"extracted": None,
							"bbox": None}, None
				except TypeError:
					return {"prediction": split[-1],
							"extracted": None,
							"bbox": None}, None
			else:
				target = split_bas_page_1[0]
				return {"prediction": target,
						"extracted": None,
						"bbox": None}, None
		lignes_questions, zone_questions = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["questions"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.8],
																		select_highest_prob_zone=True)
		lignes_questions_as_string = lignes_questions.join_transcription()
		similarite = utils.similarite_ratcliff(string_a="L'accusé a été reconduit par l'escorte à la prison; "
														"le Commissaire du Gouvernement, le Greffier et les assistants dans "
														"l'auditoire se sont retirés sur l'invitation du Président (4); "
														"Le Conseil délibérant à huis clos, le Président a posé la question, "
														"conformément à l'article 132 du Code de justice militaire, ainsi qu'il suit: ",
											   string_b=lignes_questions_as_string)
		if similarite > .8:
			questions = None
		else:
			try:
				questions = \
					utils.approximate_sentence_split(sentence=lignes_questions_as_string,
													 substring="ainsi qu'il suit:")[-1]
			except TypeError:
				questions = None
		ligne_ainsi_quil_suit, _, index = utils.match_line_by_substring(corresponding_lines=lignes_questions,
															  string_to_match="ainsi qu'il suit",
															  return_index=True)
		try:
			ligne_nom_du_soldat = lignes_questions[index + 1].prediction
		except IndexError:
			return {"prediction": lignes_questions_as_string,
				"extracted": questions,
				"bbox": zone_questions}, None
		ligne_nom_du_soldat = f"1 {ligne_nom_du_soldat}"
		NER = self.entity_spotting_pipeline(ligne_nom_du_soldat)
		nom_soldat = {"nom_2": extractions.extraire_feature(entities_list=NER, lignes=lignes_questions,
									 feature="nom_du_soldat",
															image_path=image_path)}

		return {"prediction": lignes_questions_as_string,
				"extracted": questions,
				"bbox": zone_questions}, nom_soldat




	def extraire_questions_p3(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  loaded_image: PIL.Image.Image = None) -> dict:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		lignes_questions, zone_questions = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["questions"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.8],
																		select_highest_prob_zone=True)
		lignes_questions_as_string = lignes_questions.join_transcription()


		return {"prediction": lignes_questions_as_string,
				"extracted": lignes_questions_as_string,
				"bbox": zone_questions}

	def extraire_date_1_p4(self, ocr_prediction: OCRRecord):
		"""
		Extraction de la première occurrence de la date en page 4.
		:param ocr_prediction: la transcription de la page (objet OCRRecord)
		:return: la date normalisée
		"""

		# La ligne "Le jugement a été lu par nous contient la date.
		jugement_lu, _, index = utils.match_line_by_substring(corresponding_lines=ocr_prediction,
													string_to_match="le présent jugement a été lu par nous",
													return_index=True)
		ligne_jugement_lu = jugement_lu.prediction
		NER = self.date_recognition_pipeline(ligne_jugement_lu)
		if any(["date" in item.values() for item in NER]):
			date_identifiee = extractions.extraire_feature(entities_list=NER,
												lignes=jugement_lu,
												feature="date")
		else:
			try:
				date_identifiee = utils.approximate_sentence_split(sentence=ligne_jugement_lu,
																  substring="le présent jugement a été")[0]
				date_identifiee = utils.split_before_keep_delimiter(target_string=date_identifiee, delimiter="an")[-1]
				date_identifiee = date_identifiee.replace("L'", "")
			except TypeError:
				return {
					"extracted": None,
					"corrected": None,
					"normalized": None
				}
		corrected_date = utils.correct_date(date_identifiee)
		try:
			date_normalisee = date.process_date(corrected_date)
		except TypeError:
			date_normalisee = None

		return {
			"extracted": date_identifiee,
			"corrected": corrected_date,
			"normalized": date_normalisee
		}

	def extraire_noms_p4(self, ocr_prediction: OCRRecord,
						 image_path:str) -> dict:
		"""
		Cette méthode extrait le nom tel qu'il apparaît en haut de la page 4.
		:return:
		"""
		# La première occurrence arrive sur les deux premières lignes de la page
		deux_premieres_lignes = ocr_prediction[:2].join_transcription()
		deux_premieres_lignes = f"1 {deux_premieres_lignes}"
		NER = self.entity_spotting_pipeline(deux_premieres_lignes)
		nom = {"nom_1": extractions.extraire_feature(lignes=ocr_prediction[:2],
											entities_list=NER,
											feature="nom_du_soldat",
													 image_path=image_path)}

		# La seconde, "le jugement a été lu par nous"
		jugement_lu, _, index = utils.match_line_by_substring(corresponding_lines=ocr_prediction,
													string_to_match="le présent jugement a été lu",
													return_index=True)
		target_lines = ocr_prediction[index:index+2]
		target = target_lines.join_transcription()
		target = f"1 {target}"
		NER = self.entity_spotting_pipeline(target)
		nom["nom_2"] = extractions.extraire_feature(lignes=target_lines,
											entities_list=NER,
											feature="nom_du_soldat",
													image_path=image_path)

		return nom

	def extraire_tableau_p4(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
								image_path:str,
							  loaded_image: PIL.Image.Image = None) -> dict:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		lignes_tableau, zone_tableau = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["tableau_frais"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.5],
																		select_highest_prob_zone=True)

		lignes_recapitulatif_as_string = lignes_tableau.join_transcription(merge_newlines=False)
		try:
			liste_des_frais = utils.approximate_sentence_split(sentence=lignes_recapitulatif_as_string,
															   substring="dont le détail suit:\n")[-1]
		# Voir si on peut pas trouver une autre approche en cas d'erreur
		except TypeError:
			return {"prediction": lignes_recapitulatif_as_string,
			 "extracted": None,
			 "bbox": zone_tableau}, None
		regexp_lignes_frais = re.compile(r"\n?\d{1,2}\^[oO0]\s?")
		liste_des_frais = re.split(regexp_lignes_frais, liste_des_frais)
		liste_des_frais = [item for item in liste_des_frais if item != ""]
		split_regexp = re.compile(r"\.{3,}\s?(\d)")

		# On s'occupe des frais par ligne
		frais_engages = []
		for frais in liste_des_frais[:-1]:
			split_total = re.split(split_regexp, frais)
			if len(split_total) != 1:
				joint = "".join(split_total[-2:])
				extracted = utils.extraire_frais(joint)
				clean_line = re.split(re.compile(r"\.{3,}"), frais)[0]
				frais_engages.append({"ligne": clean_line,
									  "frais": extracted,
									  "as_string": joint})

		# On s'occupe des frais totaux
		total = liste_des_frais[-1]
		split_total = "".join(re.split(split_regexp, total)[-2:])
		frais_totaux = utils.extraire_frais(split_total)
		somme = sum([item["frais"] for item in frais_engages if item["frais"]])


		# On peut aussi récupérer le nom du soldat, une fois de plus
		# On cherche la première ligne de la zone normalement. Autant chercher par string.
		nom_regexp, _, index = utils.match_line_by_substring(corresponding_lines=lignes_tableau,
												   string_to_match="Vu la procédure instruite",
													return_index=True)
		ligne_nom = " ".join([lignes_tableau[index].prediction, lignes_tableau[index + 1].prediction])
		ligne_nom = f"1 {ligne_nom}"
		NER = self.entity_spotting_pipeline(ligne_nom)
		nom = extractions.extraire_feature(entities_list=NER,
										   lignes=lignes_tableau,
										   feature="nom_du_soldat",
										   image_path=image_path)
		prenom = extractions.extraire_feature(entities_list=NER,
										   lignes=lignes_tableau,
										   feature="prenom_du_soldat",
										   image_path=image_path)

		name_information = {
			"nom_3": nom,
			"prenom": prenom
		}

		return {"prediction": lignes_recapitulatif_as_string,
				"extracted": {"frais_totaux": {"somme": somme,
											   "totaux_transcrits": frais_totaux},
							  "liste_de_frais": frais_engages},
				"bbox": zone_tableau},  name_information

	def extraire_paragraphe_final_p4(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  loaded_image: PIL.Image.Image = None) -> tuple[dict, dict]:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		lignes_recapitulatif, zone_tableau = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["recapitulatif_somme"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.5],
																		select_highest_prob_zone=True)

		lignes_recapitulatif_as_string = lignes_recapitulatif.join_transcription()
		try:
			after_somme = utils.approximate_sentence_split(sentence=lignes_recapitulatif_as_string, substring="à la somme de ")[-1]
		except TypeError:
			return {"prediction": lignes_recapitulatif_as_string,
					"frais": None,
					"bbox": zone_tableau}, None
		try:
			somme_toutes_lettres = utils.approximate_sentence_split(sentence=after_somme, substring=" du montant de laquelle")[0]
		except TypeError:
			return {"prediction": lignes_recapitulatif_as_string,
					"frais": None,
					"bbox": zone_tableau}, None
		somme_toutes_lettres = somme_toutes_lettres.strip()
		total = utils.sum_to_float(somme_toutes_lettres)
		
		# On cherche la date du jugement, dans la dernière phrase du dernier paragraphe
		try:
			phrase_date = utils.approximate_sentence_split(sentence=lignes_recapitulatif_as_string, substring="Fait en la Chambre")[-1]
		except TypeError:
			date_line, _ = utils.match_line_by_substring(corresponding_lines=ocr_prediction, string_to_match="Fait en la Chambre du Conseil de Guerre")
			try:
				phrase_date = utils.approximate_sentence_split(sentence=date_line.prediction, substring="Fait en la Chambre")[-1]
			except TypeError:
				return {"predicted": lignes_recapitulatif_as_string,
						"extracted": total,
						"bbox": zone_tableau}, None
		ner = self.date_recognition_pipeline(phrase_date.lower())
		identified_date = [item for item in ner if item['entity_group'] == "DATE"]
		try:
			corrected = utils.correct_date(identified_date[0]['word'])
		except IndexError:
			return {"predicted": lignes_recapitulatif_as_string,
				"extracted": total,
				"bbox": zone_tableau}, None
		try:
			parsed = date.process_date(corrected)
		except TypeError:
			parsed = None

		date_du_proces = {
			"predicted": identified_date[0]['word'],
			"corrected": corrected,
			"normalized": parsed
		}



		return {"predicted": lignes_recapitulatif_as_string,
				"extracted": total,
				"bbox": zone_tableau}, date_du_proces



	def extraire_decision_tribunal_p3(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
									  image_path:str,
							  loaded_image: PIL.Image.Image = None) -> dict:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		lignes_decision, zone_decision = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["decision_tribunal"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.8],
																		select_highest_prob_zone=True)
		if lignes_decision is None:
			return {"prediction": None,
					"extracted": None,
					"bbox": zone_decision}, None
		try:
			lignes_decision_as_string = lignes_decision.join_transcription()
		except TypeError:
			return {"prediction": None,
				"extracted": None,
				"bbox": zone_decision}, None

		# On doit ajouter 0 comme une métadonnée pour le modèle.
		lignes_decision_as_string = f"0 {lignes_decision_as_string}"
		entities = self.entity_spotting_pipeline(lignes_decision_as_string)

		nom_du_soldat = {"nom": extractions.extraire_feature(entities_list=entities,
													  lignes=lignes_decision,
													  feature="nom_du_soldat",
															 image_path=image_path)}
		if "condamnation" in [item['entity_group'] for item in entities]:
			decision = "condamnation"
			peine = extractions.extraire_feature(entities_list=entities,
													  lignes=lignes_decision,
													  feature="peine",
															 image_path=image_path)
			type_de_peine = [
				"travaux publics",
				"prison",
				"peine de mort"
			]
			duree_de_la_peine = {
				1/30: "un jour",
				2/30: "deux jours",
				3/30: "trois jours",
				7/30: "sept jours",
				15/30: "quinze jours",
				1: "un mois",
				2: "deux mois",
				3: "trois mois",
				4: "quatre mois",
				5: "cinq mois",
				6: "six mois",
				7: "sept mois",
				8: "huis mois",
				9: "neuf mois",
				10: "dix mois",
				11: "onze mois",
				12: "un an",
				24: "deux ans",
				36: "trois ans",
				48: "quatre ans",
				60: "cinq ans",
				72: "six ans",
				84: "sept ans"
			}
			if peine and peine["extracted"] != "" and peine["extracted"] is not None:
				# Essayer de voir si une fonction de proximité formelle ne suffirait pas
				type_peine = semantic_search.retrieve_most_similar_sentence(sentence=peine["extracted"], queries=type_de_peine,
																			 embedder=self.sentence_camembert)
			else:
				type_peine = None
			if type_peine and type_peine != "peine de mort" and peine["extracted"] != "":
				duree_peine = semantic_search.retrieve_most_similar_sentence(sentence=peine["extracted"], queries=[item for item in duree_de_la_peine.values()],
																		 embedder=self.sentence_camembert)
				duree_peine = {val:key for key, val in duree_de_la_peine.items()}[duree_peine]
			else:
				duree_peine = None

			peine = {
				"predicted": peine,
				"extracted":
					{"duree": duree_peine,
					 "type": type_peine}
			}




			unanimite = extractions.extraire_feature(entities_list=entities,
													  lignes=lignes_decision,
													  feature="unanimité",
															 image_path=image_path)


			# On traite la majorité, de la même manière
			majorite = extractions.extraire_feature(entities_list=entities,
													  lignes=lignes_decision,
													  feature="majorité",
															 image_path=image_path)
			if majorite and majorite['extracted'] != "" and majorite["extracted"] is not None:
				types_majorite = {
					"3/2": "trois voix contre deux",
					"4/1":"quatre voix contre une"
				}
				type_de_majorite = semantic_search.retrieve_most_similar_sentence(sentence=majorite["extracted"],
																			 queries=[item for item in
																					  types_majorite.values()],
																			 embedder=self.sentence_camembert)
				vote = {
					"predicted": majorite['extracted'],
					"extracted": {val: key for key, val in types_majorite.items()}[type_de_majorite]
				}

			else:
				vote = None


			sursis = extractions.extraire_feature(entities_list=entities,
													  lignes=lignes_decision,
													  feature="sursis",
															 image_path=image_path)
			extracted = {"decision": decision,
						 "peine": peine,
						 "vote": "unanimité" if unanimite['extracted'] not in ['', None] else "majoritaire",
						 "voix": vote if majorite else None,
						 "sursis": True if sursis and sursis["extracted"] != None else False}

		elif "acquittement" in [item['entity_group'] for item in entities]:
			decision = "acquittement"
			peine = None
			unanimite = None
			majorite = None
			sursis = None
			extracted = {"decision": decision,
						 "peine": peine,
						 "vote": None,
						 "voix": None,
						 "sursis": None}
		else:
			extracted = {"decision": "UNK",
						 "peine": "UNK",
						 "vote": "UNK",
						 "voix": "UNK",
						 "sursis": "UNK"}



		return {"prediction": lignes_decision_as_string,
				"bbox": zone_decision, **extracted}, nom_du_soldat




	def extraire_reponses_p3(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  loaded_image: PIL.Image.Image = None) -> dict:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		lignes_reponses, zone_reponses = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["reponse_questions"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.8],
																		select_highest_prob_zone=True)
		try:
			lignes_reponses_as_string = lignes_reponses.join_transcription()
		except AttributeError:
			return {"prediction": None,
					"extracted": None,
					"bbox": zone_reponses}


		return {"prediction": lignes_reponses_as_string,
				"extracted": lignes_reponses_as_string,
				"bbox": zone_reponses}

		return dictionnary


	def extraire_requisitoire(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  image: str = None,
							  loaded_image: PIL.Image.Image = None,
							  show_images: bool = False):
		"""
		Cette fonction extrait les informations sur le réquisitoire à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param image: [Debug] le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		"""

		lignes_requisitoire, zone_requisitoire = self.extract_lines_from_zone(annotations=annotations,
																			  target_zone=["requisitoire"],
																			  show_images=show_images,
																			  loaded_image=loaded_image,
																			  ocr_prediction=ocr_prediction,
																			  intersect_ratio=[.8],
																			  select_highest_prob_zone=True)
		if lignes_requisitoire is None:
			return None
		lignes_requisitoire_as_string = lignes_requisitoire.join_transcription()
		similarite = utils.similarite_ratcliff(string_a="Ouï M. le Commissaire du Gouvernement en ses réquisitions "
														"tendants à ce que (3)",
											   string_b=lignes_requisitoire_as_string)
		if similarite > .8:
			requisitoire = None
		else:
			try:
				requisitoire = utils.approximate_sentence_split(sentence=lignes_requisitoire_as_string,
																substring="tendant à ce que (3)")[-1]
			except TypeError:
				requisitoire = None

		if requisitoire:
			apres_application = utils.approximate_sentence_split(sentence=requisitoire,
																 substring="articles",
																 max_dist=2)
			if not apres_application:
				apres_application = utils.approximate_sentence_split(sentence=requisitoire,
																	 substring="application",
																	 max_dist=2)
			try:
				articles_vises = apres_application[-1]
			except TypeError:
				articles_vises = None
		else:
			articles_vises = None

		return {"prediction": lignes_requisitoire_as_string,
				"extracted": {
					"requisitoire": requisitoire,
					"articles_vises": articles_vises
				},
				"bbox": zone_requisitoire}

	def extraire_date_crime_ou_delit(self,
									 ocr_prediction: OCRRecord,
									 annotations: YOLORecord,
									 image: str = None,
									 loaded_image: PIL.Image.Image = None,
									 show_images: bool = True):
		"""
		Cette fonction extrait la date du crime à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: un objet OCRRecord avec les lignes prédites
		:param annotations: un objet YOLORecord avec les zones identifiées
		:param image: [Debug] le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
		  "Date": "501",
		  "baseline": [
			[2611, 555],
			[3203, 555]
		  ],
		  "bbox": [
			[2553, 178],
			[3249, 634]
		  ],
		  "certitude": 0.8,
		  "predictions": {
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""

		# On récupère les lignes concernées par les zones
		corresponding_lines, date_zone = self.extract_lines_from_zone(annotations=annotations,
																	  target_zone="MainZone-crimeDate",
																	  show_images=show_images,
																	  loaded_image=loaded_image,
																	  ocr_prediction=ocr_prediction,
																	  intersect_ratio=0.7)

		# La date du crime est toujours sur la deuxième ligne
		# sauf quand elle n'est pas renseignée
		try:
			target_line = [corresponding_lines[1]]
		except IndexError:
			return {"bbox": date_zone,
					"normalized": None,
					"extracted": None,
					"predicted": None}

		date_crime_kraken = target_line[0].prediction

		# On transcrit avec party
		if self.use_party:
			party_segmentation = self.party.create_baseline([target_line[0].baseline], image)
			party_prediction = self.party.timed_party_inference(
				segmentation=party_segmentation,
				image=loaded_image,
				objet_transcrit="date du crime")
			date_crime_party = party_prediction.prediction
		else:
			date_crime_party = date_crime_kraken

		target_line = target_line[0]

		if date_crime_party == date_crime_kraken:
			certitude = 0.8
			target_date = date_crime_kraken
		else:
			certitude = 0.5
			target_date = date_crime_party

		# On utilise le parseur pour produire la date normalisée
		try:
			# TODO: la correction supprime les tirets et la ponctuation, marqueur de période ou répétition, voir comment corriger ça
			corrected_date = utils.correct_date(target_date)
		except TypeError:
			return {"normalized": None,
					"corrected": None,
					"extracted": date_crime_party,
					"baseline": target_line.baseline,
					"bbox": date_zone,
					"certitude": certitude,
					"predicted": {"party": date_crime_party,
									"kraken": date_crime_kraken}}
		try:
			normalized_date = date.process_date(corrected_date, debug=False)
		except TypeError:
			normalized_date = None

		return {"normalized": normalized_date,
				"corrected": corrected_date,
				"extracted": date_crime_party,
				"baseline": target_line.baseline,
				"bbox": date_zone,
				"certitude": certitude,
				"predicted": {"party": date_crime_party,
								"kraken": date_crime_kraken}}

	def extraire_informations_ajouts_posterieurs(self,
												 ocr_prediction: OCRRecord,
												 annotations: YOLORecord):
		"""
		Cette fonction permet de traiter les ajouts postérieurs (actualisations sur un soldat).
		Extrait la date et classifie le type d'information d'un ajout
		du greffier.
		:param ocr_prediction: un objet OCRRecord, qui contient toutes les lignes ajoutées
		identifiées par le modèle
		:param annotations: un objet YOLORecord, qui contient toutes les zones comprenant
		un ajout.
		:return: la liste des ajouts identifiés, contenant la zone, les lignes et les prédictions,
		le type d'information contenue dans l'ajout.
		"""
		list_of_results = []
		for annotation in annotations:
			# On va commencer par filtrer les lignes dans la zone.
			zones_filtrees_as_rectangle = self.rectangle(annotation.coordinates[0][0],
														 annotation.coordinates[0][1],
														 annotation.coordinates[1][0],
														 annotation.coordinates[1][1])
			filtered_lines = utils.match_lines_in_zones(ocr_prediction=ocr_prediction,
									   zone_as_rectangle=zones_filtrees_as_rectangle,
									   intersect_ratio=0.3)
			as_record = OCRRecord()
			as_record.recreate_record(filtered_lines)

			# On essaie d'améliorer l'OCR en déplaçant de quelques pixels chaque ligne dans la direction orthogonale.
			# as_record = utils.find_best_transcription(lines=as_record,
			# 										  image_path=image_path,
			# 										  step=1,
			# 										  ranges=(-1, 1),
			# 										  ocr_model=self.kraken_model)

			sorted_lines = utils.sort_lines_with_rotation(as_record, zones_filtrees_as_rectangle)
			lignes_fusionnees = sorted_lines.join_transcription()
			lignes_fusionnees = lignes_fusionnees.lower()
			resultat = self.date_recognition_pipeline(lignes_fusionnees.lower())
			# TODO: reprendre ça, il peut y avoir plusieurs dates
			normalized_dates = []
			dates = [item for item in resultat if item['entity_group'] == 'DATE']
			for current_date in dates:
				extracted_date = current_date['word']
				try:
					corrected_date = utils.correct_date(extracted_date)
					normalized_date = date.process_date(corrected_date, debug=False)
				except TypeError:
					normalized_date = None
				normalized_dates.append(normalized_date)
			if len(normalized_dates) == 1:
				extracted_date = {"date_1": normalized_dates[0]}
			elif len(normalized_dates) == 0:
				extracted_date = None
			else:
				date_1 = normalized_dates[0]
				date_2 = normalized_dates[-1]
				extracted_date = {
					"date_1": date_1,
					"date_2": date_2
				}
			list_of_informations = [
				"Remise du restant de la peine",
				"Remise partielle de peine",
				"Décès du soldat",
				"Détention préventive",
				"Amnistie",
				"Réhabilitation du soldat",
				"Peine effectuée",
				"Exécution du jugement",
				"Exécution du jugement suspendue",
				"Jugement suspendu",
				"Annulation de la suspension d'exécution",
				"Exécution de la peine suspendue",
				"Peine commuée",
				"Sursis révoqué"
			]
			if lignes_fusionnees != "":
				information_contenue = search.retrieve_most_similar_sentence(sentence=lignes_fusionnees,
																			 queries=list_of_informations,
																			 embedder=self.sentence_camembert)
			else:
				information_contenue = None

			# TODO: cas où il y a plusieurs annotations différentes
			list_of_results.append({
				"date": extracted_date,
				"information": information_contenue,
				"prediction": lignes_fusionnees,
				"bbox": annotations[0].coordinates,
				"baselines": [item.baseline for item in as_record]
			})
		return list_of_results



	def extraire_inculpation_et_antecedents(self,
											ocr_prediction: OCRRecord,
											annotations: YOLORecord,
											loaded_image: PIL.Image.Image = None,
											show_images: bool = True):
		"""
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: Un objet OCRRecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
		  "Numéro": "501",
		  "baseline": [
			[2611, 555],
			[3203, 555]
		  ],
		  "bbox": [
			[2553, 178],
			[3249, 634]
		  ],
		  "certitude": 0.8,
		  "predictions": {
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""
		inculpation = {"antécédents": {},
					   "inculpation": {}}
		corresponding_lines, _ = self.extract_lines_from_zone(annotations=annotations,
																			  target_zone="Inculpation_antecedents",
																			  show_images=show_images,
																			  loaded_image=loaded_image,
																			  ocr_prediction=ocr_prediction,
																			  intersect_ratio=0.7)
		# On commence par l'inculpation
		corresponding_lines = utils.vertical_order_lines(corresponding_lines)
		lignes_inculpe, _, correct_index_inculpe = utils.match_line_by_substring(
			corresponding_lines=corresponding_lines,
			string_to_match=["Inculpé de:", "Prévenu de:", "Accusé de:"], return_index=True)
		ligne_condamnations, _, correct_index_condamnations = utils.match_line_by_substring(
			corresponding_lines=corresponding_lines, string_to_match="Condamnations", return_index=True)
		lignes_inculpation = corresponding_lines[correct_index_inculpe:correct_index_condamnations]
		lignes_inculpation_str = " ".join([item.prediction for item in lignes_inculpation])
		lignes_inculpation_str = utils.nfc_normalize(lignes_inculpation_str)
		inculpation["inculpation"]["predicted"] = lignes_inculpation_str
		check_inculpe, mot_inculpe = utils.check_word_in_sentence(sentence=lignes_inculpation_str,
																  target_word=["Inculpé", "Prévenu", "Accusé"],
																  sensibility=0.8)
		if check_inculpe:
			lignes_inculpation_str = lignes_inculpation_str.replace(mot_inculpe, "")

		lignes_inculpation_str = utils.strip_punctuation(lignes_inculpation_str)
		clean_regexp = re.compile("^\s?d[e']?:?\s?")
		lignes_inculpation_str = re.sub(clean_regexp, "", lignes_inculpation_str)
		inculpation["inculpation"]["extracted"] = lignes_inculpation_str

		# On fait de même pour la condamnation, en changeant un peu le split (2 mots)
		lignes_condamnation = corresponding_lines[correct_index_condamnations:]
		lignes_condamnations_str = " ".join([item.prediction for item in lignes_condamnation])
		lignes_condamnations_str = utils.nfc_normalize(lignes_condamnations_str)
		check_condamnations, mot_condamnations = utils.check_substring_in_sentence(sentence=lignes_condamnations_str,
																				   target_substring="Condamnations antérieures",
																				   max_distance=4,
																				   return_subtring=True)
		inculpation["antécédents"]["predicted"] = lignes_condamnations_str
		try:
			mot_condamnations = mot_condamnations[0].matched
		except IndexError:
			return None

		if check_condamnations:
			lignes_condamnations_str = lignes_condamnations_str.split(mot_condamnations)[-1]
		if len(lignes_condamnations_str) < 15:
			check_neant, mot_neant = utils.check_word_in_sentence(sentence=lignes_condamnations_str,
																  target_word="Néant", sensibility=0.75)
		else:
			check_neant = False
		if check_neant is True:
			inculpation["antécédents"]["extracted"] = "Néant"

		# On va essayer d'isoler chaque condamnation en considérant qu'elle est toujours bornée par une date.
		else:
			check_date_regexp = re.compile(r"\d{4}")

			all_dates = [item for item in self.date_recognition_pipeline(lignes_condamnations_str) if item['entity_group'] == "DATE"]
			bornes_dates = []
			for date in all_dates:
				is_date = len(re.findall(check_date_regexp, date['word'])) > 0
				if is_date:
					bornes_dates.append(date['start'])
			# https://stackoverflow.com/a/10851479
			condamnations_individuelles = [lignes_condamnations_str[i:j] for i, j in
										   zip(bornes_dates, bornes_dates[1:] + [None])]
			inculpation["antécédents"]["extracted"] = condamnations_individuelles

		return inculpation

	def extraire_numero_ordre(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  image_path: str = None,
							  loaded_image: PIL.Image.Image = None,
							  show_images: bool = True):
		"""
		Cette fonction extrait le numéro d'ordre à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: Objet OCRRecord
		:param annotations: Objet YOLORecord
		:param image_path: le chemin vers l'image à afficher
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		:param show_images: [Debug] afficher l'image?
		:return: Un dictionnaire de la forme:
			{
          "extracted": "501",
          "baseline": [
            [2611, 555],
            [3203, 555]
          ],
          "bbox": [
            [2553, 178],
            [3249, 634]
          ],
          "certitude": 0.8,
          "predictions": {
            "party": "N^o 501 D'ORDRE.",
            "kraken": "N^o 501 D'ORDRE."
          }
        },
		"""
		corresponding_lines, numero_ordre_zone = self.extract_lines_from_zone(annotations=annotations,
																			  target_zone="MainZone-orderNumber",
																			  show_images=show_images,
																			  loaded_image=loaded_image,
																			  ocr_prediction=ocr_prediction,
																			  intersect_ratio=0.7)
		if (corresponding_lines, numero_ordre_zone) == (None, None):
			# TODO: reprendre cela, ça semble bizarre.
			utils.log_print("Error with order number", print_message=True)
			exit(0)
			return {"extracted": None,
					"baseline": None,
					"bbox": None,
					"certitude": None,
					"predictions": {"party": None,
									"kraken": None}}

		# On va commencer par tester si la page est bien classifiée, et identifier des formulaires différents
		first_line = corresponding_lines[0].prediction
		num_regexp = re.compile(r"N\^o (\d+)")
		num_regexp_2 = re.compile(r"\d+")

		# On teste 2 expressions régulières puis une présence de substring dans la première ligne.
		try:
			numero_nomenclature = re.search(num_regexp, first_line).group(1)
		except AttributeError:
			try:
				numero_nomenclature = re.search(num_regexp_2, first_line).group()
			except AttributeError:
				numero_nomenclature = "967" if "967" in first_line else None
		if numero_nomenclature == "967":
			pass
		elif numero_nomenclature == "974":
			utils.log_print("Le formulaire 974 (bis) a été identifié: révision de procès. Le processus s'arrête pour l'instant.")
			exit(0)
		else:
			utils.log_print("Le numéro de formulaire n'est pas trouvé. Il peut s'agir d'une erreur d'OCR "
				  "ou de classification de la page.")

		target_line = []
		for line in corresponding_lines:
			prediction = line.prediction
			similarity = utils.similarite_ratcliff(prediction, "D'ORDRE.")

			# On condidère une valeur de similarité de 0.5, à modifier par l'expérience
			if similarity > .5:
				target_line.append(line)

		if len(target_line) != 1:
			utils.log_print(f"Erreur. Zéro ou  Plusieurs lignes trouvées pour le numéro d'ordre:\n"
				  f"{target_line}")
			return None

		# On transcrit avec party
		if self.use_party:
			party_segmentation = self.party.create_baseline([target_line[0].baseline], image)
			party_prediction = self.party.timed_party_inference(
				segmentation=party_segmentation,
				image=loaded_image,
				objet_transcrit="numéro d'ordre")
			numero_ordre_party = party_prediction.prediction
		else:
			numero_ordre_party = target_line[0].prediction

		target_line = target_line[0]

		numero_regexp = re.compile("\d+")
		try:
			target_number_kraken = re.search(numero_regexp, target_line.prediction).group()
		except AttributeError:
			target_number_kraken = None
		try:
			target_number_party = re.search(numero_regexp, numero_ordre_party).group()
		except AttributeError:
			target_number_party = None

		if target_number_party == target_number_kraken:
			certitude = 0.8
			target_number = target_number_kraken
		else:
			certitude = 0.5
			target_number = target_number_party
		if target_number:
			baseline = utils.get_baseline_from_string(line=target_line,
													  target_string=target_number,
													  loaded_image=loaded_image,
													  image_path=image_path,
													  show_image=False)
		else:
			baseline = None
		return {"extracted": target_number,
				"baseline": baseline,
				"bbox": numero_ordre_zone,
				"certitude": certitude,
				"predictions": {"party": numero_ordre_party,
								"kraken": target_line.prediction}}


	def extraire_description_soldat_NER_p1(self,
										   ocr_prediction: OCRRecord,
										   annotations: YOLORecord,
										   image_path:str):
		"""
		Cette fonction extrait la description du soldat à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:return: Un dictionnaire de la forme:
			{
				"extracted": {
				  "surname": "Delin",
				  "forename": "",
				  "certainty": 0.5
				},
				"bbox": [
				[[225, 2137 ], [2631, 2194]],
				[[2723,2114], [3005, 2114]]
				],
				"prediction": [
				  "",
				]
			  }
		"""

		description_du_soldat = {
			"description_physique": {}
		}
		# On commence par ne récupérer que les lignes qui décrivent le soldat.
		lignes_description_du_soldat, soldat_zone = self.extract_lines_from_zone(annotations=annotations,
																				 target_zone="Description du Soldat",
																				 show_images=False,
																				 loaded_image=None,
																				 ocr_prediction=ocr_prediction,
																				 intersect_ratio=0.1,
																				 select_highest_prob_zone=True)
		utils.log_print(lignes_description_du_soldat)
		try:
			lignes_description_soldat_raw = lignes_description_du_soldat.join_transcription()
		except AttributeError:
			return None
		try:
			target_lines = utils.approximate_sentence_split(sentence=lignes_description_soldat_raw,
															substring="A l'effet de juger")[-1]
		except TypeError:
			target_lines = lignes_description_soldat_raw


		description_du_soldat["prediction"] = target_lines
		target_lines = f"1 {target_lines}"
		entities = self.entity_spotting_pipeline(target_lines)

		# On commence par le nom du soldat
		soldat: list[YOLOZone] = annotations.filter_zones("Nom du soldat")

		# plusieurs_soldats = False
		if len(soldat) == 1:
			bbox_nom_soldat = soldat[0].coordinates
			entite_et_baseline = extractions.extraire_entite_baseline(
				entities_list=entities,
				nom_entite="nom_du_soldat",
				target_lines=lignes_description_du_soldat,
				image_path=image_path
			)
			if entite_et_baseline is None:
				description_du_soldat["identite"] = {
					"nom": {"extracted": None,
							"bbox": bbox_nom_soldat,
							"baseline": None}
				}
			elif len(entite_et_baseline) == 1:
				nom_du_soldat, baseline_nom_du_soldat = (entite_et_baseline[0]["extracted"],
														 entite_et_baseline[0]["baseline"])
				description_du_soldat["identite"] = {
					"nom": {"extracted": nom_du_soldat,
							"bbox": bbox_nom_soldat,
							"baseline": baseline_nom_du_soldat}
				}

		elif len(soldat) > 1:
			# plusieurs_soldats = True
			# bbox_nom_soldat = None
			utils.log_print("Plusieurs soldats.")
			description_du_soldat["identite"] = {
				"nom": {"extracted": None,
						"bbox": None,
						"baseline": None}
			}
		else:
			# bbox_nom_soldat = None
			utils.log_print("Aucun soldat identifié par YOLO.")


		description_du_soldat["identite"] = \
			{
				"prenom": extractions.extraire_feature(entities,
													   lignes_description_du_soldat,
													   "prénom_du_soldat",
				image_path=image_path),
				"nom": extractions.extraire_feature(entities,
													lignes_description_du_soldat,
													"nom_du_soldat",
				image_path=image_path)
			}

		description_du_soldat["parents"] = {
			"pere": {
				"prenom": extractions.extraire_feature(
					entities,
					lignes_description_du_soldat,
					"prenom_pere",
				image_path=image_path
				)
			},
			"mere": {
				"prenom": extractions.extraire_feature(
					entities,
					lignes_description_du_soldat,
					"prenom_mere",
				image_path=image_path
				),
				"nom": extractions.extraire_feature(
					entities,
					lignes_description_du_soldat,
					"nom_mere",
				image_path=image_path
				),
			}
		}

		description_du_soldat["identite"]["date_naissance"] = (
			extractions.extraire_date_naissance(entity_dict=entities,
												lignes=lignes_description_du_soldat,
												image_path=image_path)
		)

		description_du_soldat["identite"]["lieu_naissance"] = (
			extractions.extraire_lieu_naissance(entity_dict=entities,
												lignes=lignes_description_du_soldat,
												geoextractor=self.GeoExtractor,
												image_path=image_path)
		)


		description_du_soldat["identite"]["lieu_residence"] = (
			extractions.extraire_lieu_residence(entity_dict=entities,
												lignes=lignes_description_du_soldat,
												geoextractor=self.GeoExtractor,
												lieu_naissance=description_du_soldat["identite"]["lieu_naissance"],
		image_path=image_path)
		)


		description_du_soldat["identite"]["situation_maritale"] = (
			extractions.extraire_sit_maritale(entity_dict=entities,
												lignes=lignes_description_du_soldat,
		image_path=image_path)
		)

		description_du_soldat["identite"]["matricule"] = (
			extractions.extraire_feature(entities_list=entities,
										 lignes=lignes_description_du_soldat,
										 feature="matricule",
				image_path=image_path)
		)


		description_du_soldat["identite"]["affectation"] = (
			extractions.extraire_feature(entities_list=entities,
										 lignes=lignes_description_du_soldat,
										 feature="affectation_soldat",
				image_path=image_path)
		)

		rang_extrait = (
			extractions.extraire_feature(entities_list=entities,
										 lignes=lignes_description_du_soldat,
										 feature="rang_actuel",
				image_path=image_path)
		)

		description_du_soldat["identite"]["rang"] = rang_extrait
		rangs_possibles = ["canonnier", "caporal", "soldat", "soldat de 1^ere classe", "soldat de 2^eme classe",
						   "soldat détenu", "sergent-major", "travailleur", "conducteur", "sapeur", "sergent", "clairon",
						   "maître pointeur", "caporal fourrier", "maréchal des logis", "maréchal des logis fourrier",
						   "officier d'administration de 1^ere classe", "officier d'administration de 2^eme classe",
						   "médecin auxiliaire", "sous-lieutenant", "sapeur indigène", "tirailleur", "chasseur", "détenu",
						   "médecin major de 2^eme classe", "médecin major de 1^ere classe", "zouave", "canonnier fauconnier"]
		if rang_extrait["extracted"] is not None:
			# Il manque le cas "Autre", à gérer avec une distance importante
			rang_normalise, distance = utils.find_closest_word_in_list(word_list=rangs_possibles, target_word=rang_extrait["extracted"])
			if distance > (len(rang_normalise) / 2):
				rang_normalise = "UNK"
		else:
			rang_normalise = None

		description_du_soldat["identite"]["rang"]["normalized"] = rang_normalise

		# Profession
		profession = extractions.extraire_feature(
			entities,
			lignes_description_du_soldat,
			"profession",
				image_path=image_path
		)

		description_du_soldat["profession"] = profession
		if description_du_soldat["profession"]:
			normalized, distance = utils.find_closest_word_in_list(target_word=profession["extracted"],
																   word_list="src/resources/french_professions.txt",
																   load_file=True)
			if distance > (len(normalized) / 2):
				description_du_soldat["profession"]["normalized"] = None
			else:
				description_du_soldat["profession"]["normalized"] = normalized

		else:
			description_du_soldat["profession"]["normalized"] = None


		# Description physique
		description_du_soldat["description_physique"]["marques_particulières"] = extractions.extraire_feature(
			entities,
			lignes_description_du_soldat,
			feature="marques_particulières",
				image_path=image_path
		)


		description_du_soldat["description_physique"]["renseignements_complementaires"] = extractions.extraire_feature(
			entities,
			lignes_description_du_soldat,
			feature="renseignements_complementaires",
				image_path=image_path
		)

		# Items courants
		for item in ["nez", "visage", "yeux", "front", "taille", "cheveux", "bouche", "menton"]:
			description_du_soldat["description_physique"][item] = extractions.extraire_feature(
				entities,
				lignes_description_du_soldat,
				feature=item,
				image_path=image_path
			)
			if item == "taille":
				taille_courante = description_du_soldat["description_physique"][item]["extracted"]
				if taille_courante:
					description_du_soldat["description_physique"][item]["identified"] = taille_courante
					description_du_soldat["description_physique"][item]["extracted"] = extractions.traiter_taille(
						taille_courante)
		return description_du_soldat

	def extraire_date_du_proces_p1(self,
								   ocr_prediction: OCRRecord,
								   annotations: YOLORecord,
								   image_path: str,
								   loaded_image: PIL.Image.Image = None):
		"""
		Cette fonction extrait le nom du soldat à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param ocr_prediction: Une liste de dictionnaires de la forme:
		'''
		[
			{
				'baseline': [[231, 5467], [2329, 5450]],
				'prediction': "(3) Indiquer le crime ou le délit psur lequel l'accusé a été traduit devant le Conseil de guerre (art. 140)."
			},
			...,
			{
				'baseline': [[241, 5612], [731, 5619]],
				'prediction': 'FORMULE N^o 16.'
			}
		]
		'''
		:param annotations: l'objet YOLORecord avec les zones.
		:param loaded_image: [Debug] le chemin vers l'image à afficher
		:return: Un dictionnaire de la forme:
			{
				"extracted": {
				  "surname": "Delin",
				  "forename": "",
				  "certainty": 0.5
				},
				"bbox": [
				[[225, 2137 ], [2631, 2194]],
				[[2723,2114], [3005, 2114]]
				],
				"prediction": [
				  "",
				]
			  }
		"""
		corresponding_lines, zone_magistrats = self.extract_lines_from_zone(annotations=annotations,
																			target_zone="Magistrats",
																			show_images=False,
																			loaded_image=loaded_image,
																			ocr_prediction=ocr_prediction,
																			intersect_ratio=0.1)

		# La date peut être sur 2 lignes, on vérifie qu'elles n'en sont qu'une
		cejourdui_date, _ = utils.match_line_by_substring(corresponding_lines=corresponding_lines,
														  string_to_match="CEJOURD'HUI")
		an_mil_neuf_date, _ = utils.match_line_by_substring(corresponding_lines=corresponding_lines,
															string_to_match="an mil neuf cent")
		if cejourdui_date == an_mil_neuf_date:
			correct_lines = cejourdui_date
			line_as_string = cejourdui_date.prediction
		else:
			correct_lines = [cejourdui_date, an_mil_neuf_date]
			line_as_string = f"{cejourdui_date.prediction} {an_mil_neuf_date.prediction}"
		date_span = utils.approximate_word_split(line_as_string, "CEJOURD'HUI")

		date_extraite = date_span[-1]
		baseline = utils.get_baseline_from_string(line=correct_lines,
												  target_string=date_extraite,
												  show_image=False,
												  loaded_image=loaded_image,
												  image_path=image_path)
		corrected_date = utils.correct_date(date_extraite)
		try:
			normalized = date.process_date(corrected_date)
		except TypeError:
			normalized = None
		self.date_proces = normalized
		current_date = utils.DateRecord(extracted=date_extraite,
										predicted=line_as_string,
										bbox=zone_magistrats,
										baseline=baseline,
										normalized=normalized,
										corrected=corrected_date)
		return current_date.to_json()

	def extraire_magistrats(self,
							ocr_prediction: OCRRecord,
							zones_magistrats,
							image: str = None,
							show_images: bool = True):
		"""
		Cette fonction extrait les noms des magistrats et leur statut à parti
		:param ocr_prediction: un objet OCRRecord
		:param zones_magistrats: un objet YOLORecord
		:param image: [Debug] le chemin vers l'image à afficher
		:param show_images: [Debug] afficher l'image
		:return: Un dictionnaire de la forme qui suit. l'entrée baseline peut contenir plusieurs lignes, d'où une liste
		de niveau 3: [ligne[points[coords]]]
			{
			  "president": {
				"extracted": {
				  "persName": "Delin Lieut^t Cotonel",
				  "role": " de Gendarmerie Prevot de l'armée Président,",
				  "certainty": 0.5
				},
				"baseline": [
				[[225, 2137 ], [2631, 2194]],
				[[2723,2114], [3005, 2114]]
				],
				"predictions": [
				  "Delin Lieut^t Cotonel de Gendarmerie Prevot de l'armée",
				  "Président,"
				]
			  },
			  "jures": [
				{
				  "extracted": {
					"persName": "Barbancey",
					"role": ", chefde bataillon de l'Etat Hajor de l'armée",
					"certainty": 1
				  },
				  "baseline": [
					...
				  ],
				  "predictions": [
					"Barbancey, chefde bataillon de l'Etat Hajor de l'armée"
				  ]
				},
				...,
				{
				  "extracted": {
					"persName": "Remy",
					"role": ", Maréchal des logis, du 6^e Escedron du Train",
					"certainty": 1
				  },
				  "baseline": [
					...
				  ],
				  "predictions": [
					"Remy, Maréchal des logis, du 6^e Escedron du Train"
				  ]
				}
			  ]
			}
		"""

		table_dict = {}
		zone_englobante_magistrats = self.filter_zones(zones_magistrats, "Magistrats")
		column_annotation = self.filter_zones(zones_magistrats, "Colonne")
		lines_annotation = self.filter_zones(zones_magistrats, "ligne")
		lignes_table_triees = utils.vertical_order_zones(lines_annotation)
		first_column, *_ = utils.horizontal_order_zones(column_annotation)
		first_column = first_column.coordinates
		first_column_as_rectangle = self.rectangle(first_column[0][0],
												   first_column[0][1],
												   first_column[1][0],
												   first_column[1][1])
		if show_images:
			for line in lignes_table_triees:
				loaded_image = Image.open(image)
				cropped = loaded_image.crop(line["coordinates"])
				cropped.show()

		# On itère sur les zones identifiées par YOLO
		for _, line in enumerate(lignes_table_triees):
			corresponding_box = line.coordinates
			box_as_rectangle = self.rectangle(corresponding_box[0][0],
											  corresponding_box[0][1],
											  corresponding_box[1][0],
											  corresponding_box[1][1])

			# On vérifie que la ligne nous intéresse, qu'elle se trouve sur la première colonne
			overlap_ratio_first_column = utils.check_if_overlap(first_column_as_rectangle, box_as_rectangle)
			if overlap_ratio_first_column is not None and overlap_ratio_first_column < 0.7:
				continue

			# On itère sur les lignes identifiées par Kraken
			for idx, predicted_line in enumerate(ocr_prediction):
				prediction = predicted_line.prediction
				image_path = predicted_line.image_path
				baseline = predicted_line.baseline
				# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
				converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
				is_in_box = utils.check_if_line_in_box(box_coord=box_as_rectangle, baseline=converted_baseline, intersect_ratio=0.7)
				# On veut éviter que la ligne au dessus du tableau ne soit incluse.
				test_string = utils.nfc_normalize("Code de justice militaire, de MM")
				if test_string in prediction:
					continue
				# On vérifie que la ligne est bien dans la colonne 1
				is_in_correct_column = utils.check_if_line_in_box(box_coord=first_column_as_rectangle,
																  baseline=converted_baseline)
				if is_in_box is True:
					try:
						table_dict[idx].append(
							OCRLine(prediction=prediction, baseline=baseline, cuts=None, polygon=None, image_path=image_path)
						)

					except KeyError:
						table_dict[idx] = [
							OCRLine(prediction=prediction, baseline=baseline, cuts=None, polygon=None, image_path=image_path)
						]
		table_des_magistrats = [item for item in table_dict.values()]
		utils.log_print(table_des_magistrats)
		# On récupère les informations, en sachant que le premier est toujours le président
		# TODO: on peut vérifier la présence du mot `président` dans la ligne transcrite
		president = table_des_magistrats[0]
		jures = table_des_magistrats[1:]
		processed_jures = []

		# On va itérer jury par jury
		for jure in jures:
			jury_extrait = extractions.extraire_nom_et_fonction(
				prediction=" ".join(line.prediction for line in jure),
				pipeline=self.date_recognition_pipeline)
			jury_extrait['baseline'] = [line.baseline for line in jure]
			jury_extrait['prediction'] = " ".join([line.prediction for line in jure])
			utils.log_print(jury_extrait['persName'])
			if (jury_extrait['persName'] == "UNK" and (utils.similarite_ratcliff("Président",
																				" ".join(line.prediction for line in
																						 jure)) > .7) \
													  or utils.similarite_ratcliff("Juges", jury_extrait['persName']) > .7):
				utils.log_print(" ".join(line.prediction for line in jure))
				utils.log_print("ÉCARTÉ")
				continue
			jury_dict = {"extracted": jury_extrait}
			processed_jures.append(jury_dict)
		if len(processed_jures) < 4:
			utils.log_print("Warning: il manque un membre du jury")
			processed_jures.append("Juré manquant")
		processed_president = extractions.extraire_nom_et_fonction(" ".join(line.prediction for line in president),
																   self.date_recognition_pipeline)

		# On travaille sur les trois derniers noms: le gradé qui nomme le jury, le commissaire, le greffier.
		coords_zone_englobante_magistrats = zone_englobante_magistrats[0].coordinates
		zone_magistrat_as_rectangle = self.rectangle(coords_zone_englobante_magistrats[0][0],
													 coords_zone_englobante_magistrats[0][1],
													 coords_zone_englobante_magistrats[1][0],
													 coords_zone_englobante_magistrats[1][1])
		lignes_zone_magistrat = []
		for predicted_line in ocr_prediction:
			baseline = predicted_line.baseline
			# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
			converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
			is_in_box = utils.check_if_line_in_box(box_coord=zone_magistrat_as_rectangle, baseline=converted_baseline)
			if is_in_box is True:
				lignes_zone_magistrat.append(predicted_line)

		# On extrait le nom du greffier:
		greffier = extractions.extraire_greffier(lignes_zone_magistrat=lignes_zone_magistrat,
												 image_path=image_path,
												 ner_pipeline=self.date_recognition_pipeline)
		commissaire = extractions.extraire_commissaire(lignes_zone_magistrat=lignes_zone_magistrat,
													   ner_pipeline=self.date_recognition_pipeline,
													   image_path=image_path)
		general = extractions.extraire_general(lignes_zone_magistrat=lignes_zone_magistrat,
											   image_path=image_path)
		# Si on ne trouve rien, c'est que la ligne est hors de la boîte. On relance sur l'ensemble des lignes.
		if general == {"grade": None}:
			general = extractions.extraire_general(lignes_zone_magistrat=ocr_prediction, image_path=image_path)
		utils.log_print("Magistrats: OK")
		return {"president": {"extracted": processed_president,
							  "baseline": [line.baseline for line in president],
							  "predictions": [line.prediction for line in president]},
				"jures": processed_jures,
				"greffier": greffier,
				"commissaire": commissaire,
				"general": general}
