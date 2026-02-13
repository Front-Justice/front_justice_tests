###############

## Script d'extraction à partir des segmentations. À lier avec le script "segmentation_kraken_yolo.

###############
import unicodedata
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import glob
import re
from src.utils.utils import OCRRecord, YOLOZone
from src.utils.utils import OCRLine
from src.utils.utils import YOLORecord
import src.utils.utils as utils
import copy
import PIL.Image as Image
import src.Information_Extractor.semantic_search as search
import PIL
from collections import namedtuple
# import src.Vision.PARTY as PARTY
import src.date.parse_date as date
import fuzzysearch
import src.Information_Extractor.extraction_functions as extractions


class Extractor:
	"""
	Classe pour extraire les informations à partir:
	 	- d'un corpus d'annotations au format COCO
	 	- d'un corpus de documents XML au format ALTO
	 	- du même corpus d'images
	"""

	def __init__(self, party_engine,
				 resize_factor: int = 1,
				 debug: bool = False,
				 use_party=True):
		"""
		Constructeur de la classe Extractor
		:param party_engine: le moteur party (instance de classe PARTY.PartyPredict)
		:param resize_factor: le facteur de redimension des images (accélère l'ocr)
		:param debug: Active le mode debug, ne charge pas le modèle party (va créer des erreurs)
		"""

		# On initialise une pipeline de NER avec un modèle camembert adapté
		self.date_proces = ""
		self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
		self.ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
		self.ner = pipeline('ner',
							model=self.ner_model,
							tokenizer=self.tokenizer,
							aggregation_strategy="simple",
							device="cpu")

		entity_spotting_model = ("/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/scripts/src/Information_Extractor/models/model_NER")

		entity_spotting_model = ("/media/mgl/stock/Front_Justice/NER_training/BERT-NER-CoNLL/BERTNER/results_test_v3/best_model")
		tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
		self.entity_spotting_pipeline = pipeline('ner',
												 model=entity_spotting_model,
												 tokenizer=tokenizer,
												 aggregation_strategy="simple",
												 device="cuda:0")

		self.alto_namepaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		self.target_corpus = glob.glob("../Page_Classifier/data/corpus/page_1/*.jpg")
		self.conversion_dict = {item.replace("(", "-").replace(")", "").split("/")[-1].replace(".jpg", ""): item for
								item in
								self.target_corpus}
		self.resize_factor: int = resize_factor
		self.use_party = use_party
		if debug is False:
			self.party = party_engine

		self.rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

		# On récupère les images en grande taille
		self.extracted_annotations = {}
		self.excluded_classes = ["Titre"]

	def filter_zones(self, annotations: YOLORecord, category: str) -> YOLORecord:
		"""
		Fonction permettant de filtrer les zones par catégorie
		:param annotations: Une liste de dictionnaires de la forme:
		[
			{'label': 'Magistrats', 'coordinates': [113, 1362, 3038, 3235]},
			...,
			{'label': 'Table', 'coordinates': [195, 2039, 3034, 2863]}
		]
		:param category: la catégorie à filtrer
		:return: La même liste avec la zone ciblée
		"""
		return [annotation for annotation in annotations if annotation.label == category]



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
				print(f"Erreur: plusieurs zones détectées. Target zone: {target_zone}")
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
		:param party_engine: le moteur de transcription party
		:param annotations: un objet YOLORecord qui contient les coordonnées et labels de zone
		:param ocr_prediction: un objet OCRRecord qui contient les baselines, predictions et cuts d'une liste de lignes
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

	def extraire_date_naissance_soldat_p2(self,
										  ocr_prediction: OCRRecord,
										  annotations: YOLORecord,
										  image: str = None,
										  loaded_image: PIL.Image.Image = None,
										  show_images: bool = True):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
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
		extracted_lines, zone_nom_soldat = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["Nom du soldat",
																					 "identite_soldat"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[0.05, .8])
		try:
			ligne_nom_soldat = extracted_lines[0]
			nom_soldat = utils.extract_string_from_cuts(box=zone_nom_soldat[0],
														line=ligne_nom_soldat)
		except (TypeError, IndexError):
			target_string = "a repondu se nommer"
			ligne_nom_soldat, _ = utils.match_line_by_substring(corresponding_lines=ocr_prediction,
																string_to_match=target_string)
			name_split, matching_word = utils.approximate_word_split(word="nommer",
																	 sentence=ligne_nom_soldat.prediction,
																	 return_word=True)
			after_target_string = name_split[-1]
			nom_soldat = after_target_string.split()[0]
			entities = self.ner(after_target_string)
			if nom_soldat == entities[0]['word']:
				certainty = 1
			else:
				certainty = 0.5
		nom_soldat = utils.clean_small_string(nom_soldat)

		prenoms, certitude_prenoms = utils.extraction_prenom_du_soldat(ligne_nom_soldat.prediction,
																	   nom_soldat,
																	   pipeline=self.ner)
		range_prenom_debut = ligne_nom_soldat.prediction.find(nom_soldat)
		range_prenom_fin = ligne_nom_soldat.prediction.find(prenoms) + len(prenoms)
		nom_complet = ligne_nom_soldat.prediction[range_prenom_debut:range_prenom_fin]
		baseline_nom_complet = utils.get_baseline_from_string(line=ligne_nom_soldat,
															  target_string=nom_complet,
															  loaded_image=loaded_image,
															  show_image=True)

		return {"extracted":
					{"forename": {"persName": prenoms,
								  "certainty": certitude_prenoms},
					 "surname": {"persName": nom_soldat,
								 "certainty": certainty},
					 },
				"bbox": zone_nom_soldat,
				"baseline": baseline_nom_complet
				}


	def extraire_identite_soldat_p2(self,
									ocr_prediction: OCRRecord,
									annotations: YOLORecord,
									image: str = None,
									loaded_image: PIL.Image.Image = None,
									show_images: bool = True):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
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
		lignes_identite_soldat, zone_identite_soldat = self.extract_lines_from_zone(annotations=annotations,
																					target_zone=["identite_soldat"],
																					show_images=False,
																					loaded_image=loaded_image,
																					ocr_prediction=ocr_prediction,
																					intersect_ratio=[.8])
		extracted_lines, zone_nom_soldat = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["Nom du soldat",
																					 "identite_soldat"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[0.05, .8])
		try:
			if len(extracted_lines) == 1:
				ligne_nom_soldat = extracted_lines[0]
			else:
				ligne_nom_soldat = extracted_lines[1]
			nom_soldat = utils.extract_string_from_cuts(box=zone_nom_soldat[0],
														line=ligne_nom_soldat)
		except (TypeError, IndexError):
			target_string = "a repondu se nommer"
			ligne_nom_soldat, _ = utils.match_line_by_substring(corresponding_lines=ocr_prediction,
																string_to_match=target_string)
			name_split, matching_word = utils.approximate_word_split(word="nommer",
																	 sentence=ligne_nom_soldat.prediction,
																	 return_word=True)

			after_target_string = name_split[-1]
			after_target_string = utils.clean_small_string(after_target_string)
			print(f"|{after_target_string}|")
			print(ligne_nom_soldat.prediction)
			try:
				nom_soldat = after_target_string.split()[0]
			except IndexError:
				nom_soldat = None
		entities = self.ner(ligne_nom_soldat.prediction)
		if len(entities) != 0:
			if nom_soldat == entities[0]['word']:
				certainty = 0.8
			elif nom_soldat is None:
				certainty = None
			else:
				certainty = 0.5
		else:
			certainty = 0.5
		nom_soldat = utils.clean_small_string(nom_soldat)
		print(nom_soldat)
		if not nom_soldat:
			prenoms, certitude_prenoms, nom_complet, baseline_nom_complet = None, None, None, None
		else:
			prenoms, certitude_prenoms = utils.extraction_prenom_du_soldat(ligne_nom_soldat.prediction,
																		   nom_soldat,
																		   pipeline=self.ner)
			try:
				range_prenom_debut = ligne_nom_soldat.prediction.find(nom_soldat)
			except TypeError:
				range_prenom_debut = None
			try:
				range_prenom_fin = ligne_nom_soldat.prediction.find(prenoms) + len(prenoms)
				nom_complet = ligne_nom_soldat.prediction[range_prenom_debut:range_prenom_fin]
				baseline_nom_complet = utils.get_baseline_from_string(line=ligne_nom_soldat,
																	  target_string=nom_complet,
																	  loaded_image=loaded_image,
																	  show_image=False)
			except TypeError:
				nom_complet = None
				baseline_nom_complet = None
		try:
			dictionnary['age_du_soldat'] = extractions.extraire_age_soldat(
				lignes_identite_soldat=lignes_identite_soldat)
		except TypeError:
			dictionnary['age_du_soldat'] = None

		# Lieu de naissance
		naissance = utils.nfc_normalize("né à")
		if lignes_identite_soldat is None:
			dictionnary['nom_du_soldat'] = None
			return dictionnary
		lignes_identite_soldat_string = lignes_identite_soldat.join_transcription()
		splits = utils.approximate_sentence_split(sentence=utils.nfc_normalize(lignes_identite_soldat_string),
												  substring=naissance,
												  max_dist=1)
		if not splits:
			dictionnary['lieu_naissance'] = {"prediction": lignes_identite_soldat_string,
											 "extracted": "UNK"}
		else:
			splits = splits[-1]
			ner = self.ner(splits)
			first_entity = ner[0]
			if first_entity['entity_group'] == 'LOC' and first_entity['start'] == 0:
				lieu_naissance = first_entity['word']
				certainty_naissance = 0.8
			else:
				lieu_naissance = splits.split()[0]
				certainty_naissance = 0.5

			dictionnary['lieu_naissance'] = {"prediction": splits,
											 "extracted": lieu_naissance,
											 "certainty": certainty_naissance}
		dictionnary['prediction'] = lignes_identite_soldat_string
		dictionnary['nom_du_soldat'] = {"extracted":
											{"forename": {"persName": prenoms,
														  "certainty": certitude_prenoms},
											 "surname": {"persName": nom_soldat,
														 "certainty": certainty},
											 },
										"bbox": zone_nom_soldat,
										"baseline": baseline_nom_complet,
										"prediction": nom_complet
										}

		return dictionnary

	def extraire_description_soldat_p2_NER(self,
										   ocr_prediction: OCRRecord,
										   annotations: YOLORecord,
										   image: str = None,
										   loaded_image: PIL.Image.Image = None,
										   show_images: bool = True):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
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

		description_du_soldat = {}
		lignes_description_du_soldat, zone_identite_soldat = self.extract_lines_from_zone(annotations=annotations,
																					target_zone=["identite_soldat"],
																					show_images=False,
																					loaded_image=loaded_image,
																					ocr_prediction=ocr_prediction,
																					intersect_ratio=[.8])
		soldat: list[YOLOZone] = annotations.filter_zones("Nom du soldat")
		lignes_description_soldat_raw = lignes_description_du_soldat.join_transcription()
		result_spotting = self.entity_spotting_pipeline(lignes_description_soldat_raw)
		entities_as_dictionnary = utils.entities_to_dict(result_spotting)
		print(lignes_description_soldat_raw)
		print(entities_as_dictionnary)
		plusieurs_soldats = False
		if len(soldat) == 1:
			bbox_nom_soldat = soldat[0].coordinates
			entite_et_baseline = extractions.extraire_entite_baseline(
				dictionnaire=entities_as_dictionnary,
				nom_entite="nom_du_soldat",
				target_lines=lignes_description_du_soldat
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
			plusieurs_soldats = True
			bbox_nom_soldat = None
			print("Plusieurs soldats.")
		else:
			bbox_nom_soldat = None
			print("Aucun soldat identifié par YOLO.")

		description_du_soldat["prediction"] = lignes_description_soldat_raw

		description_du_soldat["identite"] = \
			{
				"prenom": extractions.extraire_feature(entities_as_dictionnary,
													   lignes_description_du_soldat,
													   "prénom_du_soldat"),
				"nom": extractions.extraire_feature(entities_as_dictionnary,
													lignes_description_du_soldat,
													"nom_du_soldat")
			}

		description_du_soldat["identite"]["date_naissance"] = (
			extractions.extraire_date_naissance(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)

		description_du_soldat["identite"]["lieu_naissance"] = (
			extractions.extraire_lieu_naissance(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)

		description_du_soldat["identite"]["lieu_residence"] = (
			extractions.extraire_lieu_residence(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)

		description_du_soldat["identite"]["situation_maritale"] = (
			extractions.extraire_sit_maritale(entity_dict=entities_as_dictionnary,
											  lignes=lignes_description_du_soldat)
		)

		description_du_soldat["identite"]["age"] = (
			extractions.extraire_feature(entities_as_dictionnary=entities_as_dictionnary,
										 lignes=lignes_description_du_soldat,
										 feature="âge")
		)
		print(entities_as_dictionnary)
		print(lignes_description_du_soldat)
		description_du_soldat["identite"]["affectation"] = (
			extractions.extraire_feature(entities_as_dictionnary=entities_as_dictionnary,
										 lignes=lignes_description_du_soldat,
										 feature="affectation_soldat")
		)

		description_du_soldat["identite"]["rang"] = (
			extractions.extraire_feature(entities_as_dictionnary=entities_as_dictionnary,
										 lignes=lignes_description_du_soldat,
										 feature="rang_actuel")
		)

		# Profession
		description_du_soldat["profession"] = extractions.extraire_feature(
			entities_as_dictionnary,
			lignes_description_du_soldat,
			"profession"
		)

		return description_du_soldat

	def extraire_identite_defenseur(self,
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
		entites_nommees = self.ner(apres_defenseur)
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
		print(matching_line_defenseur)
		baseline_nom_defenseur = utils.get_baseline_from_string(line=matching_line_defenseur,
																target_string=nom_defenseur,
																loaded_image=loaded_image,
																show_image=False)

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
		print(lignes_formalites_as_string)
		if similarite > .8:
			precisions_jugement_temoins_defense = None
		else:
			try:
				precisions_jugement_temoins_defense = \
					utils.approximate_sentence_split(sentence=lignes_formalites_as_string,
													 substring="et 319 du Code d'instruction criminelle")[-1]
			except TypeError:
				precisions_jugement_temoins_defense = None
		print(precisions_jugement_temoins_defense)

		return {"prediction": lignes_formalites_as_string,
				"extracted": precisions_jugement_temoins_defense,
				"bbox": zone_formalites}


	def extraire_questions_p2(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
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
			split, match = utils.approximate_sentence_split(sentence=prediction_as_string,
															substring=" Le Conseil délibérant à huis clos, le Président",
															return_match=True)
			target = f"{match} {split[-1]}"
			notes_bas_page_1 = "(1) et à décharge (s'il y en a)"
			notes_bas_page_2 = "(2) Indiquer si des témoins ont été entendus"
			split_bas_page_1 = utils.approximate_sentence_split(sentence=target, substring=notes_bas_page_1)
			if not split_bas_page_1:
				split_bas_page_2 = utils.approximate_sentence_split(sentence=target, substring=notes_bas_page_2)
				target = split_bas_page_2[0]
				return {"prediction": target,
						"extracted": None,
						"bbox": None}
			else:
				target = split_bas_page_1[0]
				return {"prediction": target,
						"extracted": None,
						"bbox": None}
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
		print(lignes_questions_as_string)
		if similarite > .8:
			questions = None
		else:
			try:
				questions = \
					utils.approximate_sentence_split(sentence=lignes_questions_as_string,
													 substring="ainsi qu'il suit:")[-1]
			except TypeError:
				questions = None
		print(questions)

		return {"prediction": lignes_questions_as_string,
				"extracted": questions,
				"bbox": zone_questions}




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




	def extraire_tableau_p4(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
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

		lignes_tableau_as_string = lignes_tableau.join_transcription(merge_newlines=False)


		return {"prediction": lignes_tableau_as_string,
				"extracted": lignes_tableau_as_string,
				"bbox": zone_tableau}


	def extraire_paragraphe_final_p4(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
							  loaded_image: PIL.Image.Image = None) -> dict:
		"""
		Cette fonction extrait les questions posées par le Président à partir des prédictions et des zones.
		:param ocr_prediction: un objet OCRRecord
		:param annotations: un objet YOLORecord
		:param loaded_image: l'image chargée (objet PIL.Image.Image)
		"""

		lignes_tableau, zone_tableau = self.extract_lines_from_zone(annotations=annotations,
																		target_zone=["recapitulatif_somme"],
																		show_images=False,
																		loaded_image=loaded_image,
																		ocr_prediction=ocr_prediction,
																		intersect_ratio=[.5],
																		select_highest_prob_zone=True)

		lignes_tableau_as_string = lignes_tableau.join_transcription()


		return {"prediction": lignes_tableau_as_string,
				"extracted": lignes_tableau_as_string,
				"bbox": zone_tableau}



	def extraire_decision_tribunal_p3(self,
							  ocr_prediction: OCRRecord,
							  annotations: YOLORecord,
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
		try:
			lignes_decision_as_string = lignes_decision.join_transcription()
		except AttributeError:
			return {"prediction": None,
					"extracted": None,
					"bbox": zone_decision}


		return {"prediction": lignes_decision_as_string,
				"extracted": lignes_decision_as_string,
				"bbox": zone_decision}




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
		print(lignes_requisitoire_as_string)
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
					"Date normalisee": None,
					"Date": None,
					"predictions": None}

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
			return {"date_normalisee": None,
					"Date corrigée": None,
					"Date retenue": date_crime_party,
					"baseline": target_line.baseline,
					"bbox": date_zone,
					"certitude": certitude,
					"predictions": {"party": date_crime_party,
									"kraken": date_crime_kraken}}
		try:
			normalized_date = date.process_date(corrected_date, debug=False)
		except TypeError:
			normalized_date = None

		return {"date_normalisee": normalized_date,
				"Date corrigée": corrected_date,
				"Date retenue": date_crime_party,
				"baseline": target_line.baseline,
				"bbox": date_zone,
				"certitude": certitude,
				"predictions": {"party": date_crime_party,
								"kraken": date_crime_kraken}}

	def extraire_informations_ajouts_posterieurs(self,
												 ocr_prediction: OCRRecord,
												 annotations: YOLORecord):
		"""
		Cette fonction extrait la date et classe le type d'information d'un ajout du greffier. Il y a en effet toujours une date !
		:param ocr_prediction:
		:param annotations:
		:return:
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
			print(f"La glose fait {len(as_record)} lignes.")
			lignes_fusionnees = as_record.join_transcription()
			print(lignes_fusionnees)
			resultat = self.ner(lignes_fusionnees)
			try:
				extracted_date = next(item for item in resultat if item['entity_group'] == 'DATE')['word']
			except StopIteration:
				extracted_date = None
			print(resultat)
			print(extracted_date)
			if extracted_date is not None:
				try:
					normalized_date = date.process_date(extracted_date, debug=False)
				except TypeError:
					normalized_date = None
				print(normalized_date)
			else:
				normalized_date = None
			list_of_informations = [
				"Remise du restant de la peine",
				"Décès du soldat",
				"Amnistie",
				"Peine effectuée",
				"Jugement suspendu",
				"Exécution de la peine suspendue",
				"Peine commuée"
			]
			information_contenue = search.retrieve_most_similar_sentence(sentence=lignes_fusionnees,
																		 queries=list_of_informations)
			print(information_contenue)

			# TODO: cas où il y a plusieurs annotations différentes
			list_of_results.append({
				"date": normalized_date,
				"information": information_contenue,
				"prediction": lignes_fusionnees,
				"bbox": annotations[0].coordinates
			})
		return list_of_results



	def extraire_inculpation_et_antecedents(self,
											ocr_prediction: OCRRecord,
											annotations: YOLORecord,
											image: str = None,
											loaded_image: PIL.Image.Image = None,
											show_images: bool = True):
		"""
		Cette fonction extrait le numéro d'ordre à partir des prédictions et des zones.
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
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""
		inculpation = {"antécédents": {},
					   "inculpation": {}}
		corresponding_lines, numero_ordre_zone = self.extract_lines_from_zone(annotations=annotations,
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
		mot_condamnations = mot_condamnations[0].matched
		inculpation["antécédents"]["predicted"] = lignes_condamnations_str

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

			all_dates = [item for item in self.ner(lignes_condamnations_str) if item['entity_group'] == "DATE"]
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
							  image: str = None,
							  loaded_image: PIL.Image.Image = None,
							  show_images: bool = True):
		"""
		Cette fonction extrait le numéro d'ordre à partir des prédictions et des zones.
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
		corresponding_lines, numero_ordre_zone = self.extract_lines_from_zone(annotations=annotations,
																			  target_zone="MainZone-orderNumber",
																			  show_images=show_images,
																			  loaded_image=loaded_image,
																			  ocr_prediction=ocr_prediction,
																			  intersect_ratio=0.7)
		if (corresponding_lines, numero_ordre_zone) == (None, None):
			# TODO: reprendre cela, ça semble bizarre.
			print("Error")
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
			print("Formulaire n°967 trouvé, on continue.")
		elif numero_nomenclature == "974":
			print("Le formulaire 974 (bis) a été identifié: révision de procès. Le processus s'arrête pour l'instant.")
			exit(0)
		else:
			print("Le numéro de formulaire n'est pas trouvé. Il peut s'agir d'une erreur d'OCR "
				  "ou de classification de la page.")

		target_line = []
		for line in corresponding_lines:
			prediction = line.prediction
			similarity = utils.similarite_ratcliff(prediction, "D'ORDRE.")

			# On condidère une valeur de similarité de 0.5, à modifier par l'expérience
			if similarity > .5:
				target_line.append(line)

		if len(target_line) != 1:
			print(f"Erreur. Zéro ou  Plusieurs lignes trouvées pour le numéro d'ordre:\n"
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

		return {"extracted": target_number,
				"baseline": target_line.baseline,
				"bbox": numero_ordre_zone,
				"certitude": certitude,
				"predictions": {"party": numero_ordre_party,
								"kraken": target_line.prediction}}

	def extraire_description_soldat(self,
									ocr_prediction: OCRRecord,
									annotations: YOLORecord,
									image: str = None,
									loaded_image: PIL.Image.Image = None,
									show_images: bool = False):
		"""
		Cette fonction extrait la description du soldat à partir des prédictions et des zones.
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
		:param zones_nom: une liste de dictionnaires de la forme:
		[
			{
				'label': 'nom_du_soldat',
				'coordinates': [[212, 2400], [2735, 2551]]
			}
		]
		:param image: [Debug] le chemin vers l'image à afficher
		:param show_images: [Debug] afficher l'image
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

		description_du_soldat = {}

		lignes_description_du_soldat, soldat_zone = self.extract_lines_from_zone(annotations=annotations,
																				 target_zone="Description du Soldat",
																				 show_images=show_images,
																				 loaded_image=loaded_image,
																				 ocr_prediction=ocr_prediction,
																				 intersect_ratio=0.1)
		lignes_description_soldat_raw = lignes_description_du_soldat.join_transcription()
		try:
			target_lines = utils.approximate_sentence_split(sentence=lignes_description_soldat_raw,
															substring="A l'effet de juger")[-1]
		except TypeError:
			target_lines = lignes_description_du_soldat

		# On extrait d'abord le nom du soldat
		lignes_nom_du_soldat, soldat_zone = self.extract_lines_from_zone(annotations=annotations,
																		 target_zone="Nom du soldat",
																		 show_images=show_images,
																		 loaded_image=loaded_image,
																		 ocr_prediction=ocr_prediction,
																		 intersect_ratio=0.05)
		if lignes_nom_du_soldat is None:
			print("Plusieurs soldats trouvés, pas encore pris en charge.")

			lignes_description_du_soldat, soldat_zone = self.extract_lines_from_zone(annotations=annotations,
																					 target_zone="Description du Soldat",
																					 show_images=show_images,
																					 loaded_image=loaded_image,
																					 ocr_prediction=ocr_prediction,
																					 intersect_ratio=0.1)
			target_lines = lignes_description_du_soldat.join_transcription()
			target_lines = utils.approximate_sentence_split(sentence=target_lines,
															substring="A l'effet de juger")[-1]
			return {"nom_du_soldat": "Plusieurs soldats",
					"Description du soldat": "Plusieurs soldats",
					"target_text": target_lines}

		# On peut avoir plusieurs lignes, car le nom est écrit en gros module. On va
		# Donc tester la distance au début de la ligne qui commence par "A l'effet de juger"
		# TODO: Transformer ça en fonction pour réutilisation à d'autres endroits
		name_line, _ = utils.match_line_by_substring(corresponding_lines=lignes_nom_du_soldat,
													 string_to_match="A l'effet de juger")

		# On a la ligne correspondante. Maintenant, on va identifier les caractères
		# qui sont compris dans la boîte à l'aide des cuts de kraken. On a besoin de tout convertir
		# en polygones, pour ensuite vérifier les intersections entre la boîte et les intersections
		nom_du_soldat_kraken = utils.extract_string_from_cuts(box=soldat_zone, line=name_line).strip()

		prenoms, certitude_prenoms = utils.extraction_prenom_du_soldat(name_line.prediction,
																	   nom_du_soldat_kraken,
																	   pipeline=self.ner)
		if show_images:
			cropped = loaded_image.crop(
				(
					round(soldat_zone[0][0] / self.resize_factor),
					round(soldat_zone[0][1] / self.resize_factor),
					round(soldat_zone[1][0] / self.resize_factor),
					round(soldat_zone[1][1] / self.resize_factor)
				)
			)
			cropped.show()

		# On prédit à l'aide de party la section de baseline qui correspond à la boite identifiée par Yolo
		baseline_soldat = name_line.baseline
		prediction_soldat = name_line.prediction

		# On prend le premier et le dernier point de la ligne
		# TODO: ajouter la ligne et lq baseline.
		baseline_soldat = [baseline_soldat[0], baseline_soldat[-1]]
		(x1_soldat, _), (x2_soldat, _) = soldat_zone
		(_, y1), (_, y2) = baseline_soldat
		baseline_soldat_coupee = [
			[round(x1_soldat / self.resize_factor), round(y1 / self.resize_factor)],
			[round(x2_soldat / self.resize_factor), round(y2 / self.resize_factor)]
		]

		if self.use_party:
			party_segmentation = self.party.create_baseline([baseline_soldat_coupee], image)
			party_prediction = self.party.timed_party_inference(
				segmentation=party_segmentation,
				image=loaded_image,
				objet_transcrit="nom du soldat")
			nom_soldat_party = party_prediction.prediction
		else:
			nom_soldat_party = nom_du_soldat_kraken

		# on produit le dictionnaire
		nom_du_soldat = {}
		nom_du_soldat["extracted"] = {}
		nom_du_soldat["extracted"]["forename"] = {"persName": prenoms,
												  "certainty": certitude_prenoms}
		nom_du_soldat["baseline"] = baseline_soldat_coupee
		nom_du_soldat["bbox"] = soldat_zone
		nom_du_soldat["prediction"] = prediction_soldat
		if nom_du_soldat_kraken == nom_soldat_party:
			nom_du_soldat["extracted"]["surname"] = {"persName": nom_soldat_party,
													 "certainty": 1}
		else:
			nom_du_soldat["extracted"]["surname"] = {
				"persName": nom_soldat_party if nom_soldat_party != "+" else nom_du_soldat_kraken,
				"certainty": 0.5,
				"predictions": {"kraken": nom_du_soldat_kraken,
								"party": nom_soldat_party}
			}

		description_du_soldat["nom_du_soldat"] = nom_du_soldat

		# On passe à l'extraction de toutes les autres infos
		lignes_description_du_soldat = utils.vertical_order_lines(lines=lignes_description_du_soldat)
		effet_de_juger_line, debug = utils.match_line_by_substring(corresponding_lines=lignes_description_du_soldat,
																   string_to_match="A l'effet de juger le")
		first_line_index = lignes_description_du_soldat.index(effet_de_juger_line)
		filtered_lines = lignes_description_du_soldat[first_line_index:]

		# On commence par la date de naissance
		# TODO: en faire une fonction
		lignes_description_as_string = utils.nfc_normalize(" ".join([item.prediction for item in filtered_lines]))

		try:
			date_naissance_ner = \
				[item['word'] for item in self.ner(lignes_description_as_string) if item['entity_group'] == 'DATE'][0]
		except IndexError:
			print(f"Not found: {debug}")
			print([item.prediction for item in lignes_description_du_soldat])
			print([item.prediction for item in filtered_lines])
			exit(0)
		regexp = re.compile("[^\d]*(\d*.*\d+)[^\d]*")
		date_naissance_extraite = re.sub(regexp, '\g<1>', date_naissance_ner)
		date_naissance_corrigee = utils.correct_date(date_naissance_extraite)
		try:
			date_normalisee = date.process_date(date_naissance_corrigee)
		except TypeError:
			print(f"Error with date {date_naissance_ner}")
			date_naissance = {"date_normalisee": "Échec",
							  "Date naissance extraite": date_naissance_extraite,
							  "Date corrigée": date_naissance_corrigee,
							  "prediction": lignes_description_as_string}
			description_du_soldat["date_de_naissance"] = date_naissance
			date_normalisee = None
		try:
			age = utils.calcule_age(date_normalisee['when'], date_proces=self.date_proces['when'])
		except TypeError:
			age = "UNK"
		date_naissance = {"date_normalisee": date_normalisee,
						  "Date naissance extraite": date_naissance_extraite,
						  "Date corrigée": date_naissance_corrigee,
						  "prediction": lignes_description_as_string}
		description_du_soldat["date_de_naissance"] = date_naissance
		description_du_soldat["age"] = age

		# On s'occupe ensuite de la situation maritale
		try:
			apres_date_naissance = lignes_description_as_string.split(date_naissance_ner)[-1]
		except IndexError:
			apres_date_naissance = lignes_description_as_string
		try:
			apres_profession = utils.approximate_word_split(apres_date_naissance, "profession", sensibility=0.85)[-1]
			print(apres_profession)
			profession_et_situation_maritale = \
				utils.approximate_word_split(apres_profession.lower(), "residant", sensibility=0.8)[0]
			(check_veuf,
			 token_veuf,
			 check_celibataire,
			 celibataire,
			 token_celibataire,
			 marie,
			 token_marie,
			 nombre_enfants) = extractions.extraire_situation_maritale(
				profession_et_situation_maritale)

			# Si on infère le célibat, il n'a peut être pas été trouvé. On va chercher en ampliant la zone à l'ensemble des lignes.
			if celibataire is True and check_celibataire is False:
				print("La situation maritale n'a peut être pas été identifiée. On élargit la zone de recherche.")
				check_veuf, token_veuf, _, celibataire, token_celibataire, marie, token_marie, nombre_enfants = extractions.extraire_situation_maritale(
					lignes_description_as_string.split(date_naissance_extraite)[-1])
			if nombre_enfants:
				enfants = True
				try:
					nombre_enfants = int(nombre_enfants)
				except ValueError:
					pass
			else:
				enfants = False
			# Si on trouve des enfants mais pas la chaîne correspondant à "marié"
			if nombre_enfants and marie is False:
				marie = "Probablement"

			situation_maritale = {"marié": marie,
								  "veuf": check_veuf,
								  "enfants": enfants,
								  "nombre_enfants": nombre_enfants,
								  "célibataire": celibataire}

			description_du_soldat["situation_maritale"] = situation_maritale
			de_regexp = re.compile(r"^d['ue]\s*")
			cleaned_profession = re.sub(de_regexp, "", profession_et_situation_maritale.strip())
			if marie is True and isinstance(token_marie, str):
				cleaned_profession = cleaned_profession.split(token_marie)[0]
			elif celibataire is True and isinstance(token_celibataire, str):
				cleaned_profession = utils.strip_punctuation(cleaned_profession.split(token_celibataire)[0].strip())
			corrected_profession = utils.full_clean_string(cleaned_profession)
			matching_line_profession = utils.match_line_by_substring(filtered_lines, cleaned_profession)[0]
			description_du_soldat["profession"] = {
				"matching": cleaned_profession,
				"extracted": corrected_profession,
				"baseline": utils.get_baseline_from_string(matching_line_profession, cleaned_profession)
			}
		except TypeError:
			description_du_soldat["situation_maritale"] = None
			description_du_soldat["profession"] = None
			celibataire = None

		# Le rang du soldat se trouve juste avant le nom
		apres_effet_de_juger = utils.split_after_keep_delimiter(name_line.prediction, "juger")[-1]
		target_text = utils.split_after_keep_delimiter(lignes_description_soldat_raw, "juger")[-1]
		description_du_soldat["prediction"] = lignes_description_soldat_raw
		description_du_soldat["target_text"] = target_text
		try:
			rang = apres_effet_de_juger.split(nom_du_soldat_kraken)[0].replace("le", "").strip()
			description_du_soldat['rang'] = {
				"prediction": rang,
				"extracted": rang,
				"baseline": name_line.baseline
			}
		except ValueError:
			description_du_soldat['rang'] = {
				"prediction": apres_effet_de_juger,
				"extracted": None,
				"baseline": name_line.baseline
			}

		# Le lieu de naissance est après la date de naissance
		informations_naissance = {}
		apres_date_naissance = \
			utils.split_after_keep_delimiter(lignes_description_as_string, date_naissance_extraite.split()[-1])[1]
		split_lieu_naissance = utils.approximate_word_split(apres_date_naissance, "profession", sensibility=0.90)
		if celibataire is True and isinstance(token_celibataire, str):
			lieu_naissance = split_lieu_naissance[0].replace(token_celibataire, "")
		else:
			try:
				lieu_naissance = split_lieu_naissance[0]
			except TypeError:
				return None
		lieu_naissance_corrige = utils.strip_punctuation(lieu_naissance)
		informations_naissance = extractions.extraction_geographique(lieu_naissance_corrige, informations_naissance,
																	 self.ner)

		informations_naissance["prediction"] = lieu_naissance
		description_du_soldat["lieu_naissance"] = informations_naissance

		informations_residence = {}
		split_residence = utils.approximate_word_split(split_lieu_naissance[-1], "service", sensibility=.75)
		if split_residence is None:
			split_residence = utils.approximate_word_split(split_lieu_naissance[-1], "mobilisation", sensibility=.75)
		try:
			split_taille = utils.approximate_word_split(split_residence[-1], "Taille", sensibility=0.85)
			lieu_residence = split_taille[0]
		except TypeError:
			# On agrandit al recherche si cela ne donne rien.
			split_taille = utils.approximate_word_split(split_lieu_naissance[-1], "Taille", sensibility=0.85)
			lieu_residence = split_taille[0]
		informations_residence = extractions.extraction_geographique(lieu_residence, informations_residence, self.ner)
		informations_residence["prediction"] = lieu_residence
		description_du_soldat["lieu_residence"] = informations_residence

		# On s'occupe maintenant de la description physique du soldat
		_, _, index_ligne_taille = utils.match_line_by_substring(corresponding_lines=lignes_description_du_soldat,
																 string_to_match="taille d'", return_index=True)
		lignes_description_physique = lignes_description_du_soldat[index_ligne_taille:]

		# On extrait le numéro de matricule, s'il est présent.
		description_du_soldat["matricule"] = extractions.extraire_matricule(
			lignes_description_physique,
			target_text
		)
		description_du_soldat["physique"] = {}
		# On passe à la taille
		description_du_soldat["physique"]["taille"] = extractions.extraire_taille(
			lignes_description_physique,
			target_text
		)
		# On passe aux cheveux
		description_du_soldat["physique"]["cheveux"] = extractions.extraire_cheveux(
			lignes_description_physique,
			target_text
		)

		# Au front
		description_du_soldat["physique"]["front"] = extractions.extraire_front(
			lignes_description_physique,
			target_text
		)

		# Aux yeux
		description_du_soldat["physique"]["yeux"] = extractions.extraire_yeux(
			lignes_description_physique,
			target_text
		)

		# Au nez
		description_du_soldat["physique"]["nez"] = extractions.extraire_nez(
			lignes_description_physique,
			target_text
		)

		# Au visage
		description_du_soldat["physique"]["visage"] = extractions.extraire_visage(
			lignes_description_physique,
			target_text
		)

		# TODO: ajouter les renseignements physiques complémentaires et les marques particulières
		description_du_soldat["physique"][
			"renseignements_complementaires"] = extractions.extraire_renseignements_complementaires(
			lignes_description_physique,
			description_du_soldat["matricule"]
		)

		description_du_soldat["physique"]["marques_particulieres"] = extractions.extraire_marques_particulieres(
			lignes_description_physique,
			description_du_soldat["matricule"],
		)

		description_du_soldat["affectation"] = extractions.extraire_affectation_soldat(
			lignes_description_physique
		)

		return description_du_soldat

	def extraire_description_soldat_NER(self,
										ocr_prediction: OCRRecord,
										annotations: YOLORecord,
										image: str = None,
										loaded_image: PIL.Image.Image = None,
										show_images: bool = False):
		"""
		Cette fonction extrait la description du soldat à partir des prédictions et des zones.
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
		:param zones_nom: une liste de dictionnaires de la forme:
		[
			{
				'label': 'nom_du_soldat',
				'coordinates': [[212, 2400], [2735, 2551]]
			}
		]
		:param image: [Debug] le chemin vers l'image à afficher
		:param show_images: [Debug] afficher l'image
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
																				 show_images=show_images,
																				 loaded_image=loaded_image,
																				 ocr_prediction=ocr_prediction,
																				 intersect_ratio=0.1)
		lignes_description_soldat_raw = lignes_description_du_soldat.join_transcription()
		print(lignes_description_soldat_raw)
		try:
			target_lines = utils.approximate_sentence_split(sentence=lignes_description_soldat_raw,
															substring="A l'effet de juger")[-1]
		except TypeError:
			target_lines = lignes_description_soldat_raw


		description_du_soldat["prediction"] = target_lines
		result_spotting = self.entity_spotting_pipeline(target_lines)
		entities_as_dictionnary = utils.entities_to_dict(result_spotting)

		# On commence par le nom du soldat
		soldat: list[YOLOZone] = annotations.filter_zones("Nom du soldat")

		plusieurs_soldats = False
		if len(soldat) == 1:
			bbox_nom_soldat = soldat[0].coordinates
			entite_et_baseline = extractions.extraire_entite_baseline(
				dictionnaire=entities_as_dictionnary,
				nom_entite="nom_du_soldat",
				target_lines=lignes_description_du_soldat
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
			plusieurs_soldats = True
			bbox_nom_soldat = None
			print("Plusieurs soldats.")
		else:
			bbox_nom_soldat = None
			print("Aucun soldat identifié par YOLO.")


		description_du_soldat["identite"] = \
			{
				"prenom": extractions.extraire_feature(entities_as_dictionnary,
													   lignes_description_du_soldat,
													   "prénom_du_soldat"),
				"nom": extractions.extraire_feature(entities_as_dictionnary,
													lignes_description_du_soldat,
													"nom_du_soldat")
			}

		description_du_soldat["parents"] = {
			"pere": {
				"prenom": extractions.extraire_feature(
					entities_as_dictionnary,
					lignes_description_du_soldat,
					"prenom_pere"
				)
			},
			"mere": {
				"prenom": extractions.extraire_feature(
					entities_as_dictionnary,
					lignes_description_du_soldat,
					"prenom_mere"
				),
				"nom": extractions.extraire_feature(
					entities_as_dictionnary,
					lignes_description_du_soldat,
					"nom_mere"
				),
			}
		}

		description_du_soldat["identite"]["date_naissance"] = (
			extractions.extraire_date_naissance(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)

		description_du_soldat["identite"]["lieu_naissance"] = (
			extractions.extraire_lieu_naissance(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)


		description_du_soldat["identite"]["lieu_residence"] = (
			extractions.extraire_lieu_residence(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)


		description_du_soldat["identite"]["situation_maritale"] = (
			extractions.extraire_sit_maritale(entity_dict=entities_as_dictionnary,
												lignes=lignes_description_du_soldat)
		)

		description_du_soldat["identite"]["matricule"] = (
			extractions.extraire_feature(entities_as_dictionnary=entities_as_dictionnary,
										lignes=lignes_description_du_soldat,
										 feature="matricule")
		)


		description_du_soldat["identite"]["affectation"] = (
			extractions.extraire_feature(entities_as_dictionnary=entities_as_dictionnary,
										lignes=lignes_description_du_soldat,
										 feature="affectation_soldat")
		)

		description_du_soldat["identite"]["rang"] = (
			extractions.extraire_feature(entities_as_dictionnary=entities_as_dictionnary,
										lignes=lignes_description_du_soldat,
										 feature="rang_actuel")
		)

		# Profession
		description_du_soldat["profession"] = extractions.extraire_feature(
			entities_as_dictionnary,
			lignes_description_du_soldat,
			"profession"
		)


		# Description physique
		description_du_soldat["description_physique"]["marques_particulières"] = extractions.extraire_feature(
			entities_as_dictionnary,
			lignes_description_du_soldat,
			feature="marques_particulières"
		)


		description_du_soldat["description_physique"]["renseignements_complementaires"] = extractions.extraire_feature(
			entities_as_dictionnary,
			lignes_description_du_soldat,
			feature="renseignements_complementaires"
		)

		# Items courants
		for item in ["nez", "visage", "yeux", "front", "taille", "cheveux", "bouche", "menton"]:
			description_du_soldat["description_physique"][item] = extractions.extraire_feature(
				entities_as_dictionnary,
				lignes_description_du_soldat,
				feature=item
			)
			if item == "taille":
				taille_courante = description_du_soldat["description_physique"][item]["extracted"]
				if taille_courante:
					description_du_soldat["description_physique"][item]["identified"] = taille_courante
					description_du_soldat["description_physique"][item]["extracted"] = extractions.traiter_taille(
						taille_courante)

		return description_du_soldat

	def extraire_date_du_proces(self,
								ocr_prediction: OCRRecord,
								annotations: YOLORecord,
								image: str = None,
								loaded_image: PIL.Image.Image = None,
								show_images: bool = False):
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
		:param zones_nom: une liste de dictionnaires de la forme:
		[
			{
				'label': 'nom_du_soldat',
				'coordinates': [[212, 2400], [2735, 2551]]
			}
		]
		:param image: [Debug] le chemin vers l'image à afficher
		:param show_images: [Debug] afficher l'image
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
			correct_line = cejourdui_date
			line_as_string = cejourdui_date.prediction
			baseline = correct_line.baseline
		else:
			correct_lines = [cejourdui_date, an_mil_neuf_date]
			line_as_string = f"{cejourdui_date.prediction} {an_mil_neuf_date.prediction}"
			baseline = [item.baseline for item in correct_lines]
		date_span = utils.approximate_word_split(line_as_string, "CEJOURD'HUI")

		date_extraite = date_span[-1]
		print(f"Orig: {date_extraite}")
		corrected_date = utils.correct_date(date_extraite)
		print(f"Corrected: {corrected_date}")
		try:
			extracted = date.process_date(corrected_date)
		except TypeError:
			extracted = "Échec"
		self.date_proces = extracted
		return {
			"bbox": zone_magistrats,
			"baseline": baseline,
			"prediction": line_as_string,
			"Date corrigée": corrected_date,
			"date_normalisee": extracted
		}

	def extraire_magistrats(self,
							ocr_prediction,
							zones_magistrats,
							image: str = None,
							show_images: bool = True):
		"""
		Cette fonction extrait les noms des magistrats et leur statut à parti
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
		:param zones_magistrats: une liste de dictionnaires de la forme:
		[
			{
				'label': 'ligne',
				'coordinates': [[212, 2400], [2735, 2551]]
			},
			...,
			{
				'label': 'ligne',
				'coordinates': [[209, 2537], [2744, 2682]]
			}
		]
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
		for idx, line in enumerate(lignes_table_triees):
			corresponding_box = line.coordinates
			box_as_rectangle = self.rectangle(corresponding_box[0][0],
											  corresponding_box[0][1],
											  corresponding_box[1][0],
											  corresponding_box[1][1])

			# On vérifie que la ligne nous intéresse, qu'elle se trouve sur la première colonne
			overlap_ratio_first_column = utils.check_if_overlap(first_column_as_rectangle, box_as_rectangle)
			if overlap_ratio_first_column is not None and overlap_ratio_first_column < 0.5:
				continue

			# On itère sur les lignes identifiées par Kraken
			for idx, predicted_line in enumerate(ocr_prediction):
				prediction = predicted_line.prediction
				baseline = predicted_line.baseline
				# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
				converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
				is_in_box = utils.check_if_line_in_box(box_coord=box_as_rectangle, baseline=converted_baseline)

				# On vérifie que la ligne est bien dans la colonne 1
				is_in_correct_column = utils.check_if_line_in_box(box_coord=first_column_as_rectangle,
																  baseline=converted_baseline)
				if is_in_box is True:
					try:
						table_dict[idx].append(
							OCRLine(prediction=prediction, baseline=baseline, cuts=None)
						)

					except KeyError:
						table_dict[idx] = [
							OCRLine(prediction=prediction, baseline=baseline, cuts=None)
						]
		table_des_magistrats = [item for item in table_dict.values()]

		# On récupère les informations, en sachant que le premier est toujours le président
		# TODO: on peut vérifier la présence du mot `président` dans la ligne transcrite
		president = table_des_magistrats[0]
		jures = table_des_magistrats[1:]
		processed_jures = []

		# On va itérer jury par jury
		for jure in jures:
			jury_extrait = extractions.extraire_nom_et_fonction(
				prediction=" ".join(line.prediction for line in jure),
				pipeline=self.ner)
			if jury_extrait['persName'] == "UNK" and (utils.similarite_ratcliff("Président",
																				" ".join(line.prediction for line in
																						 jure)) > .7 \
													  or utils.similarite_ratcliff("Juges", " ".join(
						line.prediction for line in jure)) > .7):
				print(" ".join(line.prediction for line in jure))
				print("ÉCARTÉ")
				continue
			jury_dict = {"extracted": jury_extrait,
						 "baseline": [line.baseline for line in jure],
						 "predictions": [line.prediction for line in jure]}
			processed_jures.append(jury_dict)
		if len(processed_jures) < 4:
			print("Warning: il manque un membre du jury")
			processed_jures.append("Juré manquant")
		processed_president = extractions.extraire_nom_et_fonction(" ".join(line.prediction for line in president),
																   self.ner)

		# On travaille sur les trois derniers noms: le gradé qui nomme le jury, le commissaire, le greffier.
		coords_zone_englobante_magistrats = zone_englobante_magistrats[0].coordinates
		zone_magistrat_as_rectangle = self.rectangle(coords_zone_englobante_magistrats[0][0],
													 coords_zone_englobante_magistrats[0][1],
													 coords_zone_englobante_magistrats[1][0],
													 coords_zone_englobante_magistrats[1][1])
		lignes_zone_magistrat = []
		for predicted_line in ocr_prediction:
			prediction = predicted_line.prediction
			baseline = predicted_line.baseline
			# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
			converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
			is_in_box = utils.check_if_line_in_box(box_coord=zone_magistrat_as_rectangle, baseline=converted_baseline)
			if is_in_box is True:
				lignes_zone_magistrat.append(predicted_line)

		# On extrait le nom du greffier:
		greffier = extractions.extraire_greffier(lignes_zone_magistrat=lignes_zone_magistrat, ner_pipeline=self.ner)
		commissaire = extractions.extraire_commissaire(lignes_zone_magistrat=lignes_zone_magistrat,
													   ner_pipeline=self.ner)
		general = extractions.extraire_general(lignes_zone_magistrat=lignes_zone_magistrat, ner_pipeline=self.ner)
		# Si on ne trouve rien, c'est que la ligne est hors de la boîte. On relance sur l'ensemble des lignes.
		if general == {"grade": None}:
			general = extractions.extraire_general(lignes_zone_magistrat=ocr_prediction, ner_pipeline=self.ner)
		return {"president": {"extracted": processed_president,
							  "baseline": [line.baseline for line in president],
							  "predictions": [line.prediction for line in president]},
				"jures": processed_jures,
				"greffier": greffier,
				"commissaire": commissaire,
				"general": general}

	def finetune_categories(self):
		"""
		Cette fonction permet de produire l'extraction de l'information.
		:return:
		"""
		clean_annotations = copy.deepcopy(self.extracted_annotations)
		# On nettoie ensuite
		soldat_description_split_regexp = re.compile("([AÀ] l'effet)")
		condamnation_split_regexp = re.compile("([cC]ondamnations? ant[ée]rieures?)")
		crimeDate_split_regexp = re.compile("du délit\.\s?")
		place_split_regexp = re.compile("(GUERRE permanent d\S+)")

		for document, annotations in self.extracted_annotations.items():
			del clean_annotations[document]["Magistrats"]
			del clean_annotations[document]["Colonne"]
			del clean_annotations[document]["Table"]
			for category, annotation in annotations.items():
				if category in self.excluded_classes:
					del clean_annotations[document][category]
				if category == "Description du Soldat":
					clean = \
						utils.split_before_keep_delimiter(annotation, delimiter=soldat_description_split_regexp)[1]
					clean_annotations[document][category] = clean
				elif category == "MainZone-crimeDate":
					annotation = " ".join(annotation)
					annotation = unicodedata.normalize("NFC", annotation)
					clean_date = utils.split_after_keep_delimiter(annotation, delimiter=crimeDate_split_regexp)[1]
					clean_annotations[document]["Date du crime ou délit"] = clean_date
					if clean_date != "":
						del clean_annotations[document][category]

				elif category == "MainZone-judgementPlace":
					annotation = " ".join(annotation)
					clean_place = utils.split_after_keep_delimiter(annotation, delimiter=place_split_regexp)[1]
					if clean_place != "":
						del clean_annotations[document][category]
						clean_annotations[document]["lieu_jugement"] = clean_place

				elif category == "Inculpation_antecedents":
					annotation = [unicodedata.normalize("NFC", line) for line in annotation]
					if any(["néant" in line.lower() for line in annotation]):
						condamnations = False
						clean_annotations[document]["antécédents"] = None
					else:
						merged = " ".join(annotation)
						condamnations = \
							utils.split_after_keep_delimiter(merged, delimiter=condamnation_split_regexp)[1]
						clean_annotations[document]["antécédents"] = condamnations

					merged = " ".join(annotation)
					inculpation = \
						utils.split_before_keep_delimiter(merged, delimiter=condamnation_split_regexp)[0]
					clean_annotations[document]["inculpation"] = inculpation
					if inculpation and condamnations:
						del clean_annotations[document][category]
