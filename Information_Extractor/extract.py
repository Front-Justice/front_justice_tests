###############

## Script d'extraction à partir des segmentations. À lier avec le script "segmentation_kraken_yolo.

###############

import time
from yaspin import yaspin
import unicodedata
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import glob
import re
import Information_Extractor.utils as utils
import copy
import PIL.Image as Image
import PIL
from collections import namedtuple



class Extractor:
	"""
	Classe pour extraire les informations à partir:
	 	- d'un corpus d'annotations au format COCO
	 	- d'un corpus de documents XML au format ALTO
	 	- du même corpus d'images
	"""
	
	def __init__(self):

		self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
		self.ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

		self.nlp = pipeline('ner', model=self.ner_model, tokenizer=self.tokenizer, aggregation_strategy="simple")
		self.alto_namepaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		self.target_corpus = glob.glob("../Page_Classifier/data/corpus/page_1/*.jpg")
		self.conversion_dict = {item.replace("(", "-").replace(")", "").split("/")[-1].replace(".jpg", ""): item for item in
						   self.target_corpus}
		# Le format doit être COCO

		self.rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


		# On récupère les images en grande taille
		self.extracted_annotations = {}
		self.excluded_classes = ["Titre"]




	def filter_zones(self, annotations:list[dict], category:str) -> list[dict]:
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
		return [annotation for annotation in annotations if annotation['label'] == category]

	def extraire_numero_jugement(self,
							  ocr_prediction: list[dict],
							  annotations: list[dict],
							  party_engine,
							  image: str = None,
							  loaded_image: PIL.Image.Image = None,
							  show_images: bool = True):
		"""
		Cette fonction extrait le numéro de jugement à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param party_engine: le moteur de transcription party
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
		  "confidence": 0.8,
		  "predictions": {
			"party": "N^o 501 D'ORDRE.",
			"kraken": "N^o 501 D'ORDRE."
		  }
		},
		"""

		# On récupère la boîte correspondante
		numero_jugement = self.filter_zones(annotations, "MainZone-judgementNumber")

		if len(numero_jugement) == 0:
			return None

		# On teste s'il y a plusieurs soldats
		assert len(numero_jugement) == 1, "Erreur: plusieurs zones détectées"

		# On va commencer par identifier le nom prédit par kraken
		numero_jugement_zone = numero_jugement[0]['coordinates']
		numero_jugement_as_rectangle = self.rectangle(numero_jugement_zone[0][0],
												   numero_jugement_zone[0][1],
												   numero_jugement_zone[1][0],
												   numero_jugement_zone[1][1])

		if show_images:
			cropped = loaded_image.crop((numero_jugement_zone[0][0],
										 numero_jugement_zone[0][1],
										 numero_jugement_zone[1][0],
										 numero_jugement_zone[1][1]))
			cropped.show()

		# On cherche la ligne qui entre dans la zone du nom du soldat
		corresponding_lines = utils.match_lines_in_zones(ocr_prediction=ocr_prediction,
														 zone_as_rectangle=numero_jugement_as_rectangle,
														 intersect_ratio=0.7)
		corresponding_lines = utils.vertical_order_lines(lines=corresponding_lines)

		target_line = []
		for line in corresponding_lines:
			prediction = line['prediction']
			# On va chercher une ligne avec un nombre uniquement ici
			expression_jugement = re.compile("\d+")
			is_jugement = re.match(expression_jugement, prediction)
			if is_jugement:
				target_line.append(line)

		assert len(target_line) == 1, "Erreur. Plusieurs lignes trouvées pour le numéro de jugement."

		# On transcrit avec party
		party_segmentation = party_engine.create_baseline(target_line[0]['baseline'], image)
		party_prediction = utils.measured_party_inference(party_engine=party_engine,
														  segmentation=party_segmentation,
														  image=loaded_image,
														  objet_transcrit="numéro de jugement")
		numero_jugement_party = party_prediction.prediction
		target_line = target_line[0]
		numero_jugement_kraken = target_line['prediction']


		if numero_jugement_party == numero_jugement_kraken:
			confidence = 0.8
			target_number = numero_jugement_kraken
		else:
			confidence = 0.5
			target_number = numero_jugement_party

		return {"Numéro": target_number,
				"baseline": target_line['baseline'],
				"bbox": numero_jugement_zone,
				"confidence": confidence,
				"predictions": {"party": numero_jugement_party,
								"kraken": numero_jugement_kraken}}




	def extraire_numero_ordre(self,
							  ocr_prediction:list[dict],
							  annotations:list[dict],
							  party_engine,
							  image:str=None,
							  loaded_image:PIL.Image.Image=None,
							  show_images:bool=True):
		"""
		Cette fonction extrait le numéro d'ordre à partir des prédictions et des zones.
		On va comparer la prédiction de Kraken et de Party pour arriver à un meilleur résultat.
		:param party_engine: le moteur de transcription party
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
          "confidence": 0.8,
          "predictions": {
            "party": "N^o 501 D'ORDRE.",
            "kraken": "N^o 501 D'ORDRE."
          }
        },
		"""

		# On récupère la boîte correspondante
		numero_ordre = self.filter_zones(annotations, "MainZone-orderNumber")

		if len(numero_ordre) == 0:
			return None

		# On teste s'il y a plusieurs soldats
		assert len(numero_ordre) == 1, "Erreur: plusieurs zones détectées"


		# On va commencer par identifier le nom prédit par kraken
		numero_ordre_zone = numero_ordre[0]['coordinates']
		numero_ordre_as_rectangle = self.rectangle(numero_ordre_zone[0][0],
												  numero_ordre_zone[0][1],
												  numero_ordre_zone[1][0],
												  numero_ordre_zone[1][1])


		if show_images:
			cropped = loaded_image.crop((numero_ordre_zone[0][0],
									 numero_ordre_zone[0][1],
									 numero_ordre_zone[1][0],
									 numero_ordre_zone[1][1]))
			cropped.show()

		# On cherche la ligne qui entre dans la zone du nom du soldat
		corresponding_lines = utils.match_lines_in_zones(ocr_prediction=ocr_prediction,
														 zone_as_rectangle=numero_ordre_as_rectangle,
														 intersect_ratio=0.7)
		corresponding_lines = utils.vertical_order_lines(lines=corresponding_lines)

		target_line = []
		for line in corresponding_lines:
			prediction = line['prediction']
			similarity = utils.similarite_ratcliff(prediction, "D'ORDRE.")

			# On condidère une valeur de similarité de 0.5, à modifier par l'expérience
			if similarity > .5:
				target_line.append(line)

		assert len(target_line) == 1, "Erreur. Plusieurs lignes trouvées pour le numéro d'ordre."



		# On transcrit avec party
		party_segmentation = party_engine.create_baseline(target_line[0]['baseline'], image)
		party_prediction = utils.measured_party_inference(party_engine=party_engine,
														  segmentation=party_segmentation,
														  image=loaded_image,
														  objet_transcrit="numéro d'ordre")
		numero_ordre_party = party_prediction.prediction


		target_line = target_line[0]

		numero_regexp = re.compile("\d+")
		target_number_kraken = re.search(numero_regexp, target_line['prediction']).group()
		target_number_party = re.search(numero_regexp, numero_ordre_party).group()

		if target_number_party == target_number_kraken:
			confidence = 0.8
			target_number = target_number_kraken
		else:
			confidence = 0.5
			target_number = target_number_party


		return {"Numéro": target_number,
				"baseline": target_line['baseline'],
				"bbox": numero_ordre_zone,
				"confidence": confidence,
				"predictions": {"party": numero_ordre_party,
								"kraken": target_line['prediction']}}



	def extraire_nom_soldat(self,
							ocr_prediction:list[dict],
							annotations:list[dict],
							party_engine,
							image:str=None,
							loaded_image:PIL.Image.Image=None,
							show_images:bool=False):
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
				'label': 'Nom du soldat',
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

		# On récupère la boîte correspondante
		nom_du_soldat = self.filter_zones(annotations, "Nom du soldat")

		# On teste s'il y a plusieurs soldats
		assert len(nom_du_soldat) == 1, "Jugement de plusieurs soldats. À implémenter"


		# On va commencer par identifier le nom prédit par kraken
		soldat_zone = nom_du_soldat[0]['coordinates']
		soldat_zone_as_rectangle = self.rectangle(soldat_zone[0][0],
												  soldat_zone[0][1],
												  soldat_zone[1][0],
												  soldat_zone[1][1])

		# On cherche la ligne qui entre dans la zone du nom du soldat

		corresponding_lines = utils.match_lines_in_zones(ocr_prediction=ocr_prediction,
														 zone_as_rectangle=soldat_zone_as_rectangle,
														 intersect_ratio=0.1)

		# On peut avoir plusieurs lignes, car le nom est écrit en gros module. On va
		# Donc tester la distance au début de la ligne qui commence par "A l'effet de juger"
		distances = []
		if len(corresponding_lines) > 1:
			for idx, ligne in enumerate(corresponding_lines):
				prediction = ligne['prediction']
				# On identifie la ligne pouvant contenir à l'effet de juger
				dist = utils.similarite_ratcliff(prediction, "A l'effet de juger")
				distances.append(dist)
		correct_line_index = distances.index(max(distances))
		name_line = corresponding_lines[correct_line_index]

		# On a la ligne correspondante. Maintenant, on va identifier les caractères
		# qui sont compris dans la boîte à l'aide des cuts de kraken. On a besoin de tout convertir
		# en polygones, pour ensuite vérifier les intersections entre la boîte et les intersections
		nom_du_soldat_kraken = utils.extract_string_from_cuts(box=soldat_zone, line=name_line).strip()
		prenoms, certitude_prenoms = utils.extraction_prenom_du_soldat(name_line['prediction'],
																	   nom_du_soldat_kraken,
																	   pipeline=self.nlp)

		if show_images:
			cropped = loaded_image.crop((soldat_zone[0][0],
									 soldat_zone[0][1],
									 soldat_zone[1][0],
									 soldat_zone[1][1]))
			cropped.show()

		# On prédit à l'aide de party la baseline qui se trouve à l'intérieur de la boite
		party_segmentation = party_engine.create_baseline(soldat_zone, image)
		party_prediction = utils.measured_party_inference(party_engine=party_engine,
														  segmentation=party_segmentation,
														  image=loaded_image,
														  objet_transcrit="nom du soldat")

		nom_soldat_party = party_prediction.prediction

		# on produit le dictionnaire
		extracted = {}
		extracted["forename"] = {"value": prenoms,
								 "certainty": certitude_prenoms}
		if nom_du_soldat_kraken == nom_soldat_party:
			extracted["surname"] = {"value": nom_soldat_party,
									"certainty": 1}
		else:
			extracted["surname"] = {"value": nom_soldat_party if nom_soldat_party != "+" else nom_du_soldat_kraken,
									"certainty": 0.5,
									"predictions": {"kraken": nom_du_soldat_kraken,
													"party": nom_soldat_party}
									}

		return {"bbox": soldat_zone,
				"extracted": extracted}



	def extraire_table_magistrats(self,
								  ocr_prediction,
								  zones_magistrats,
								  image:str=None,
								  show_images:bool=True):
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
		:return: Un dictionnaire de la forme:
			{
			  "Président": {
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
			  "Jurés": [
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
		column_annotation = self.filter_zones(zones_magistrats, "Colonne")
		lines_annotation = self.filter_zones(zones_magistrats, "ligne")
		sorted_lines = utils.vertical_order_zones(lines_annotation)
		first_column, _ = utils.horizontal_order_zones(column_annotation)
		first_column = first_column["coordinates"]
		first_column_as_rectangle = self.rectangle(first_column[0][0],
												   first_column[0][1],
												   first_column[1][0],
												   first_column[1][1])
		if show_images:
			for line in sorted_lines:
				loaded_image = Image.open(image)
				cropped = loaded_image.crop(line["coordinates"])
				cropped.show()


		# On itère sur les zones identifiées par YOLO
		for idx, line in enumerate(sorted_lines):
			corresponding_box = line["coordinates"]
			box_as_rectangle = self.rectangle(corresponding_box[0][0],
											  corresponding_box[0][1],
											  corresponding_box[1][0],
											  corresponding_box[1][1])

			# On vérifie que la ligne nous intéresse, qu'elle se trouve sur la première colonne
			overlap_ratio_first_column = utils.check_if_overlap(first_column_as_rectangle, box_as_rectangle)
			if overlap_ratio_first_column is not None and overlap_ratio_first_column < 0.5:
				continue

			# On itère sur les lignes identifiées par Kraken
			for predicted_line in ocr_prediction:
				prediction = predicted_line["prediction"]
				baseline = predicted_line["baseline"]
				# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
				converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
				is_in_box = utils.check_if_line_in_box(box_coord=box_as_rectangle, baseline=converted_baseline)

				# On vérifie que la ligne est bien dans la colonne 1
				is_in_correct_column = utils.check_if_line_in_box(box_coord=first_column_as_rectangle, baseline=converted_baseline)
				if is_in_box is True:
					try:
						table_dict[idx].append(
							{
								"prediction": prediction,
								"baseline": baseline
							 }
						)

					except KeyError:
						table_dict[idx] = [
							{
								"prediction": prediction,
								"baseline": baseline
							 }
						]
		table_list = [item for item in table_dict.values()]

		# On récupère les informations, en sachant que le premier est toujours le président
		# TODO: on peut vérifier la présence du mot `président` dans la ligne transcrite
		president = table_list[0]
		jures = table_list[1:]
		processed_jures = []

		# On va itérer jury par jury
		for jure in jures:
			extracted_entities = utils.extract_magistrates_names(" ".join(line['prediction'] for line in jure), self.nlp)
			jury_dict = {"extracted": extracted_entities,
						 "baseline": [line['baseline'] for line in jure],
						 "predictions": [line['prediction'] for line in jure]}
			processed_jures.append(jury_dict)
		processed_president = utils.extract_magistrates_names(" ".join(line['prediction'] for line in president), self.nlp)
		return {"Président": {"extracted": processed_president,
							  "baseline": [line['baseline'] for line in president],
							  "predictions": [line['prediction'] for line in president]},
								"Jurés": processed_jures}

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
				if category  == "Description du Soldat":
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
					print("---")
					print([line for line in annotation])
					annotation = [unicodedata.normalize("NFC", line) for line in annotation]
					print(["néant" in line.lower() for line in annotation])
					if any(["néant" in line.lower() for line in annotation]):
						condamnations = False
						clean_annotations[document]["antécédents"] = None
					else:
						merged = " ".join(annotation)
						print(merged)
						condamnations = \
						utils.split_after_keep_delimiter(merged, delimiter=condamnation_split_regexp)[1]
						clean_annotations[document]["antécédents"] = condamnations

					merged = " ".join(annotation)
					print(merged)
					inculpation = \
					utils.split_before_keep_delimiter(merged, delimiter=condamnation_split_regexp)[0]
					clean_annotations[document]["inculpation"] = inculpation
					if inculpation and condamnations:
						del clean_annotations[document][category]








