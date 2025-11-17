###############

## Script d'extraction à partir des segmentations. À lier avec le script "segmentation_kraken_yolo.

###############


import unicodedata

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import glob
import re
import Information_Extractor.utils as utils
import copy
import PIL.Image as Image
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

	def horizontal_order_zones(self, annotations):
		pass



	def extract_magistrates_table(self,
								  ocr_prediction,
								  zones_magistrats,
								  image:str=None,
								  show_images:bool=True):
		"""
		Cette fonction extrait les noms des magistrats et leur statut à parti
		:param ocr_prediction: Une liste de dictionnaires de la forme:
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
		:param image: 
		:param show_images: 
		:return: 
		"""

		table_dict = {}
		column_annotation = self.filter_zones(zones_magistrats, "Colonne")
		lines_annotation = self.filter_zones(zones_magistrats, "ligne")
		sorted_lines = utils.vertical_order_zones(lines_annotation)
		first_column, second_column = utils.horizontal_order_zones(column_annotation)
		first_column, second_column = first_column["coordinates"], second_column["coordinates"]
		print(sorted_lines)
		if show_images:
			for line in sorted_lines:
				loaded_image = Image.open(image)
				cropped = loaded_image.crop(line["coordinates"])
				cropped.show()


		# On itère sur les zones identifiées par YOLO
		for line in sorted_lines:
			corresponding_box = line["coordinates"]
			print(corresponding_box)
			box_as_rectangle = self.rectangle(corresponding_box[0][0],
											  corresponding_box[0][1],
											  corresponding_box[1][0],
											  corresponding_box[1][1])
			print(first_column)
			first_column = self.rectangle(first_column[0][0],
										 first_column[0][1],
										 first_column[1][0],
										 first_column[1][1])
			second_column = self.rectangle(second_column[0][0],
										 second_column[0][1],
										 second_column[1][0],
										 second_column[1][1])

			# On vérifie que la ligne nous intéresse, qu'elle se trouve sur la première colonne
			overlap_ratio_first_column = utils.check_if_overlap(first_column, box_as_rectangle)
			if overlap_ratio_first_column < 0.5:
				continue

			# On itère sur les lignes identifiées par Kraken
			for predicted_line in ocr_prediction:
				prediction = predicted_line["prediction"]
				baseline = predicted_line["baseline"]
				# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
				converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
				is_in_box = utils.check_if_line_in_box(box_coord=box_as_rectangle, baseline=converted_baseline)

				# On vérifie que la ligne est bien dans la colonne 1
				is_in_correct_column = utils.check_if_line_in_box(box_coord=first_column, baseline=converted_baseline)
				if is_in_box is True:
					try:
						table_dict[document]
					except KeyError:
						table_dict[document] = {}
					try:
						table_dict[document][idx].append(line["string"])
					except KeyError:
						table_dict[document][idx] = [line["string"]]
		print(len(table_dict))
		for document, annotations in table_dict.items():
			president = annotations[0]
			jures = [annotation for key, annotation in annotations.items() if key != 0]
			processed_jures = []
			for jure in jures:
				jure = utils.process_name(jure, self.nlp)
				processed_jures.append(jure)
			processed_president = utils.process_name(president, self.nlp)
			table_dict[document] = {"Président": processed_president,
									"Jurés": processed_jures}
		return table_dict

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


	def extract(self):
		for annotation in self.annotations:
			corresp = self.images_dict[annotation["image_id"]]
			image_id = corresp.replace(".jpg", "").split("/")[-1]
			corresponding_tree = self.xml_dict[image_id]

			corresponding_category = self.categories_dict[annotation["category_id"]]
			# print(f"Corresponding image: {corresp}")
			# print(f"Corresponding category: {corresponding_category}")
			corresponding_box = annotation["bbox"]
			converted = utils.convert_coco_coordinates(corresponding_box)
			for line in corresponding_tree["lines"]:
				baseline = [int(item) for item in line["baseline"].split(" ")]
				# Dans les cas où il y aurait plus de 2 points
				converted_baseline = [baseline[0], baseline[1], baseline[-2], baseline[-1]]
				is_in_box = utils.check_if_line_in_box(box_coord=converted, baseline=converted_baseline)
				if is_in_box is True:
					try:
						self.extracted_annotations[image_id]
					except KeyError:
						self.extracted_annotations[image_id] = {}
					try:
						self.extracted_annotations[image_id][corresponding_category]
					except KeyError:
						self.extracted_annotations[image_id] = {**self.extracted_annotations[image_id], **{corresponding_category: []}}
					try:
						self.extracted_annotations[image_id][corresponding_category].append(line["string"])
					except KeyError:
						self.extracted_annotations[image_id][corresponding_category] = [line["string"]]
				# print([round(item) for item in corresponding_box])
				# cropped = loaded_image.crop(converted)
				# cropped.show()
				# exit(0)








if __name__ == '__main__':
	annotations_first_page = "data/first_page/annotations.json"
	annotations_table = "data/table_management/annotations.json"
	extractor = Extractor(path_to_annotations=annotations_table)
	extractor.extract()
	extractor.finetune_categories()
	final_dict = {}
	table_magistrats = extractor.extract_magistrates_table()
	print(table_magistrats)
	for document, annotations in extractor.extracted_annotations.items():
		print(document)
		for document_magistrat, magistrats in table_magistrats.items():
			print(document_magistrat)
			if document_magistrat == document:
				print("Merging.")
				final_dict[document] = {**magistrats, **annotations}
				print(final_dict[document])

		del final_dict[document]["Magistrats"]
		del final_dict[document]["Colonne"]
		del final_dict[document]["Table"]
		del final_dict[document]["ligne"]

	with open("result/annotations.json", "w") as input_Json:
		json.dump(final_dict, input_Json, indent=4)
