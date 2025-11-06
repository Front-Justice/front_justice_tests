import unicodedata

import kraken
import json
import glob
import PIL.Image as Image
import lxml.etree as ET
import re
import utils as utils
import copy
from collections import namedtuple


class Extractor:
	"""
	Classe pour extraire les self.extracted_annotations à partir:
	 	- d'un corpus d'annotations au format COCO
	 	- d'un corpus de documents XML au format ALTO
	 	- du même corpus d'images
	"""
	
	def __init__(self,
				 path_to_annotations: str, ):
		self.alto_namepaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		self.target_corpus = glob.glob("../Page_Classifier/data/corpus/page_1/*.jpg")
		self.conversion_dict = {item.replace("(", "-").replace(")", "").split("/")[-1].replace(".jpg", ""): item for item in
						   self.target_corpus}
		# Le format doit être COCO

		self.rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
		with open(path_to_annotations, "r") as input_Json:
			self.json_data = json.load(input_Json)

		# On construit le dictionnaire de catégories: {id:label}
		self.categories = self.json_data["categories"]
		self.categories_dict = {category['id']: category['name'] for category in self.categories}
		self.reverse_categories_dict = {value:key for key, value in self.categories_dict.items()}

		# On récupère les images en grande taille
		self.images_dict = {item['id']: self.conversion_dict[item['file_name'].split("_jpg")[0]] for item in self.json_data["images"]}
		self.annotations = self.json_data["annotations"]
		self.xml_dict = {}
		self.extracted_annotations = {}
		self.excluded_classes = ["Titre"]


		self.build_xml_dict()
		self.filtered_by_document = {}
		for annotation in self.annotations:
			corresp = self.images_dict[annotation["image_id"]]
			image_id = corresp.replace(".jpg", "").split("/")[-1]
			corresponding_tree = self.xml_dict[image_id]
			try:
				self.filtered_by_document[image_id].append(annotation)
			except KeyError:
				self.filtered_by_document[image_id] = [annotation]


	def build_xml_dict(self):
		""" Construit un dictionnaire de la forme:
			 Dict: {
			 "basename":
			 {"tree": lxml.Element,
			 "lines":
			 			[
			 			{"element": lxml.Element,
			 				"id": id
			 				"baseline": baseline
			 				"string": string			}
			 			},
			 				etc...		]
			 }
		:return:
		"""
		for annotation in self.annotations:
			corresp = self.images_dict[annotation["image_id"]]
			corresp_xml = corresp.replace(".jpg", ".xml")
			basename = corresp_xml.split("/")[-1]
			correct_path = f"data/xml/{basename}"
			print(correct_path)
			as_tree = ET.parse(correct_path)
			all_lines = as_tree.xpath("//alto:TextLine", namespaces=self.alto_namepaces)
			all_lines_id = as_tree.xpath("//alto:TextLine/@ID", namespaces=self.alto_namepaces)
			all_lines_baseline = as_tree.xpath("//alto:TextLine/@BASELINE", namespaces=self.alto_namepaces)
			all_lines_string = as_tree.xpath("//alto:TextLine/alto:String/@CONTENT", namespaces=self.alto_namepaces)
			assert len(all_lines) == len(all_lines_string) == len(all_lines_baseline) == len(all_lines_id)
			zipped = list(zip(all_lines, all_lines_id, all_lines_baseline, all_lines_string))
			lines = [{"element": element, "id": id, "baseline": baseline, "string": string} for
					 element, id, baseline, string in zipped]


			self.xml_dict[basename.replace(".xml", "")] = {"tree": as_tree, "lines": lines}

	def filter_zones(self, annotations, category):
		corresp_id = self.reverse_categories_dict[category]
		return [annotation for annotation in annotations if annotation['category_id'] == corresp_id]

	def horizontal_order_zones(self, annotations):
		pass



	def reconstruct_magistrates_table(self):
		"""
		Cette fonction reconstruit les tables contenant le nom des magistrats
		:return:
		"""

		# Pour chaque document
		table_dict = {}
		for document, annotations in self.filtered_by_document.items():
			print(document)
			table_annotation = self.filter_zones(annotations, "Table")
			column_annotation = self.filter_zones(annotations, "Colonne")
			lines_annotation = self.filter_zones(annotations, "ligne")
			sorted_lines = utils.vertical_order_zones(lines_annotation)
			sorted_columns = utils.horizontal_order_zones(column_annotation)
			first_column = utils.convert_coco_coordinates(sorted_columns[0][1]['bbox'])
			second_column = utils.convert_coco_coordinates(sorted_columns[1][1]['bbox'])
			corresponding_tree = self.xml_dict[document]

			# On itère sur les zones identifiées par YOLO
			for idx, (_, line_zone) in enumerate(sorted_lines):
				corresponding_box = line_zone["bbox"]
				converted = utils.convert_coco_coordinates(corresponding_box)
				box_as_rectangle = self.rectangle(converted[0], converted[1], converted[2], converted[3])
				first_column = self.rectangle(first_column[0], 
											 first_column[1], 
											 first_column[2], 
											 first_column[3])
				second_column = self.rectangle(second_column[0], 
											 second_column[1], 
											 second_column[2], 
											 second_column[3])

				# On vérifie que la ligne nous intéresse, qu'elle se trouve sur la première colonne
				overlap_ratio_first_column = utils.check_if_overlap(first_column, box_as_rectangle)
				if overlap_ratio_first_column < 0.5:
					continue
				# On itère sur les lignes identifiées par Kraken
				for line in corresponding_tree["lines"]:
					baseline = [int(item) for item in line["baseline"].split(" ")]
					# Dans les cas où il y aurait plus de 2 points, on prend le premier et le dernier point
					converted_baseline = [baseline[0], baseline[1], baseline[-2], baseline[-1]]
					is_in_box = utils.check_if_line_in_box(box_coord=converted, baseline=converted_baseline)

					# On vérifie que la ligne est bien dans la colonne 1
					is_in_correct_column = utils.check_if_line_in_box(box_coord=first_column, baseline=converted_baseline)
					if is_in_box is True and is_in_correct_column is True:
						try:
							table_dict[document]
						except KeyError:
							table_dict[document] = {}
						try:
							table_dict[document][idx].append(line["string"])
						except KeyError:
							table_dict[document][idx] = [line["string"]]
		print(table_dict)
		exit(0)

	def clean_annotations(self):
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
			self.reconstruct_magistrates_table(document, annotations)
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

		with open("result/annotations.json", "w") as input_Json:
			json.dump(clean_annotations, input_Json, indent=4)

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
	table_magistrats = extractor.reconstruct_magistrates_table()
	extractor.extract()
	extractor.clean_annotations()