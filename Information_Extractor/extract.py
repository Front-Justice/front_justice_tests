import kraken
import json
import glob
import PIL.Image as Image
import lxml.etree as ET
import re
import utils as utils


class Extractor:
	"""
	Classe pour extraire les informations à partir:
	 	- d'un corpus d'annotations au format COCO
	 	- d'un corpus de documents XML au format ALTO
	 	- du même corpus d'images
	"""
	
	def __init__(self):
		self.alto_namepaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		self.target_corpus = glob.glob("../Page_Classifier/data/corpus/page_1/*.jpg")
		self.conversion_dict = {item.replace("(", "-").replace(")", "").split("/")[-1].replace(".jpg", ""): item for item in
						   self.target_corpus}
		# Le format doit être COCO
		with open("data/annotations.json", "r") as input_Json:
			self.json_data = json.load(input_Json)

		# On construit le dictionnaire de catégories: {id:label}
		self.categories = self.json_data["categories"]
		self.categories_dict = {category['id']: category['name'] for category in self.categories}

		# On récupère les images en grande taille
		self.images_dict = {item['id']: self.conversion_dict[item['file_name'].split("_jpg")[0]] for item in self.json_data["images"]}
		self.annotations = self.json_data["annotations"]
		self.xml_dict = {}

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

	def main(self):
		xml_dict = {}


		informations = {}

		for annotation in self.annotations:
			corresp = self.images_dict[annotation["image_id"]]
			image_id = corresp.replace(".jpg", "").split("/")[-1]
			corresponding_tree = self.xml_dict[image_id]

			loaded_image = utils.load(corresp)
			test = "test"
			corresponding_category = self.categories_dict[annotation["category_id"]]
			print(f"Corresponding image: {corresp}")
			print(f"Corresponding category: {corresponding_category}")
			corresponding_box = annotation["bbox"]
			converted = utils.convert_coco_coordinates(corresponding_box)
			if corresponding_category == "Description du Soldat":
				for line in corresponding_tree["lines"]:
					baseline = [int(item) for item in line["baseline"].split(" ")]
					# Dans les cas où il y aurait plus de 2 points
					converted_baseline = [baseline[0], baseline[1], baseline[-2], baseline[-1]]
					is_in_box = utils.check_if_line_in_box(box_coord=converted, baseline=converted_baseline)
					if is_in_box is True:
						try:
							informations[image_id]["Description du Soldat"]
						except KeyError:
							informations[image_id] = {"Description du Soldat": []}
						try:
							informations[image_id]["Description du Soldat"].append(line["string"])
						except KeyError:
							informations[image_id]["Description du Soldat"] = [line["string"]]
				# print([round(item) for item in corresponding_box])
				# cropped = loaded_image.crop(converted)
				# cropped.show()
				# exit(0)

		# On nettoie ensuite
		description_split_regexp = re.compile("([AÀ] l'effet)")
		print(informations)
		for page, info in informations.items():
			description = info["Description du Soldat"]
			info["Description du Soldat"] = "\n".join(description)
			clean = utils.split_keep_delimiter(info["Description du Soldat"], delimiter=description_split_regexp)[1]
			print(clean)
			info["Description du Soldat"] = clean
		# description_du_soldat = "\n".join(informations["Description du Soldat"])
		# specific_span = description_du_soldat.split("A l'effet")
		# print(specific_span)
		print(informations)







if __name__ == '__main__':
	extractor = Extractor()
	extractor.build_xml_dict()
	extractor.main()