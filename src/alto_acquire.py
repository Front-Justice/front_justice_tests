import argparse
import copy
import os
import shutil
import uuid
from collections import namedtuple
import multiprocessing as mp
import tqdm

import torch.cuda
import zipfile
import utils.utils as utils
import Page_Classifier.train as PC
import Vision.KRAKEN as KRAKEN
import Information_Extractor.extract as extract
import glob
import PIL.Image as Image
import json
import lxml.etree as ET

import Vision.YOLO as YOLO
from src.utils.utils import OCRRecord


class Pipeline():
	def __init__(self,
				 page_classifier_model,
				 page_classifier_vocab,
				 yolo_models,
				 debug:bool = False,
				 resegment=False,
				 retranscribe=False,
				 device="cpu"):
		self.debug = debug
		self.device = device
		self.alto_ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		self.page_classifier = PC.PageClassifier(build_vocab=False,
												 model=page_classifier_model,
												 vocab=page_classifier_vocab)
		self.current_image = None
		self.current_image_path = None
		self.current_page_transcription = None
		self.minutes = {}
		self.images_name_list = []

		self.rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
		self.current_image_idx = 0
		self.pages_classees = []

		# Les modèles de zones
		self.yolo_models = {}
		for name, path in yolo_models.items():
			assert os.path.exists(path), f"{path} n'existe pas."
			self.yolo_models[name] = YOLO.load(path)
		self.YOLO_Segmenter = YOLO.YOLOSegmenter()

		# Les modèles d'OCR
		self.resegment = resegment
		self.retranscribe = retranscribe
		self.global_yolo_model = YOLO.load("src/Vision/models/yolo_all_pages.pt")
		self.kraken_lines_model = {
			 0: "src/Vision/models/lignes_ajouts.mlmodel",
			 1: "src/Vision/models/modele_ligne_page_1.mlmodel",
			 2: "src/Vision/models/modele_ligne_page_2.mlmodel",
			 3: "src/Vision/models/modele_ligne_page_3.mlmodel",
			 4: "src/Vision/models/modele_page_4.mlmodel",
			"autre": "src/Vision/models/modele_ligne_page_1.mlmodel",
			 }
		self.kraken_ocr_model = "src/Vision/models/htr_29500l.mlmodel"
		self.kraken_gloses_model = "src/Vision/models/strate_2_3000l.mlmodel"
		self.party_model = "src/Vision/models/model.safetensors"
		self.minutes_annotation_file = ""

		self.segmOnto_labels = """
	<Tags>
		<OtherTag ID="BT16575" LABEL="MarginTextZone" DESCRIPTION="block type MarginTextZone"/>
		<OtherTag ID="BT16576" LABEL="DamageZone" DESCRIPTION="block type DamageZone"/>
		<OtherTag ID="BT16577" LABEL="DigitizationArtefactZone" DESCRIPTION="block type DigitizationArtefactZone"/>
		<OtherTag ID="BT16578" LABEL="DropCapitalZone" DESCRIPTION="block type DropCapitalZone"/>
		<OtherTag ID="BT16579" LABEL="GraphicZone" DESCRIPTION="block type GraphicZone"/>
		<OtherTag ID="BT16581" LABEL="NumberingZone" DESCRIPTION="block type NumberingZone"/>
		<OtherTag ID="BT16582" LABEL="QuireMarksZone" DESCRIPTION="block type QuireMarksZone"/>
		<OtherTag ID="BT16583" LABEL="RunningTitleZone" DESCRIPTION="block type RunningTitleZone"/>
		<OtherTag ID="BT16584" LABEL="SealZone" DESCRIPTION="block type SealZone"/>
		<OtherTag ID="BT16585" LABEL="StampZone" DESCRIPTION="block type StampZone"/>
		<OtherTag ID="BT26269" LABEL="MainZone-orderNumber" DESCRIPTION="block type MainZone-orderNumber"/>
		<OtherTag ID="BT26270" LABEL="MainZone-crimeDate" DESCRIPTION="block type MainZone-crimeDate"/>
		<OtherTag ID="BT26271" LABEL="MainZone-judgementPlace" DESCRIPTION="block type MainZone-judgementPlace"/>
		<OtherTag ID="BT26272" LABEL="MarginTextZone-ajout" DESCRIPTION="block type MarginTextZone-ajout"/>
		<OtherTag ID="BT26273" LABEL="MainZone" DESCRIPTION="block type MainZone"/>
		<OtherTag ID="BT26274" LABEL="MarginTextZone-note" DESCRIPTION="block type MarginTextZone-note"/>
		<OtherTag ID="BT26275" LABEL="QuireMarksZone-signature" DESCRIPTION="block type QuireMarksZone-signature"/>
		<OtherTag ID="BT26276" LABEL="MainZone-judgementNumber" DESCRIPTION="block type MainZone-judgementNumber"/>
		<OtherTag ID="BT26277" LABEL="CustomZone-signature" DESCRIPTION="block type CustomZone-signature"/>
		<OtherTag ID="LT6926" LABEL="DefaultLine" DESCRIPTION="line type DefaultLine"/>
		<OtherTag ID="LT6927" LABEL="HeadingLine" DESCRIPTION="line type HeadingLine"/>
		<OtherTag ID="LT6928" LABEL="InterlinearLine" DESCRIPTION="line type InterlinearLine"/>
		<OtherTag ID="LT10703" LABEL="CustomLine:margin" DESCRIPTION="line type CustomLine:margin"/>
		<OtherTag ID="LT10704" LABEL="CustomLine:scratched" DESCRIPTION="line type CustomLine:scratched"/>
		<OtherTag ID="LT10705" LABEL="CustomLine:signature" DESCRIPTION="line type CustomLine:signature"/>
		<OtherTag ID="LT10706" LABEL="default" DESCRIPTION="line type default"/>
		<OtherTag ID="LT12191" LABEL="CustomLine:addition" DESCRIPTION="line type CustomLine:addition"/>
	</Tags>
	"""

		self.tagrefs_dict = {"BT16575": "MarginTextZone",
"BT16576": "DamageZone",
"BT16577": "DigitizationArtefactZone",
"BT16578": "DropCapitalZone",
"BT16579": "GraphicZone",
"BT16581": "NumberingZone",
"BT16582": "QuireMarksZone",
"BT16583": "RunningTitleZone",
"BT16584": "SealZone",
"BT16585": "StampZone",
"BT26269": "MainZone-orderNumber",
"BT26270": "MainZone-crimeDate",
"BT26271": "MainZone-judgementPlace",
"BT26272": "MarginTextZone-ajout",
"BT26273": "MainZone",
"BT26274": "MarginTextZone-note",
"BT26275": "QuireMarksZone-signature",
"BT26276": "MainZone-judgementNumber",
"BT26277": "CustomZone-signature",
"LT6926": "DefaultLine",
"LT6927": "HeadingLine",
"LT6928": "InterlinearLine",
"LT10703": "CustomLine:margin",
"LT10704": "CustomLine:scratched",
"LT10705": "CustomLine:signature",
"LT10706": "default",
"LT12191": "CustomLine:addition"}
		self.reverse_tagrefs_dict = {val:key for key, val in self.tagrefs_dict.items()}

	def load_image(self, image):
		self.current_image_path = image

	def classify_image(self):
		self.current_page_type = self.page_classifier.predict(image=self.current_image_path)

	def classification_images(self, images):
		"""
		Cette fonction classe toutes les images à l'aide d'un Random Forest
		:param images: la liste d'images
		:return:
		"""
		# On commence par classer toutes les images du dossier
		print("Classification des images")
		for image in tqdm.tqdm(images):
			dossier, ident = utils.get_name_from_path(image)
			# On vérifie s'il n'y a pas de problème de disparition d'image
			self.check_image_consistency(ident)
			self.images_name_list.append(ident)
			self.load_image(image)
			self.classify_image()
			self.pages_classees.append(((dossier, ident, image), self.current_page_type))
			if image == images[-1]:
				print("Dossier terminé")

	def regroupement_minutes(self, out_dir):
		"""
		Cette fonction regroupe les minutes
		:return: None, mais produit le dictionnaire self.minutes de la forme:
		 ```JSON
		 {0: [
		 {'répertoire': '11_J_187(1)',
		 'id': 33,
		 'image_path': 'data/minute_test/11_J_187(1)_0033.jpg',
		 'classe': 'page_1'},
		 ...
		 {'répertoire': '11_J_187(1)',
		 'id': 36,
		 'image_path': 'data/minute_test/11_J_187(1)_0036.jpg',
		 'classe': 'page_4'}]
		 }```
		"""
		print("Reconstitution des minutes")
		current_minute = []
		current_minute_number = 0
		# Puis on rassemble les minutes
		for idx, ((dossier, ident, image), classe) in enumerate(self.pages_classees):
			current_image = {}
			current_image["répertoire"] = dossier
			current_image["id"] = ident
			current_image["image_path"] = image
			current_image["classe"] = classe
			current_minute.append(current_image)
			if ident == self.pages_classees[-1][0][1]:
				print("Dossier terminé")
				self.minutes[current_minute_number] = current_minute
				break
			if classe in ["page_4", "page_autre"] and self.pages_classees[idx + 1][1] == "page_1":
				print("Minute terminée")
				self.minutes[current_minute_number] = current_minute
				current_minute = []
				current_minute_number += 1
		utils.save_as_dict(self.minutes, out_dir)

	def check_image_consistency(self, current_image):
		"""
		Cette fonction vérifie s'il y a un problème au sein des fichiers et si une image est manquante,
		fondé sur la liste des images qui doit être une liste suivie d'entier
		:param current_image:
		:return:
		"""
		if len(self.images_name_list) != 0 and current_image - self.images_name_list[-1] != 1:
			print(f"Il manque probablement une image.")
			print(f"Image courante: {current_image}. \n"
				  f"Image précédente: {self.images_name_list[-1]}.\n"
				  f"On passe à la minute suivante.")

	def transcription_kraken(self,
							 image:str,
							 current_page:int,
							 model=None,
							 return_alto=False) -> OCRRecord:
		"""
		On segmente et on transcrit avec kraken
		:param image: Le chemin vers l'image
		:param transcription_only: faut-il lancer la transcription uniquement ?
		:return:
		"""
		if not model:
			model = self.kraken_ocr_model
		assert os.path.isfile(model), f"No model named '{model}'"
		loaded_page = Image.open(image)
		kraken_ocr = KRAKEN.KRAKEN(segmentation_model=self.kraken_lines_model[current_page],
								   ocr_model=model,
								   device=self.device)
		baseline = kraken_ocr.segment_lines_with_kraken(image=loaded_page)
		if return_alto is True:
			preds = kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline, return_kraken_preds=True, image_name=image.split("/")[-1])
			return kraken_ocr.serialize(preds)
		else:
			return kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline, return_kraken_preds=False, extract_polygons=True)




	def transcribe_to_alto(self,
						   page:str):
		"""
		Fonction wrapper de transcription d'une page
		:param page: La page à transcrire
		:param show_image: Montrer l'image transcrite avec les lignes ?
		:return:
		"""
		print("Cas 1")
		print(f"Segmentation/Transcription with kraken of page {page['image_path']}")
		try:
			current_page = int(page['classe'].split("_")[-1])
		except ValueError:
			current_page = "autre"
		return self.transcription_kraken(
			image=page["image_path"],
			current_page=current_page,
		return_alto=True)

	def process_additions(self, page:json, show_image=False):
		"""
		Cette fonction gère les ajouts postérieurs.
		:param page: the page metadata as json
		:param show_image: montrer l'image ou pas.
		:return:
		"""

		# On segmente la page 1: boxes générales
		print(f"Checking additions")


		zones_identifiees, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=["MarginTextZone-ajout"],
																		   confidence=0.6,
																		   model=self.yolo_models["ajouts"],
																		   show_image=False)
		zone_dict = {}
		zone_dict["zones_manquantes"] = zones_manquantes
		if len(zones_manquantes) == 0:
			transcription = self.transcription_kraken(
				image=page["image_path"],
				current_page=0,
				model=self.kraken_gloses_model,
				return_alto=False
			)
			# utils.draw_lines_on_image(image_path=page["image_path"],
			# 						  baselines=[item.baseline for item in transcription])
			ordered_lines = []
			for annotation in zones_identifiees:
				# On va commencer par filtrer les lignes dans la zone.
				zones_filtrees_as_rectangle = self.rectangle(annotation.coordinates[0][0],
															 annotation.coordinates[0][1],
															 annotation.coordinates[1][0],
															 annotation.coordinates[1][1])
				filtered_lines = utils.match_lines_in_zones(ocr_prediction=transcription,
															zone_as_rectangle=zones_filtrees_as_rectangle,
															intersect_ratio=0.3)
				print("---")
				as_record = OCRRecord()
				as_record.recreate_record(filtered_lines)
				sorted_lines = utils.sort_lines_with_rotation(lines_as_record=as_record, zone=zones_filtrees_as_rectangle)
				ordered_lines.append(sorted_lines)
			return list(zip(zones_identifiees, ordered_lines))
		else:
			return None

	def update_label(self, transcription):
		try:
			default_line = transcription.xpath("//alto:Tags/alto:OtherTag[@LABEL = 'default']", namespaces=self.alto_ns)[
				-1]
			default_line.set('LABEL', "DefaultLine")
		except IndexError:
			return transcription
		return transcription


	def merge_transcriptions(self,
									  transcription_1,
									  zones_et_lignes):
		"""
		Cette fonction met à jour une sérialisation ALTO avec le résultat de transcription.
		On ajoute les lignes supplémentaires à la fin de l'ALTO, indépendamment de leur
		:param transcription_1:
		:param lignes_ajout:
		:return:
		"""
		# On enlève la déclaration XML
		transcription_finale = transcription_1
		transcription_finale = self.update_label(transcription_finale)
		tous_blocs_ajouts = transcription_finale.xpath(f"//alto:TextBlock[@TAGREFS='{self.reverse_tagrefs_dict['MarginTextZone-ajout']}']", namespaces=self.alto_ns)
		for bloc in tous_blocs_ajouts:
			for idx, (zone, lignes) in enumerate(zones_et_lignes):
				if bloc.xpath("@ID")[0] != f"AJOUT_{idx}":
					continue
				for line in lignes:
					created_line = ET.Element("TextLine")
					created_line.set("BASELINE", utils.convert_baseline_coordinates_to_alto(line.baseline))
					hpos, vpos = min([item[0] for item in line.polygon]), min([item[1] for item in line.polygon])
					height = max([item[0] for item in line.polygon]) - min([item[0] for item in line.polygon])
					width = max([item[1] for item in line.polygon]) - min([item[1] for item in line.polygon])
					created_line.set("HPOS", str(hpos))
					created_line.set("ID", f"l_{uuid.uuid4()}")
					created_line.set("VPOS", str(vpos))
					created_line.set("HEIGHT", str(height))
					created_line.set("WIDTH", str(width))
					created_line.set("TAGREFS", self.reverse_tagrefs_dict['CustomLine:addition'])
					shape = ET.Element("Shape")
					polygon = ET.Element("Polygon")
					string = ET.Element("String")
					string.set("CONTENT", line.prediction)
					polygon.set("POINTS", utils.convert_baseline_coordinates_to_alto(line.polygon))
					shape.append(polygon)
					created_line.append(shape)
					created_line.append(string)
					bloc.append(created_line)
		return transcription_finale






	def workflow(self, images:list):
		"""
		La fonction qui classe les pages, produit les minutes
		et distribue les tâches en fonction de la classe de la page
		:param images: Les images à traiter
		:param target: [DEBUG] l'image à traiter dans le corpus
		:param start_after: [DEBUG] commencer le traitement avec l'image X
		:return:
		"""
		print("Début du workflow")

		try:
			os.mkdir("results/alto_results")
		except (IsADirectoryError, FileExistsError):
			pass
		self.images_basedir = "_".join(images[0].split("/")[:-1])
		self.classification_images(images)
		self.regroupement_minutes(out_dir=f"results/{self.images_basedir}_minutes.json")
		print("Pages classées, minutes regroupées")
		for minute_id, pages in self.minutes.items():
			for page in pages:
				print("---")
				print(f"Treating {page}")
				boxes, _ =	 self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=["MarginTextZone-ajout"],
																		   confidence=0.1,
																		   model=self.global_yolo_model,
																		   show_image=False)
				lignes_ajoutees_zones_ajouts = self.process_additions(page=page)
				alto_transcription = self.transcribe_to_alto(page=page)
				alto_transcription = "\n".join([line for line in alto_transcription.split("\n")[1:]])
				alto_transcription = ET.fromstring(alto_transcription)
				# with open("file.xml", "w") as output_file:
				# 	output_file.write(alto_transcription)
				# alto_transcription = ET.parse("file.xml")
				if lignes_ajoutees_zones_ajouts:
					additions = [ajouts for ajouts, lignes in lignes_ajoutees_zones_ajouts]
				else:
					additions = None
				alto_transcription = self.insert_zones(zones=boxes, transcription=alto_transcription, additions=additions)
				if lignes_ajoutees_zones_ajouts:
					alto_transcription = self.merge_transcriptions(transcription_1=alto_transcription,
											  zones_et_lignes=lignes_ajoutees_zones_ajouts)
				else:
					alto_transcription = self.update_label(alto_transcription)
				with open(f"results/alto_results/{page['image_path'].split('/')[-1].split('.')[0]}.xml", "w") as output_xml:
					output_xml.write(ET.tostring(alto_transcription, pretty_print=True, encoding='utf-8').decode())
				shutil.copy(page["image_path"], f"results/alto_results/")

	def insert_zones(self, zones, transcription, additions):
		tags = ET.fromstring(self.segmOnto_labels)
		transcription.insert(1, tags)
		all_zones = [item for item in zones] + [item for item in additions] if additions else zones
		print(all_zones)
		n_zone = 0
		for idx, zone in enumerate(all_zones):
			if zone.label == "MarginTextZone-addition":
				continue
			new_zone = ET.Element("{http://www.loc.gov/standards/alto/ns-v4#}TextBlock")
			new_zone.set("HPOS", str(zone.coordinates[0][0]))
			new_zone.set("VPOS", str(zone.coordinates[0][1]))
			new_zone.set("WIDTH", str(zone.coordinates[1][0] - zone.coordinates[0][0]))
			new_zone.set("HEIGHT", str(zone.coordinates[1][1] - zone.coordinates[0][1]))
			if zone.label != "MarginTextZone-ajout":
				new_zone.set("ID", f"ID_{idx}")
			else:
				new_zone.set("ID", f"AJOUT_{n_zone}")
				n_zone += 1
			new_zone.set("TAGREFS", self.reverse_tagrefs_dict[zone.label])
			new_zone.set("COORDS", utils.convert_baseline_coordinates_to_alto(zone.coordinates))
			printSpace = transcription.xpath("//alto:PrintSpace", namespaces=self.alto_ns)[0]
			printSpace.append(new_zone)
			for idx, line in enumerate(transcription.xpath("//alto:TextLine", namespaces=self.alto_ns)):
				baseline = line.xpath("@BASELINE")[0]
				baseline = utils.convert_alto_coordinates_to_baseline(baseline)
				# Si la ligne de base comprend plus d'un point, on simplifie en prenant les extrémités
				converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
				if additions:
					for addition in additions:
						addition_as_rectangle = self.rectangle(addition.coordinates[0][0],
														   addition.coordinates[0][1],
														   addition.coordinates[1][0],
														   addition.coordinates[1][1])
						is_in_box = utils.check_if_line_in_box(box_coord=addition_as_rectangle,
														 baseline=converted_baseline,
														 intersect_ratio=0.7)
						if is_in_box:
							line.getparent().remove(line)



				zone_as_rectangle = self.rectangle(zone.coordinates[0][0],
															 zone.coordinates[0][1],
															 zone.coordinates[1][0],
															 zone.coordinates[1][1])
				is_in_box = utils.check_if_line_in_box(box_coord=zone_as_rectangle,
												 baseline=converted_baseline,
												 intersect_ratio=0.7)

				if is_in_box:
					new_zone.append(line)

		return transcription

def main(images:list,
		 debug:bool=False,
		 device:str='cpu'):

	yolo_models = {
		"page_1": "src/Vision/models/yolov12_page_1.pt",
		"magistrats": "src/Vision/models/yolov11_table_magistrats.pt",
		"page_2": "src/Vision/models/yolov11_page_2.pt",
		"page_3": "src/Vision/models/yolo26x_page_3.pt",
		"page_4": "src/Vision/models/yolo26_page_4.pt",
		"ajouts": "src/Vision/models/yolo26_ajouts.pt"
	}
	pipeline = Pipeline(page_classifier_model="src/Page_Classifier/models/PageClassifier_RF.joblib",
						page_classifier_vocab="src/Page_Classifier/models/vocab_RF.joblib",
						yolo_models=yolo_models,
						debug=debug,
						device=device)
	pipeline.workflow(images)
	print(f"Images: {images} done.")


if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("-i", "--images", help="Input folder")
	arguments.add_argument("-db", "--debug", help="Debug mode", default=False)
	arguments.add_argument("-rs", "--resegment", help="Launch new segmentation", default=False)
	arguments.add_argument("-rt", "--retranscribe", help="Launch new transcription", default=False)
	arguments.add_argument("-d", "--device", help="Device", default=1)
	arguments.add_argument("-w", "--workers", help="Number of workers", default="cpu")
	arguments.add_argument("-c", "--clusters", help="Number of images per worker", default=8)
	arguments = arguments.parse_args()
	images_dir = arguments.images
	device = arguments.device
	workers = arguments.workers
	clusters = int(arguments.clusters)
	resegment = arguments.resegment
	retranscribe = True if arguments.retranscribe == "True" else False
	debug = True if arguments.debug == "True" else False
	images = glob.glob(f"{images_dir}/*.jpg")
	images.sort(key=lambda x: int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	grouped_images = [images[idx:idx + clusters] for idx in range(0, len(images), clusters)]
	print(grouped_images)
	with mp.Pool(processes=int(workers)) as pool:
		data = [(images, False, device) for images in grouped_images]
		pool.starmap(main, data)
	# main(images[0:2], debug=False, device="cpu")
	with zipfile.ZipFile('results/files.zip', 'w') as myzip:
		all_files = list(set([item.split('.')[0].split('/')[-1] for item in glob.glob(f"results/alto_results/*")]))
		all_files.sort(key=lambda x: int(x))
		for file in all_files:
			myzip.write(f"results/alto_results/{file}.xml")
			myzip.write(f"results/alto_results/{file}.jpg")
