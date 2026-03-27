import argparse
import copy
import os
import shutil

import tqdm

import torch.cuda
import zipfile
import utils.utils as utils
import Page_Classifier.page_classifier as PC
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
				 retranscribe=False):
		self.debug = debug
		self.page_classifier = PC.PageClassifier(build_vocab=False,
												 model=page_classifier_model,
												 vocab=page_classifier_vocab)
		self.current_image = None
		self.current_image_path = None
		self.current_page_transcription = None
		self.minutes = {}
		self.images_name_list = []
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
		self.kraken_lines_model = {
			 0: "/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/scripts/src/Vision/models/lignes_ajouts.mlmodel",
			 1: "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/modele_page_1_200p_best.mlmodel",
			 2: "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/lignes_page_2.mlmodel",
			 3: "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/lignes_page_3.mlmodel",
			 4: "/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/scripts/src/Vision/models/modele_page_4.mlmodel"
			 }
		self.kraken_ocr_model = "src/Vision/models/htr_29500l.mlmodel"
		self.kraken_gloses_model = "src/Vision/models/strate_2_3000l.mlmodel"
		self.party_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/scripts/src/Vision/models/model.safetensors"
		self.minutes_annotation_file = ""
		# L'outil d'extraction de l'information
		self.resize_factor = 1
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.extractor = extract.Extractor(party_engine=None,
										   kraken_model_annotations=self.kraken_gloses_model,
										   kraken_model_transcription=self.kraken_ocr_model,
										   resize_factor=self.resize_factor,
										   debug=debug,
										   device=device,
										   minutier=self.minutes)

	def reaffecter_dictionnaire(self, minute_courante):
		"""
		Cette fonction met à jour le dictionnaire dans la classe extractor.
		:return:
		"""
		self.extractor.update_dict(minute_courante)  # Met à jour B

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
							 return_alto=True) -> OCRRecord:
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
								   ocr_model=model)
		baseline = kraken_ocr.segment_lines_with_kraken(image=loaded_page)
		if return_alto:
			preds = kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline, return_kraken_preds=True, image_name=image.split("/")[-1])
			return kraken_ocr.serialize(preds)
		else:
			return kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline, return_kraken_preds=True)




	def transcribe_to_alto(self,
						   page:str,
						   extract_polygons:bool = False):
		"""
		Fonction wrapper de transcription d'une page
		:param page: La page à transcrire
		:param show_image: Montrer l'image transcrite avec les lignes ?
		:return:
		"""
		print("Cas 1")
		print(f"Segmentation/Transcription with kraken of page {page['image_path']}")
		return self.transcription_kraken(
			image=page["image_path"],
			current_page=int(page['classe'].split("_")[-1]))

	def process_additions(self, page:json, show_image=False):
		"""
		Cette fonction gère les ajouts postérieurs.
		:param page: the page metadata as json
		:param show_image: montrer l'image ou pas.
		:return:
		"""

		# On segmente la page 1: boxes générales
		print(f"Checking additions")


		_, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=["MarginTextZone-ajout"],
																		   confidence=0.1,
																		   model=self.yolo_models["ajouts"],
																		   show_image=False)


		zone_dict = {}
		zone_dict["zones_manquantes"] = zones_manquantes
		if len(zones_manquantes) == 0:
			return self.transcription_kraken(
				image=page["image_path"],
				current_page=0,
				model=self.kraken_gloses_model
			)
		else:
			return None

	def merge_transcriptions(self,
									  transcription_1,
									  transcription_2):
		"""
		Cette fonction met à jour une sérialisation ALTO avec le résultat de transcription.
		On ajoute les lignes supplémentaires à la fin de l'ALTO, indépendamment de leur
		:param alto_serialization:
		:param transcription_json:
		:return:
		"""
		alto_ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
		# On enlève la déclaration XML
		transcription_1 = "\n".join([line for line in transcription_1.split("\n")[1:]])
		transcription_2 = "\n".join([line for line in transcription_2.split("\n")[1:]])
		transcription_finale = ET.fromstring(transcription_1)
		default_line = transcription_finale.xpath("//alto:Tags/alto:OtherTag[@LABEL = 'default']", namespaces=alto_ns)[-1]
		default_line.set('LABEL', "DefaultLine")
		added_lines = ET.Element("OtherTag")
		added_lines.set("LABEL", "CustomLine:addition")
		added_lines.set("ID", "TYPE_2")
		default_line.addnext(added_lines)
		transcription_cible = ET.fromstring(transcription_2)
		all_lines_cible = transcription_cible.xpath("//alto:TextLine", namespaces=alto_ns)
		text_bloc_finale = transcription_finale.xpath("//alto:TextBlock", namespaces=alto_ns)[-1]
		for line in all_lines_cible:
			line.set("TAGREFS", "TYPE_2")
			text_bloc_finale.insert(-1, line)
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
		# Il faudra supprimer ça pour la mise en production
		self.images_basedir = "_".join(images[0].split("/")[:-1])
		self.classification_images(images)
		self.regroupement_minutes(out_dir=f"results/{self.images_basedir}_minutes.json")
		print("Pages classées, minutes regroupées")
		for minute_id, pages in self.minutes.items():
			for page in pages:
				print("---")
				print(f"Treating {page}")
				alto_transcription = self.transcribe_to_alto(page=page)
				lignes_ajoutees = self.process_additions(page=page)
				if lignes_ajoutees:
					print(lignes_ajoutees)
					alto_transcription = self.merge_transcriptions(transcription_1=alto_transcription,
											  transcription_2=lignes_ajoutees)
				else:
					alto_transcription = "\n".join([line for line in alto_transcription.split("\n")[1:]])
					alto_transcription = ET.fromstring(alto_transcription)
				with open(f"results/alto_results/{self.images_basedir}.xml", "w") as output_xml:
					output_xml.write(ET.tostring(alto_transcription, pretty_print=True, encoding='utf-8').decode())
				shutil.copy(page["image_path"], f"results/alto_results/")
		with zipfile.ZipFile('files.zip', 'w') as myzip:
			for file in glob.glob(f"results/alto_results/*"):
				myzip.write(file)



def main(images_dir:str,
		 debug:bool=False):
	images = glob.glob(f"{images_dir}/*.jpg")
	# Attention, cette façon de trier ne peut fonctionner qu'au sein d'un même minutier
	try:
		images.sort(key=lambda x: int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	except:
		images.sort(key= lambda x: int(x.split("/")[-1].split(".jpg")[0]))
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
						debug=debug)
	pipeline.workflow(images)


if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("-i", "--images", help="Input folder")
	arguments.add_argument("-d", "--debug", help="Debug mode", default=False)
	arguments.add_argument("-rs", "--resegment", help="Launch new segmentation", default=False)
	arguments.add_argument("-rt", "--retranscribe", help="Launch new transcription", default=False)
	arguments = arguments.parse_args()
	images_dir = arguments.images
	resegment = arguments.resegment
	retranscribe = True if arguments.retranscribe == "True" else False
	debug = True if arguments.debug == "True" else False
	main(images_dir, debug)
