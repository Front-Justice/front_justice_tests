import argparse
import json
import os
import sys
import utils.utils as utils
import Page_Classifier.page_classifier as PC
import Vision.KRAKEN as KRAKEN
import Information_Extractor.extract as extract
import src.Vision.PARTY as PARTY
import glob
import PIL.Image as Image

import Vision.yolo as yolo


class Pipeline():
	def __init__(self,
				 page_classifier_model,
				 page_classifier_vocab,
				 yolo_models,
				 debug:bool = False,
				 use_party=True):
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
			self.yolo_models[name] = yolo.load(path)
		self.YOLO_Segmenter_P1 = yolo.YOLOSegmenter(models=self.yolo_models)

		# Les modèles d'OCR
		self.kraken_lines_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/lignes_updated.mlmodel"
		self.kraken_ocr_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/ocr_updated_150p.mlmodel"
		self.party_model = "/home/mgl/Bureau/Travail/scripts_et_programmes/party/models/final.safetensors"

		# L'outil d'extraction de l'information
		self.resize_factor = 1
		if debug == True:
			party_engine = None
		else:
			party_engine = PARTY.PartyPredict()
		self.extractor = extract.Extractor(party_engine=party_engine,
										   resize_factor=self.resize_factor,
										   debug=debug,
										   use_party=use_party)

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
		for image in images:
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
		print(self.minutes)
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

	def transcription_kraken(self, image):
		loaded_page = Image.open(image)
		kraken_ocr = KRAKEN.KRAKEN(segmentation_model=self.kraken_lines_model,
								   ocr_model=self.kraken_ocr_model)
		baseline = kraken_ocr.segment_lines_with_kraken(image=loaded_page)
		return kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline)

	def traitement_p_1(self, page):
		"""
		Extraction d'information de la première page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 1: boxes générales
		print("---")
		print(f"Treating {page}")
		# On s'occupe d'abord de la transcription des lignes

		# TODO: à supprimer, éventuellement, utile pour le debug. On vérifie si le fichier n'existe pas
		# Il faut re-lancer les prédictions en cas de nouveau modèle d'HTR/Segmentation
		target_transcription = f"results/ocr_predictions/{page['image_path'].replace('/', '_').replace('.jpg', '.json')}"
		if not os.path.isfile(target_transcription):
			print("Segmentation/Transcription with kraken")
			self.current_page_transcription = self.transcription_kraken(image=page["image_path"])
			utils.save_as_dict(self.current_page_transcription, target_transcription)
		else:
			print("Found existing kraken transcription")
			self.current_page_transcription = utils.load_json_to_dict(target_transcription)
		# Puis ont travaille sur les zones qu'on extrait entièrement

		# La liste qui suit permet de vérifier si une zone est manquante.
		classes_page_1 = ["Description du Soldat",
						  "Inculpation_antecedents",
						  "Magistrats",
						  "MainZone-crimeDate",
						  "MainZone-judgementNumber",
						  "MainZone-judgementPlace",
						  "MainZone-orderNumber",
						  "Nom du soldat"]
		current_dict = {}
		loaded_image = Image.open(page["image_path"])
		width, height = loaded_image.size

		# Tests de redimensionnement des images pour accélérer l'inférence avec Party, pas convainquant:
		# 2 à 3s sur 30 pour un facteur de 3 sur CPU.
		new_size = (int(width / self.resize_factor), int(height / self.resize_factor))
		loaded_image = loaded_image.resize(new_size)
		zones_page_1, zones_manquantes = self.YOLO_Segmenter_P1.segment_zones(page["image_path"],
																			  target_classes=classes_page_1,
																			  confidence=0.5,
																			  model=self.yolo_models["page_1"],
																			  show_image=False)
		current_dict["general"] = zones_page_1
		current_dict["zones_manquantes"] = zones_manquantes

		current_dict["date_proces"] = self.extractor.extraire_date_du_proces(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_1,
			image=page["image_path"],
			show_images=False,
			loaded_image=loaded_image)
		return current_dict

		# On extrait le numéro d'ordre en premier, cas il y a une vérification de la classification.
		if "MainZone-orderNumber" in current_dict["zones_manquantes"]:
			current_dict["numer_ordre"] = None
		else:
			current_dict["numer_ordre"] = self.extractor.extraire_numero_ordre(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)

		# On extrait les noms de magistrats
		if "Magistrats" in current_dict["zones_manquantes"]:
			current_dict["magistrats"] = None
		else:
			classes_magistrats = ["ligne",
								  "Colonne"]
			magistrats, _ = self.YOLO_Segmenter_P1.segment_zones(page["image_path"],
																 target_classes=classes_magistrats,
																 confidence=0.5,
																 model=self.yolo_models["magistrats"],
																 show_image=False)
			current_dict["magistrats"] = self.extractor.extraire_magistrats(
				ocr_prediction=self.current_page_transcription,
				zones_magistrats=magistrats,
				image=page["image_path"],
				show_images=False)


		# On extrait le nom et prénom du soldat
		# TODO: normaliser les noms de zone
		if "Nom du soldat" in current_dict["zones_manquantes"]:
			current_dict["nom_du_soldat"] = None
		else:
			current_dict["nom_du_soldat"] = self.extractor.extraire_nom_soldat(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)



		# On extrait la date du crime
		# TODO: normaliser les noms de zone
		if "MainZone-crimeDate" in current_dict["zones_manquantes"]:
			current_dict["date_du_crime_ou_delit"] = None
		else:
			current_dict["date_du_crime_ou_delit"] = self.extractor.extraire_date_crime(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)

		# On extrait le lieu du jugement
		# TODO: normaliser les noms de zone
		if "MainZone-judgementPlace" in current_dict["zones_manquantes"]:
			current_dict["lieu_du_jugement"] = None
		else:
			current_dict["lieu_du_jugement"] = self.extractor.extraire_lieu_jugement(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)





		# On extrait le numéro de jugement
		# TODO: normaliser les noms de zone
		if "MainZone-judgementNumber" in current_dict["zones_manquantes"]:
			current_dict["Numéro de jugement"] = None
		else:
			current_dict["Numéro de jugement"] = self.extractor.extraire_numero_jugement(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)

		return current_dict

	def traitement_p2(self):
		pass

	def traitement_p3(self):
		pass

	def traitement_p4(self):
		pass

	def traitement_p_autre(self):
		pass

	def workflow(self, images:list, target:str|None):
		"""
		La fonction qui classe les pages, produit les minutes
		et distribue les tâches en fonction de la classe de la page
		:param images: Les images à traiter
		:param target: [DEBUG] l'image à traiter dans le corpus
		:return:
		"""
		print("Début du workflow")
		# Il faudra supprimer ça pour la mise en production
		self.images_basedir = "_".join(images[0].split("/")[:-1])
		if os.path.isfile(f"results/{self.images_basedir}_minutes.json"):
			self.minutes = utils.load_json_to_dict(f"results/{self.images_basedir}_minutes.json")
		else:
			self.classification_images(images)
			self.regroupement_minutes(out_dir=f"results/{self.images_basedir}_minutes.json")
		print("Pages classées, minutes regroupées")

		for minute_id, pages in self.minutes.items():
			for page in pages:
				if target:
					if page['image_path'] != target:
						continue
				if page["classe"] == "page_1":
					annotations = self.traitement_p_1(page)
					page["annotations"] = annotations
				utils.save_as_dict(self.minutes, f"results/{self.images_basedir}_minutes_annotations.json")
		exit(0)


def main(images_dir:str, target:str=None, debug:bool=False, use_party:bool=True):
	images = glob.glob(f"{images_dir}/*.jpg")
	if target:
		images = [item for item in images if item == target]
	else:
		target = None
	try:
		images.sort(key=lambda x: int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	except:
		images.sort(key= lambda x: int(x.split("/")[-1].split(".jpg")[0]))
	yolo_models = {
		"page_1": "src/Vision/models/yolov11_page_1.pt",
		"magistrats": "src/Vision/models/yolov11_table_magistrats.pt",
	}
	pipeline = Pipeline(page_classifier_model="src/Page_Classifier/models/PageClassifier.joblib",
						page_classifier_vocab="src/Page_Classifier/models/vocab.joblib",
						yolo_models=yolo_models,
						debug=debug,
						use_party=use_party)
	pipeline.workflow(images, target)


if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("-i", "--images", help="Input folder")
	arguments.add_argument("-d", "--debug", help="Debug mode", default=False)
	arguments.add_argument("-t", "--target", help="Target one specific file", default=None)
	arguments.add_argument("-up", "--use_party", help="Use party to confirm key OCR predictions", default=True)
	arguments = arguments.parse_args()
	images_dir = arguments.images
	target = arguments.target
	use_party = True if arguments.use_party == "True" else False
	debug = True if arguments.debug == "True" else False
	main(images_dir, target, debug, use_party)
