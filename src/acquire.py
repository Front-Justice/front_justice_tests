import os
import sys
import utils.utils as utils
import Page_Classifier.page_classifier as PC
import Vision.KRAKEN as KRAKEN
import Vision.PARTY as PARTY
import Information_Extractor.extract as extract
import glob
import PIL.Image as Image

import Vision.yolo as yolo

class Pipeline():
	def __init__(self,
				 page_classifier_model,
				 page_classifier_vocab,
				 yolo_models):
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
		self.extractor = extract.Extractor()

		self.party = PARTY.PartyPredict()

	def load_image(self, image):
		self.current_image_path = image

	def classify_image(self):
		self.current_page_type = self.page_classifier.predict(image=self.current_image_path)


	def classification_images(self, images):
		"""
		Cette fonction classe toutes les images à l'aide d'un Random Forest
		:param images:
		:return:
		"""
		# On commence par classer toutes les images du dossier
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

	def regroupement_minutes(self):
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
		utils.save_as_dict(self.minutes, "results/minutes.json")

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
			exit(0)

	def transcription_kraken(self, image):
		loaded_page = Image.open(image)
		kraken_ocr = KRAKEN.KRAKEN(segmentation_model=self.kraken_lines_model,
								   ocr_model=self.kraken_ocr_model)
		baseline = kraken_ocr.segment_lines_with_kraken(image=loaded_page)
		return kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline)


	def traitement_p_1(self, page):
		"""
		Extraction d'information de la première page du procès
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
		target_transcription = f"results/ocr_prediction_{page['image_path'].replace('/', '_').replace('.jpg', '.json')}"
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
		zones_page_1, zones_manquantes = self.YOLO_Segmenter_P1.segment_zones(page["image_path"],
																			  target_classes=classes_page_1,
																			  confidence=0.5,
																			  model=self.yolo_models["page_1"],
																			  show_image=False)
		current_dict["general"] = zones_page_1
		current_dict["manquantes"] = zones_manquantes



		current_dict["Lieu du jugement"] = self.extractor.extraire_lieu_jugement(ocr_prediction=self.current_page_transcription,
																			  annotations=zones_page_1,
																			  image=page["image_path"],
																			  show_images=False,
																			  loaded_image=loaded_image,
																			  party_engine=self.party)


		# On extrait le numéro d'ordre
		current_dict["Numéro d'ordre"] = self.extractor.extraire_numero_ordre(ocr_prediction=self.current_page_transcription,
																			  annotations=zones_page_1,
																			  image=page["image_path"],
																			  show_images=False,
																			  loaded_image=loaded_image,
																			  party_engine=self.party)


		# On s'occupe de la table des magistrats
		classes_magistrats = ["ligne",
							  "Colonne"]
		magistrats, _ = self.YOLO_Segmenter_P1.segment_zones(page["image_path"],
																			target_classes=classes_magistrats,
																			confidence=0.2,
																			model=self.yolo_models["magistrats"],
																			show_image=False)



		# On extrait les noms de magistrats
		current_dict["magistrats"] = self.extractor.extraire_table_magistrats(ocr_prediction=self.current_page_transcription,
																			  zones_magistrats=magistrats,
																			  image=page["image_path"],
																			  show_images=False)



		# On extrait le numéro de jugement
		current_dict["Numéro de jugement"] = self.extractor.extraire_numero_jugement(ocr_prediction=self.current_page_transcription,
																			  annotations=zones_page_1,
																			  image=page["image_path"],
																			  show_images=False,
																			  loaded_image=loaded_image,
																			  party_engine=self.party)

		# Puis le nom et prénom du soldat
		current_dict["Nom du soldat"] = self.extractor.extraire_nom_soldat(ocr_prediction=self.current_page_transcription,
																		   annotations=zones_page_1,
																		   image=page["image_path"],
																		   show_images=False,
																		   loaded_image=loaded_image,
																		   party_engine=self.party)


		return current_dict

	def traitement_p2(self):
		pass

	def traitement_p3(self):
		pass

	def traitement_p4(self):
		pass

	def traitement_p_autre(self):
		pass


	def workflow(self, images):
		self.classification_images(images)
		self.regroupement_minutes()

		for minute_id, pages in self.minutes.items():
			for page in pages:
				if page["classe"] == "page_1":
					annotations = self.traitement_p_1(page)
					page["annotations"] = annotations
				utils.save_as_dict(self.minutes, "results/minutes_annotations.json")
		exit(0)


def main(images_dir):
	images = glob.glob(f"{images_dir}/*.jpg")
	images.sort(key=lambda x:int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	yolo_models = {
		"page_1": "src/Vision/models/yolov11_page_1.pt",
		"magistrats": "src/Vision/models/yolov11_table_magistrats.pt",
	}
	pipeline = Pipeline(page_classifier_model="src/Page_Classifier/models/PageClassifier.joblib",
						page_classifier_vocab="src/Page_Classifier/models/vocab.joblib",
						yolo_models=yolo_models)
	pipeline.workflow(images)

if __name__ == '__main__':
	images_dir = sys.argv[1]
	main(images_dir)