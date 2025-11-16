import os
import sys
import utils.utils as utils
import Page_Classifier.page_classifier as PC
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
		self.minutes = {}
		self.images_name_list = []
		self.current_image_idx = 0
		self.pages_classees = []

		# Modèles
		self.yolo_models = {}
		for name, path in yolo_models.items():
			assert os.path.exists(path), f"{path} n'existe pas."
			self.yolo_models[name] = yolo.load(path)


	def load_image(self, image):
		# self.current_image = Image.open(image)
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
			result = self.check_image_consistency(ident)
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
			if classe == "page_4" and self.pages_classees[idx + 1][1] == "page_1":
				print("Minute terminée")
				self.minutes[current_minute_number] = current_minute
				current_minute = []
			elif classe == "page_4" and self.pages_classees[idx + 1][1] == "page_autre":
				print("Un document autre est adjoint. On vérifie s'il appartient à la minute en cours")
			else:
				print("On continue")
		print(self.minutes)

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

	def traitement_p1(self, page):
		"""
		Extraction d'information de la première page du procès
		:return:
		"""
		print(page)
		YOLO_Segmenter_P1 = yolo.YOLOSegmenter(model=self.yolo_models["page_1"])
		YOLO_Segmenter_P1.segment(page["image_path"])
		exit(0)

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
					self.traitement_p1(page)
				elif page["classe"] != "page_autre":
					self.traitement_p_autre()



def main(images_dir):
	images = glob.glob(f"{images_dir}/*.jpg")
	images.sort(key=lambda x:int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	yolo_models = {
		"page_1": "Vision/segmentation_models/yolov11_page1.pt",
		"magistrats": "Vision/segmentation_models/yolov11_table_magistrats.pt",
	}
	pipeline = Pipeline(page_classifier_model="Page_Classifier/models/PageClassifier.joblib",
						page_classifier_vocab="Page_Classifier/models/vocab.joblib",
						yolo_models=yolo_models)
	pipeline.workflow(images)

if __name__ == '__main__':
	images_dir = sys.argv[1]
	main(images_dir)