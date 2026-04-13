import argparse
import copy
import os
import time

import PIL.Image
import tqdm
import torch
import multiprocessing as mp
import utils.utils as utils
import Page_Classifier.page_classifier as PC
import Vision.KRAKEN as KRAKEN
import Information_Extractor.extract as extract
import Information_Extractor.reconciliation as reconciliation
import glob
import PIL.Image as Image
import json

from torchvision import transforms
import Vision.YOLO as YOLO
from src.utils.utils import OCRRecord
import src.varia.Line_Deletion.deletions as deletions


class Pipeline():
	def __init__(self,
				 page_classifier_model,
				 page_classifier_vocab,
				 yolo_models,
				 debug:bool = False,
				 use_party=True,
				 resegment=False,
				 retranscribe=False,
				 device="cpu",
				 images_dir=None,
				 current_minute = None):
		self.debug = debug
		self.page_classifier = PC.PageClassifier(build_vocab=False,
												 model=page_classifier_model,
												 vocab=page_classifier_vocab)
		self.current_image = None
		self.current_image_path = None
		self.current_page_transcription = None
		self.minutes = current_minute
		self.device = device
		self.images_name_list = []
		self.current_image_idx = 0
		self.pages_classees = []
		self.images_basedir = images_dir.replace("/", "_")

		# Les modèles de zones
		self.yolo_models = {}
		for name, path in yolo_models.items():
			assert os.path.exists(path), f"{path} n'existe pas."
			self.yolo_models[name] = YOLO.load(path)
			self.yolo_models[name].to(device)
		self.YOLO_Segmenter = YOLO.YOLOSegmenter()

		# Les modèles d'OCR
		self.resegment = resegment
		self.retranscribe = retranscribe
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
		# L'outil d'extraction de l'information
		self.resize_factor = 1
		print(f"Use party: {use_party} and debug: {debug}")
		if debug == True or use_party == False:
			print("Setting party engine to None")
			party_engine = None
		else:
			party_engine = PARTY.PartyPredict()
		self.extractor = extract.Extractor(party_engine=party_engine,
										   kraken_model_annotations=self.kraken_gloses_model,
										   kraken_model_transcription=self.kraken_ocr_model,
										   resize_factor=self.resize_factor,
										   debug=debug,
										   use_party=use_party,
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

	def transcription_kraken(self, image:str,
							 transcription_only:bool,
							 current_page:int,
							 suffix="",
							 extract_polygons:bool=False,
							 model=None) -> OCRRecord:
		"""
		On segmente et on transcrit avec kraken
		:param image: Le chemin vers l'image
		:param transcription_only: faut-il lancer la transcription uniquement ?
		:return:
		"""
		if not model:
			model = self.kraken_ocr_model
		assert os.path.isfile(model), f"No model named '{model}'"
		# assert os.path.isfile(self.kraken_lines_model), f"No model named {self.kraken_lines_model}"
		segmentation_json = f'results/ocr_predictions/{image.replace("/", "_").replace(f"{suffix}.jpg", "_segments.pickle")}'
		loaded_page = Image.open(image)
		kraken_ocr = KRAKEN.KRAKEN(segmentation_model=self.kraken_lines_model[current_page],
								   ocr_model=model,
								   device=self.device)
		if transcription_only:
			baseline = utils.unpickle_object(path=segmentation_json)
		else:
			baseline = kraken_ocr.segment_lines_with_kraken(image=loaded_page)
			utils.pickle_object(obj=baseline, path=segmentation_json)
		return kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline, extract_polygons=extract_polygons)


	def traitement_p_2(self, page, show_image=False):
		"""
		Extraction d'information de la deuxième page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 2: boxes générales
		# On s'occupe d'abord de la transcription des lignes

		classes_page_2 = ["avertissement",
						  "defense",
						  "formalites",
						  "identite_soldat",
						  "Nom du soldat",
						  "questions",
						  "requisitoire",
						  "seance_ouverte"]
		current_dict = {}
		loaded_image = Image.open(page["image_path"])
		width, height = loaded_image.size
		# Tests de redimensionnement des images pour accélérer l'inférence avec Party, pas convainquant:
		# 2 à 3s sur 30 pour un facteur de 3 sur CPU.
		new_size = (int(width / self.resize_factor), int(height / self.resize_factor))
		loaded_image = loaded_image.resize(new_size)
		zones_page_2, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=classes_page_2,
																		   confidence=0.5,
																		   model=self.yolo_models["page_2"],
																		   show_image=False)

		current_dict["soldat"] = self.extractor.extraire_description_soldat_NER_p2(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			loaded_image=loaded_image)


		current_dict["defenseur"] = self.extractor.extraire_identite_defenseur(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			image=page["image_path"],
			loaded_image=loaded_image)


		current_dict["renseignements_procedure_complementaires"] = self.extractor.extraire_informations_procedure(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			image=page["image_path"],
			loaded_image=loaded_image)


		current_dict["requisitoire"] = self.extractor.extraire_requisitoire(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			image=page["image_path"],
			loaded_image=loaded_image)



		current_dict["questions"], identite = self.extractor.extraire_questions_p2(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			loaded_image=loaded_image)
		if identite:
			current_dict["soldat"]["identite"] = {**current_dict["soldat"]["identite"], **identite}
		zone_dict = {}
		zone_dict["zones_identifiees"] = zones_page_2.to_json()
		zone_dict["zones_manquantes"] = zones_manquantes
		return zone_dict, current_dict


	def traitement_p_3(self, page, show_image=False):
		"""
		Extraction d'information de la deuxième page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 3: boxes générales
		# On s'occupe d'abord de la transcription des lignes


		classes_page_3 = ["questions",
							"reponse_questions",
							"decision_tribunal"]
		current_dict = {}
		loaded_image = Image.open(page["image_path"])
		width, height = loaded_image.size
		# Tests de redimensionnement des images pour accélérer l'inférence avec Party, pas convainquant:
		# 2 à 3s sur 30 pour un facteur de 3 sur CPU.
		new_size = (int(width / self.resize_factor), int(height / self.resize_factor))
		loaded_image = loaded_image.resize(new_size)
		zones_page_3, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=classes_page_3,
																		   confidence=0.5,
																		   model=self.yolo_models["page_3"],
																		   show_image=False)



		if "questions" not in zones_manquantes:
			current_dict["questions"] = self.extractor.extraire_questions_p3(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_3,
				loaded_image=loaded_image)


		current_dict["reponse_questions"] = self.extractor.extraire_reponses_p3(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_3,
			loaded_image=loaded_image)


		current_dict["decision_tribunal"], current_dict["identite"] = self.extractor.extraire_decision_tribunal_p3(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_3,
			loaded_image=loaded_image)

		zone_dict = {}
		zone_dict["zones_identifiees"] = zones_page_3.to_json()
		zone_dict["zones_manquantes"] = zones_manquantes
		return zone_dict, current_dict

	def traitement_p_4(self, page, show_image=False):
		"""
		Extraction d'information de la deuxième page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 3: boxes générales
		# On s'occupe d'abord de la transcription des lignes


		classes_page_4 = ["recapitulatif_somme",
							"tableau_frais"]
		current_dict = {}
		loaded_image = Image.open(page["image_path"])
		width, height = loaded_image.size
		# Tests de redimensionnement des images pour accélérer l'inférence avec Party, pas convainquant:
		# 2 à 3s sur 30 pour un facteur de 3 sur CPU.
		new_size = (int(width / self.resize_factor), int(height / self.resize_factor))
		loaded_image = loaded_image.resize(new_size)

		if show_image == True:
			print("Show image")
			utils.draw_lines_on_image(image_path=page["image_path"],
									  baselines=[line.baseline for line in self.current_page_transcription])

		# current_dict["polygon_signature_greffier"] = self.extractor.extract_signature_greffier(ocr_prediction=self.current_page_transcription,
		# 										  image=page["image_path"])

		zones_page_4, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=classes_page_4,
																		   confidence=0.5,
																		   model=self.yolo_models["page_4"],
																		   show_image=False)

		current_dict["date_proces_1"] = self.extractor.extraire_date_1_p4(ocr_prediction=self.current_page_transcription)
		current_dict["identite"] = self.extractor.extraire_noms_p4(ocr_prediction=self.current_page_transcription)

		if "tableau_frais" not in zones_manquantes:
			current_dict["tableau_frais"], nom_2 = self.extractor.extraire_tableau_p4(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_4,
				loaded_image=loaded_image)

		try:
			current_dict["identite"] = {**current_dict["identite"], **nom_2}
		except TypeError:
			pass

		if "recapitulatif_somme" not in zones_manquantes:
			current_dict["dernier_paragraphe"], current_dict["date_proces_2"] = self.extractor.extraire_paragraphe_final_p4(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_4,
				loaded_image=loaded_image)





		zone_dict = {}
		zone_dict["zones_identifiees"] = zones_page_4.to_json()
		zone_dict["zones_manquantes"] = zones_manquantes
		return zone_dict, current_dict


	def transcription_page(self,
						   page:str,
						   show_image:bool = False,
						   force_resegment:bool = False,
						   extract_polygons:bool = False):
		"""
		Fonction wrapper de transcription d'une page
		:param page: La page à transcrire
		:param show_image: Montrer l'image transcrite avec les lignes ?
		:return:
		"""
		target_transcription = f"results/ocr_predictions/{page['image_path'].replace('/', '_').replace('.jpg', '.json')}"
		if not os.path.isfile(target_transcription) or self.resegment or self.retranscribe or force_resegment:
			print("Cas 1")
			print(f"Segmentation/Transcription with kraken of page {page['image_path']}")
			self.current_page_transcription = self.transcription_kraken(
				image=page["image_path"],
				transcription_only=self.resegment is False
								   and self.retranscribe is True,
				current_page=int(
					page['classe'].split("_")[-1],),
				extract_polygons=True
																		)

			utils.serialize_dict(self.current_page_transcription.to_json(), target_transcription)
		else:
			print("Found existing kraken transcription: " + target_transcription)
			print("Cas 2")
			self.current_page_transcription = OCRRecord()
			self.current_page_transcription.from_json(path=target_transcription)
		print(self.current_page_transcription)
		image = PIL.Image.open(page["image_path"])
		for line in self.current_page_transcription:
			cropped = utils.polygon_extraction(line.polygon, image, keep_alpha=False, return_image=True)
			cropped = cropped.convert("L")
			transform = deletions.transform(image_size=(65, 1500))
			normalized = transform(cropped)
			normalized = normalized.unsqueeze(1)
			preds = deletions.predict(model_path="src/varia/Line_Deletion/models/line_deletion.pth", image=normalized)
			if preds == "deleted":
				print("\n\n---")
				print("Deletion identified.")
				cropped.show()
				for char_poly, char in zip(line.cuts, line.prediction):
					current_char = utils.polygon_extraction(char_poly, image, keep_alpha=False, return_image=True, vertical_padding=12)
					current_char = current_char.convert("L")
					transform = deletions.transform(image_size=(65, 65))
					normalized = transform(current_char)
					normalized = normalized.unsqueeze(1)
					preds = deletions.predict(model_path="src/varia/Line_Deletion/models/chars_deletion.pth", image=normalized)
					if preds == "deleted":
						print("-")
						print(char)
						print(preds)

		if show_image:
			baselines = [line.baseline for line in self.current_page_transcription]
			utils.draw_lines_on_image(image_path=page["image_path"], baselines=baselines)


	def process_additions(self, page:json, show_image=False):
		"""
		Cette fonction gère les ajouts postérieurs.
		:param page: the page metadata as json
		:param show_image: montrer l'image ou pas.
		:return:
		"""

		# On segmente la page 1: boxes générales
		print(f"Checking additions")

		current_dict = {}

		zones_ajouts, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=["MarginTextZone-ajout"],
																		   confidence=0.1,
																		   model=self.yolo_models["ajouts"],
																		   show_image=False)


		zone_dict = {}
		zone_dict["zones_identifiées"] = zones_ajouts.to_json()
		zone_dict["zones_manquantes"] = zones_manquantes
		if len(zones_manquantes) == 0:
			target_transcription = f"results/ocr_predictions/{page['image_path'].replace('/', '_').replace('.jpg', '.ajouts.json')}"
			if not os.path.isfile(target_transcription):
				lignes_glosees = self.transcription_kraken(
					image=page["image_path"],
					transcription_only=False,
					current_page=0,
					suffix=".ajouts",
					model=self.kraken_gloses_model
				)
				utils.serialize_dict(lignes_glosees.to_json(), target_transcription)
			else:
				print("Found existing kraken transcription: " + target_transcription)
				print("Cas 2")
				lignes_glosees = OCRRecord()
				lignes_glosees.from_json(path=target_transcription)

			# baselines = [line.baseline for line in lignes_glosees]
			# utils.draw_lines_on_image(image_path=page["image_path"], baseline=baselines)
		else:
			return None, None

		informations_ajouts = self.extractor.extraire_informations_ajouts_posterieurs(ocr_prediction=lignes_glosees,
																					  annotations=zones_ajouts,
																					  image_path=page["image_path"])

		informations_ajouts = {"annotations_ajouts": informations_ajouts}
		return zone_dict, informations_ajouts



	def traitement_p_1(self, page, show_image=False):
		"""
		Extraction d'information de la première page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 1: boxes générales
		# On s'occupe d'abord de la transcription des lignes

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
		zones_page_1, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=classes_page_1,
																		   confidence=0.5,
																		   model=self.yolo_models["page_1"],
																		   show_image=False)

		zone_dict = {}
		zone_dict["zones_identifiées"] = zones_page_1.to_json()
		zone_dict["zones_manquantes"] = zones_manquantes

		if "Magistrats" not in zones_manquantes:
			current_dict["date_proces"] = self.extractor.extraire_date_du_proces_p1(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)

		# On extrait le nom et prénom du soldat
		if "Description du Soldat" in zones_manquantes:
			current_dict['soldat'] = None
		else:
			current_dict['soldat'] = self.extractor.extraire_description_soldat_NER_p1(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				loaded_image=loaded_image)
			# Production de corpus, à supprimer

		try:
			date_naissance = current_dict["soldat"]["identite"]["date_naissance"]["extracted"]["when"]
		except (KeyError, TypeError):
			date_naissance = None
		try:
			current_dict["soldat"]["identite"]["age"] = utils.calcule_age(date_naissance,
																   date_proces=current_dict["date_proces"]["date_normalisee"]["when"])
		except (TypeError, KeyError, ValueError):
			current_dict["soldat"]["identite"]["age"] = None



		# if "Inculpation_antecedents" in zones_manquantes or current_dict['soldat']['nom_du_soldat'] == "Plusieurs soldats":
		if "Inculpation_antecedents" in zones_manquantes:
			current_dict["Inculpation"], current_dict["Antécédents"] = None, None
		else:
			accusation_antecedents =  self.extractor.extraire_inculpation_et_antecedents(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)
			try:
				current_dict["chef_accusation"] = accusation_antecedents["inculpation"]
				current_dict["antécédents"] = accusation_antecedents["antécédents"]
			except TypeError:
				pass



		# On extrait le numéro d'ordre en premier, cas il y a une vérification de la classification.
		if "MainZone-orderNumber" in zones_manquantes:
			current_dict["numero_ordre"] = None
		else:
			current_dict["numero_ordre"] = self.extractor.extraire_numero_ordre(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)

		# On extrait les noms de magistrats
		if "Magistrats" in zones_manquantes:
			current_dict["magistrats"] = None
		else:
			classes_magistrats = ["ligne",
								  "Colonne"]
			magistrats, _ = self.YOLO_Segmenter.segment_zones(page["image_path"],
															  target_classes=classes_magistrats,
															  confidence=0.5,
															  model=self.yolo_models["magistrats"],
															  show_image=False)
			test_lignes = utils.test_number_of_zones(magistrats, label="ligne", number=6)
			if test_lignes is False:
				print("Warning: une des lignes du jury n'a pas été identifiée.")
			current_dict["magistrats"] = self.extractor.extraire_magistrats(
				ocr_prediction=self.current_page_transcription,
				zones_magistrats=magistrats,
				image=page["image_path"],
				show_images=False)





		# On extrait la date du crime
		if "MainZone-crimeDate" in zones_manquantes:
			current_dict["date_du_crime_ou_delit"] = None
		else:
			current_dict["date_du_crime_ou_delit"] = self.extractor.extraire_date_crime_ou_delit(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)

		# On extrait le lieu du jugement
		if "MainZone-judgementPlace" in zones_manquantes:
			current_dict["lieu_jugement"] = None
		else:
			current_dict["lieu_jugement"] = self.extractor.extraire_lieu_jugement(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)








		# On extrait le numéro de jugement
		# TODO: normaliser les noms de zone
		if "MainZone-judgementNumber" in zones_manquantes:
			current_dict["numero_jugement"] = None
		else:
			current_dict["numero_jugement"] = self.extractor.extraire_numero_jugement(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				image=page["image_path"],
				show_images=False,
				loaded_image=loaded_image)
		return zone_dict, current_dict



	def workflow(self, minute:dict, target:str|None=None, start_after:int=0):
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
		minute_number = list(minute.keys())[0]
		self.minutes_annotation_file = f"results/{self.images_basedir}_minutes_annotations_{minute_number}.json"
		self.minutes_reconciliees_file = f"results/{self.images_basedir}_minutes_annotations_{minute_number}_reconcilie.json"
		self.minutes_reconciliees = None
		image_index = 0
		previous_pages = None
		print(self.minutes)
		for minute_id, pages in self.minutes.items():
			for page in pages:
				if start_after > image_index:
					image_index += 1
					continue
				else:
					image_index += 1
				if target:
					if page['image_path'] != target:
						continue
				# Attention, cause un bug si la page n'est pas présente dans la liste. Effets non prévus.
				if page['classe'] in ["page_2", "page_1", "page_3", "page_4"]:
					print("---")
					print(f"Treating {page}")
					if page["classe"] == "page_4":
						force_resegment = True
						extract_polygons = True
					else:
						force_resegment = False
						extract_polygons = False
					force_resegment = False
					self.transcription_page(page=page,
											show_image=False,
											force_resegment=force_resegment,
											extract_polygons=extract_polygons)
					zones_ajouts, ajouts = self.process_additions(page=page)
					if ajouts is None:
						ajouts = {"ajouts": None}
					# utils.save_as_dict(self.minutes, self.minutes_annotation_file)
				if page["classe"] == "page_1":
					zones, annotations = self.traitement_p_1(page=page, show_image=False)
				elif page['classe'] == "page_2":
					zones, annotations = self.traitement_p_2(page=page, show_image=False)
				elif page['classe'] == "page_3":
					zones, annotations = self.traitement_p_3(page=page, show_image=False)
				elif page['classe'] == "page_4":
					zones, annotations = self.traitement_p_4(page=page, show_image=False)
				else:
					continue
				page["extractions"] = {**annotations, **ajouts}
				page["zones"] = zones
				self.reaffecter_dictionnaire(pages)
			if not target:
				reconciliator = reconciliation.Reconciliator(minute_list=pages, previous_minute=previous_pages)
				previous_pages = copy.copy(pages)
				reconciliator.reconciliate_minute()
				self.minutes_reconciliees = reconciliator.reconciliated_minute


def regroupement_minutes(pages_classees):
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
	minutes = {}
	# Puis on rassemble les minutes
	for idx, ((dossier, ident, image), classe) in enumerate(pages_classees):
		current_image = {}
		current_image["répertoire"] = dossier
		current_image["id"] = ident
		current_image["image_path"] = image
		current_image["classe"] = classe
		current_minute.append(current_image)
		if ident == pages_classees[-1][0][1]:
			print("Dossier terminé")
			minutes[current_minute_number] = current_minute
			break
		if classe in ["page_4", "page_autre"] and pages_classees[idx + 1][1] == "page_1":
			minutes[current_minute_number] = current_minute
			current_minute = []
			current_minute_number += 1
	return minutes
	# utils.save_as_dict(minutes, out_dir)

def classification_images(images, page_classifier_model, page_classifier_vocab):
	"""
	Cette fonction classe toutes les images à l'aide d'un Random Forest
	:param images: la liste d'images
	:return:
	"""
	# On commence par classer toutes les images du dossier
	print("Classification des images")
	images_name_list = []
	pages_classees =  []
	page_classifier = PC.PageClassifier(build_vocab=False,
											 model=page_classifier_model,
											 vocab=page_classifier_vocab)
	for image in tqdm.tqdm(images):
		dossier, ident = utils.get_name_from_path(image)
		# On vérifie s'il n'y a pas de problème de disparition d'image
		check_image_consistency(ident, images_name_list)
		images_name_list.append(ident)
		current_page_type = page_classifier.predict(image=image)
		pages_classees.append(((dossier, ident, image), current_page_type))
		if image == images[-1]:
			print("Dossier terminé")
	print(pages_classees)
	return pages_classees

def check_image_consistency(current_image, images_name_list):
	"""
	Cette fonction vérifie s'il y a un problème au sein des fichiers et si une image est manquante,
	fondé sur la liste des images qui doit être une liste suivie d'entier
	:param current_image:
	:return:
	"""
	if len(images_name_list) != 0 and current_image - images_name_list[-1] != 1:
		print(f"Il manque probablement une image.")
		print(f"Image courante: {current_image}. \n"
			  f"Image précédente: {images_name_list[-1]}.\n"
			  f"On passe à la minute suivante.")

def main(images_dir:str,
		 target:str=None,
		 debug:bool=False,
		 use_party:bool=True,
		 resegment:bool=False,
		 retranscribe:bool=False,
		 start_after:int=0,
		 device:str="cpu",
		 workers:int=1):
	images = glob.glob(f"{images_dir}/*.jpg")
	if target:
		images = [item for item in images if item == target]
	else:
		target = None

	try:
		images.sort(key=lambda x: int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	except:
		images.sort(key= lambda x: int(x.split("/")[-1].split(".jpg")[0]))
	images_number = len(images)
	print(images_dir)
	images_basedir = images_dir.replace("/", "_")
	minutes_dir = f"results/{images_basedir}_minutes.json"
	if os.path.isfile(minutes_dir):
		minutes = utils.load_json_to_dict(minutes_dir)
	else:
		pages_classees = classification_images(images=images,
							  page_classifier_model="src/Page_Classifier/models/PageClassifier_RF.joblib",
							  page_classifier_vocab="src/Page_Classifier/models/vocab_RF.joblib")
		minutes = regroupement_minutes(pages_classees=pages_classees)
		utils.serialize_dict(minutes, minutes_dir)
	minutes_number = len(minutes)
	print("Starting.")
	minute_annotee = {}
	minute_reconciliee = {}
	if workers != 1:
		torch.set_num_threads(1)
		with mp.Pool(processes=workers) as pool:
			data = [({k:v}, images_dir, device) for k, v in minutes.items()]
			for annotations, reconciliation in tqdm.tqdm(pool.starmap(single_minute_workflow, data)):
				minute_annotee = {**minute_annotee, **annotations}
				# minute_reconciliee = {**minute_reconciliee, **reconciliation}
	else:
		for idx, minute in minutes.items():
			annotations, reconciliation = single_minute_workflow({idx:minute}, images_dir=images_dir, device=device)
			minute_annotee = {**minute_annotee, **annotations}
			minute_reconciliee = {**minute_reconciliee, **reconciliation}
	utils.serialize_dict(minute_annotee, "test.json")
	# utils.serialize_dict(minute_annotee, "test_reconcilie.json")
	return images_number, minutes_number

def single_minute_workflow(minute:dict, images_dir:str, device:str):
	yolo_models = {
		"page_1": "src/Vision/models/yolov12_page_1.pt",
		"magistrats": "src/Vision/models/yolov11_table_magistrats.pt",
		"page_2": "src/Vision/models/yolov11_page_2.pt",
		"page_3": "src/Vision/models/yolo26x_page_3.pt",
		"page_4": "src/Vision/models/yolo26_page_4.pt",
		"ajouts": "src/Vision/models/yolo26_ajouts.pt"
	}
	use_party = False
	if use_party:
		import src.Vision.PARTY as PARTY
	resegment = False
	retranscribe = True
	debug = False
	print("Initiating.")
	pipeline = Pipeline(page_classifier_model="src/Page_Classifier/models/PageClassifier_RF.joblib",
						page_classifier_vocab="src/Page_Classifier/models/vocab_RF.joblib",
						yolo_models=yolo_models,
						debug=debug,
						use_party=False,
						resegment=resegment,
						retranscribe=retranscribe,
						device=device,
						images_dir=images_dir,
						current_minute = minute)
	pipeline.workflow(minute)
	return pipeline.minutes, pipeline.minutes_reconciliees

if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("-i", "--images", help="Input folder")
	arguments.add_argument("-db", "--debug", help="Debug mode", default=False)
	arguments.add_argument("-d", "--device", help="Device", default="cpu")
	arguments.add_argument("-w", "--workers", help="Workers", default=1)
	arguments.add_argument("-t", "--target", help="Target one specific file", default=None)
	arguments.add_argument("-sa", "--start_after", help="Start after given image index", default=0)
	arguments.add_argument("-rs", "--resegment", help="Launch new segmentation", default=False)
	arguments.add_argument("-rt", "--retranscribe", help="Launch new transcription", default=False)
	arguments.add_argument("-up", "--use_party", help="Use party to confirm key OCR predictions", default=True)
	arguments = arguments.parse_args()
	images_dir = arguments.images
	target = arguments.target
	workers = int(arguments.workers)
	device = arguments.device
	resegment = arguments.resegment
	retranscribe = True if arguments.retranscribe == "True" else False
	use_party = True if arguments.use_party == "True" else False
	start_after = int(arguments.start_after)
	debug = True if arguments.debug == "True" else False

	start_time = time.time()
	nombre_images, nombre_minutes = main(images_dir=images_dir,
		 target=target,
		 debug=debug,
		 use_party=use_party,
		 resegment=resegment,
		 retranscribe=retranscribe,
		 start_after=start_after,
		 device=device,
		 workers=workers)
	end_time = time.time()
	elapsed_time = end_time - start_time
	ratio_images = nombre_images / elapsed_time
	ratio_minutes = nombre_minutes / elapsed_time
	print(f"Fait en: {elapsed_time} secondes: {ratio_images} image par seconde et {ratio_minutes} minute par seconde.")


