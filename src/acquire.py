import argparse
import copy
import os
import re
import shutil
import PIL.Image
import tqdm
import torch
import multiprocessing as mp
import glob
import PIL.Image as Image
import json
import time
import pandas as pd

import utils.utils as utils
import Page_Classifier.classify as PC
import Vision.KRAKEN as KRAKEN
import Information_Extractor.extract as extract
import Information_Extractor.reconciliation as reconciliation
import Vision.YOLO as YOLO
from src.utils.utils import OCRRecord
import src.Vision.LinesDeletion as Deletions

import logging

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
)



class ListHandler(logging.Handler):

    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        self.logs.append(self.format(record))

class Pipeline:
	"""
	Classe principale de production des données. Réalise la classification et le tri des images,
	la reconstitution des minutes, l'acquisition du texte, l'extraction des informations.
	"""
	def __init__(self,
				 yolo_models,
				 debug:bool = False,
				 use_party=True,
				 resegment=False,
				 retranscribe=False,
				 device="cpu",
				 images_dir=None,
				 current_minute = None,
				 databases_dict={}):
		self.minutes_reconciliees = None
		self.minutes_reconciliees_file = None
		self.current_page_type = None
		self.debug = debug
		self.logger = logging.getLogger(__name__)
		self.current_image = None
		self.current_image_path = None
		self.current_page_transcription = None
		self.minutes = current_minute
		self.device = device
		self.images_name_list = []
		self.current_image_idx = 0
		self.pages_classees = []
		self.images_basedir = images_dir.replace("/", "_")
		self.LinesDeletionsIdentifier = Deletions.LinesDeletionsIdentifier(model_lines="src/Vision/models/line_deletion.pth",
																		   model_chars="src/Vision/models/chars_deletion_v5.pth")
		self.databases_dict = databases_dict

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
		self.kraken_ocr_model = "src/Vision/models/modele_33500l.mlmodel"
		self.kraken_ocr_model = "src/Vision/models/htr_33000l.safetensors"
		self.kraken_gloses_model = "src/Vision/models/modele_33500l.mlmodel"
		self.kraken_gloses_model = self.kraken_ocr_model
		self.party_model = "src/Vision/models/model.safetensors"
		self.minutes_annotation_file = ""
		# L'outil d'extraction de l'information
		self.resize_factor = 1
		utils.log_print(f"Use party: {use_party} and debug: {debug}")
		if debug == True or use_party == False:
			utils.log_print("Setting party engine to None")
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
										   minutier=self.minutes,
										   logger=self.logger,
										   databases_dict=databases_dict)

	def reaffecter_dictionnaire(self, minute_courante):
		"""
		Cette fonction met à jour le dictionnaire dans la classe extractor.
		:return:
		"""
		self.extractor.update_dict(minute_courante)  # Met à jour B

	def load_image(self, image):
		self.current_image_path = image


	def check_image_consistency(self, current_image):
		"""
		Cette fonction vérifie s'il y a un problème au sein des fichiers et si une image est manquante,
		fondé sur la liste des images qui doit être une liste suivie d'entier
		:param current_image:
		:return:
		"""
		if len(self.images_name_list) != 0 and current_image - self.images_name_list[-1] != 1:
			utils.log_print(f"Il manque probablement une image.")
			utils.log_print(f"Image courante: {current_image}. \n"
				  f"Image précédente: {self.images_name_list[-1]}.\n"
				  f"On passe à la minute suivante.")

	def transcription_kraken(self, image:str,
							 transcription_only:bool,
							 current_page:int,
							 suffix="",
							 extract_polygons:bool=False,
							 model=None) -> OCRRecord:
		"""
		Segmentation et transcription avec Kraken
		:param model: le modèle à utiliser
		:param current_page: la classe de la page en cours, pour utiliser un modèle de
		segmentation adapté.
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
		return kraken_ocr.predict_with_kraken(im=loaded_page, segments=baseline, extract_polygons=extract_polygons,
											  image_name=image)


	def traitement_p_2(self, page):
		"""
		Extraction d'information de la deuxième page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 2: boxes générales
		# On s'occupe d'abord de la transcription des lignes


		logging.info(f"Traitement de la page 2 ({page['image_path']})")
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

		# On va compter les mots qui entendent plusieurs soldats, pour identifier les minutes avec plusieurs soldats.
		split_regexp = re.compile(r"[.;!?'\"\s\-:]")
		vocabulaire = re.split(pattern=split_regexp, string=self.current_page_transcription.join_transcription())
		vocab_count = (vocabulaire.count("accusés")
					   + vocabulaire.count("leur")
					   + vocabulaire.count("leurs")
					   + vocabulaire.count("eux"))
		if vocab_count > 4:
			print("Plusieurs soldats.")
			return None, None

		current_dict["soldat"] = self.extractor.extraire_description_soldat_NER_p2(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			loaded_image=loaded_image,
		image_path=page["image_path"])


		current_dict["defenseur"] = self.extractor.extraire_identite_defenseur(
			ocr_prediction=self.current_page_transcription,
			annotations=zones_page_2,
			image_path=page["image_path"],
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
			image_path=page["image_path"],
			loaded_image=loaded_image)
		if identite:
			try:
				current_dict["soldat"]["identite"] = {**current_dict["soldat"]["identite"], **identite}
			except TypeError:
				current_dict["soldat"] = {"identite": identite}
		zone_dict = {"zones_identifiees": zones_page_2.to_json(), "zones_manquantes": zones_manquantes}
		return zone_dict, current_dict


	def traitement_p_3(self, page):
		"""
		Extraction d'information de la deuxième page du procès. On propose une approche modulaire: une méthode par type
		d'information recherchée
		On extrait d'abord toutes les informations à l'aide d'un modèle généraliste
		Puis on extrait les magistrats
		:return: On amende le fichier JSON général.
		"""

		# On segmente la page 3: boxes générales
		# On s'occupe d'abord de la transcription des lignes


		logging.info(f"Traitement de la page 3 ({page['image_path']})")
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
			image_path=page["image_path"],
			loaded_image=loaded_image)

		zone_dict = {"zones_identifiees": zones_page_3.to_json(), "zones_manquantes": zones_manquantes}
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


		logging.info(f"Traitement de la page 4 ({page['image_path']})")
		classes_page_4 = ["recapitulatif_somme",
							"tableau_frais"]
		current_dict = {}
		loaded_image = Image.open(page["image_path"])
		width, height = loaded_image.size
		# Tests de redimensionnement des images pour accélérer l'inférence avec Party, pas convainquant:
		# 2 à 3s sur 30 pour un facteur de 3 sur CPU.
		new_size = (int(width / self.resize_factor), int(height / self.resize_factor))
		loaded_image = loaded_image.resize(new_size)

		if show_image:
			utils.log_print("Show image")
			utils.draw_lines_on_image(image=page["image_path"],
									  baselines=[line.baseline for line in self.current_page_transcription])

		# current_dict["polygon_signature_greffier"] = self.extractor.extract_signature_greffier(ocr_prediction=self.current_page_transcription,
		# 										  image=page["image_path"])

		zones_page_4, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=classes_page_4,
																		   confidence=0.5,
																		   model=self.yolo_models["page_4"],
																		   show_image=False)

		current_dict["date_proces_1"] = self.extractor.extraire_date_1_p4(ocr_prediction=self.current_page_transcription)
		current_dict["identite"] = self.extractor.extraire_noms_p4(ocr_prediction=self.current_page_transcription,
																   image_path=page['image_path'])

		if "tableau_frais" not in zones_manquantes:
			current_dict["tableau_frais"], nom_2 = self.extractor.extraire_tableau_p4(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_4,
				image_path=page["image_path"],
				loaded_image=loaded_image)
		else:
			nom_2 = {}

		try:
			current_dict["identite"] = {**current_dict["identite"], **nom_2}
		except TypeError:
			pass

		if "recapitulatif_somme" not in zones_manquantes:
			current_dict["dernier_paragraphe"], current_dict["date_proces_2"] = self.extractor.extraire_paragraphe_final_p4(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_4,
				loaded_image=loaded_image)





		zone_dict = {"zones_identifiees": zones_page_4.to_json(), "zones_manquantes": zones_manquantes}
		return zone_dict, current_dict


	def transcription_page(self,
						   page:str,
						   show_image:bool = False,
						   force_resegment:bool = False) -> None:
		"""
		Fonction wrapper de transcription d'une page. Produit également la reconnaissance des
		mots biffés dans la ligne. Met à jour l'objet self.current_page_transcription
		:param force_resegment: Faut-il resegmenter la page?
		:param page: Le chemin vers la page à transcrire
		:param show_image: Montrer l'image transcrite avec les lignes ?
		"""
		target_transcription = f"results/ocr_predictions/{page['image_path'].replace('/', '_').replace('.jpg', '.json')}"
		if not os.path.isfile(target_transcription) or self.resegment or self.retranscribe or force_resegment:
			utils.log_print(f"Segmentation/Transcription with kraken of page {page['image_path']}")
			self.current_page_transcription = self.transcription_kraken(
				image=page["image_path"],
				transcription_only=self.resegment is False
								   and self.retranscribe is True,
				current_page=int(
					page['classe'].split("_")[-1],),
				extract_polygons=True
																		)
			t1 = time.time()
			for line in self.current_page_transcription:
				line.prediction_with_deletion = None
				continue
				sentence = self.LinesDeletionsIdentifier.identify_deletions(line, image, level="char")
				line.prediction_with_deletion = sentence
			t2 = time.time()
			elapsed = t2 - t1
			# utils.log_print(f"Fait en {elapsed} secondes")
			utils.serialize_dict(self.current_page_transcription.to_json(), target_transcription)
		else:
			utils.log_print("Found existing kraken transcription: " + target_transcription)
			self.current_page_transcription = OCRRecord()
			self.current_page_transcription.from_json(path=target_transcription)
			t1 = time.time()
			image = PIL.Image.open(page["image_path"])
			if self.current_page_transcription is None:
				utils.log_print(f"Error with page {page['image_path']}")
			# try:
			# 	for line in self.current_page_transcription:
			# 		line.prediction_with_deletion = None
			# 		continue
			# 		sentence = self.LinesDeletionsIdentifier.identify_deletions(line, image, level="char")
			t2 = time.time()
			elapsed = t2 - t1
			# utils.log_print(f"Fait en {elapsed} secondes")

		if show_image:
			baselines = [line.baseline for line in self.current_page_transcription]
			utils.draw_lines_on_image(image=page["image_path"], baselines=baselines)


	def process_additions(self, page:json) -> tuple[dict, dict]:
		"""
		Cette fonction gère les ajouts postérieurs.
		:param page: the page metadata as json
		:return: le dictionnaire contenant les zones, et le dictionnaire avec les informations extraites.
		"""

		# On segmente la page 1: boxes générales
		# utils.log_print(f"Checking additions")


		zones_ajouts, zones_manquantes = self.YOLO_Segmenter.segment_zones(page["image_path"],
																		   target_classes=["MarginTextZone-ajout"],
																		   confidence=0.1,
																		   model=self.yolo_models["ajouts"],
																		   show_image=False)


		zone_dict = {"zones_identifiées": zones_ajouts.to_json(), "zones_manquantes": zones_manquantes}
		if len(zones_manquantes) == 0:
			target_transcription = f"results/ocr_predictions/{page['image_path'].replace('/', '_').replace('.jpg', '.ajouts.json')}"
			if not os.path.isfile(target_transcription) or self.retranscribe is True:
				lignes_glosees = self.transcription_kraken(
					image=page["image_path"],
					transcription_only=False,
					current_page=0,
					suffix=".ajouts",
					model=self.kraken_gloses_model
				)
				utils.serialize_dict(lignes_glosees.to_json(), target_transcription)
			else:
				utils.log_print("Found existing kraken transcription: " + target_transcription)
				lignes_glosees = OCRRecord()
				lignes_glosees.from_json(path=target_transcription)

			# baselines = [line.baseline for line in lignes_glosees]
			# utils.draw_lines_on_image(image_path=page["image_path"], baseline=baselines)
		else:
			return None, None

		informations_ajouts = self.extractor.extraire_informations_ajouts_posterieurs(ocr_prediction=lignes_glosees,
																					  annotations=zones_ajouts)

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
		logging.info(f"Traitement de la page 1 ({page['image_path']})")
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
		if len(zones_page_1.filter_zones("Nom du soldat")) > 1:
			print("Plusieurs soldats")
			return None, None

		zone_dict = {"zones_identifiées": zones_page_1.to_json(), "zones_manquantes": zones_manquantes}

		if "Magistrats" not in zones_manquantes:
			current_dict["date_proces"] = self.extractor.extraire_date_du_proces_p1(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
				loaded_image=loaded_image,
			    image_path=page["image_path"])
		else:
			current_dict["date_proces"] = None

		# On extrait le nom et prénom du soldat
		if "Description du Soldat" in zones_manquantes:
			current_dict['soldat'] = None
			return zone_dict, current_dict
		else:
			current_dict['soldat'] = self.extractor.extraire_description_soldat_NER_p1(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
			image_path=page["image_path"])


		if "Inculpation_antecedents" in zones_manquantes:
			current_dict["Inculpation"], current_dict["Antécédents"] = None, None
		else:
			accusation_antecedents =  self.extractor.extraire_inculpation_et_antecedents(
				ocr_prediction=self.current_page_transcription,
				annotations=zones_page_1,
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
				image_path=page["image_path"],
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
			# if test_lignes is False:
				# utils.log_print("Warning: une des lignes du jury n'a pas été identifiée.")
			current_dict["magistrats"] = self.extractor.extraire_magistrats(
				ocr_prediction=self.current_page_transcription,
				zones_magistrats=magistrats,
				image=page["image_path"],
				show_images=False)





		# On extrait la date du crime
		if "MainZone-crimeDate" in zones_manquantes:
			logging.error("Zone de date du crime non identifiée.")
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



	def workflow(self,
				 minute:dict,
				 target:str|None=None,
				 start_after:int=0):
		"""
		La fonction qui classe les pages, produit les minutes
		et distribue les tâches en fonction de la classe de la page
		:param minute: le minutier complet, organisé en minutes
		:param target: [DEBUG] l'image à traiter dans le corpus
		:param start_after: [DEBUG] commencer le traitement avec l'image X
		:return:
		"""
		handler = ListHandler()
		handler.setFormatter(
			logging.Formatter(
				"%(asctime)s | %(name)s | %(levelname)s | %(message)s"
			)
		)

		root_logger = logging.getLogger()
		root_logger.addHandler(handler)
		utils.log_print("Début du workflow")
		try:
			# Il faudra supprimer ça pour la mise en production
			minute_number = list(minute.keys())[0]
			self.minutes_annotation_file = f"results/{self.images_basedir}_minutes_annotations_{minute_number}.json"
			self.minutes_reconciliees_file = f"results/{self.images_basedir}_minutes_annotations_{minute_number}_reconcilie.json"
			self.minutes_reconciliees = None
			image_index = 0
			previous_pages = None
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
						utils.log_print("---", print_message=True)
						utils.log_print(f"Treating {page}", print_message=True)
						if page["classe"] == "page_4":
							force_resegment = True
							extract_polygons = True
						else:
							force_resegment = False
							extract_polygons = False
						force_resegment = False
						self.transcription_page(page=page,
												show_image=False,
												force_resegment=force_resegment)
						zones_ajouts, ajouts = self.process_additions(page=page)
						if ajouts is None:
							ajouts = {"ajouts": None}
					if page["classe"] == "page_1":
						zones, annotations = self.traitement_p_1(page=page, show_image=False)
					elif page['classe'] == "page_2":
						zones, annotations = self.traitement_p_2(page=page)
					elif page['classe'] == "page_3":
						zones, annotations = self.traitement_p_3(page=page)
					elif page['classe'] == "page_4":
						zones, annotations = self.traitement_p_4(page=page, show_image=False)
					else:
						continue
					if (zones, annotations) == (None, None):
						page["extractions"] = {"commentaire": "Plusieurs soldats"}
						break
					page["extractions"] = {**annotations, **ajouts}
					page["zones"] = zones
					self.reaffecter_dictionnaire(pages)
				if target is None:
					reconciliator = reconciliation.Reconciliator(minute_list=pages,
																 previous_minute=previous_pages,
																 databases=self.databases_dict)
					previous_pages = copy.copy(pages)
					reconciliator.reconciliate_minute()
					self.minutes_reconciliees = reconciliator.reconciliated_minute
					# self.minutes_reconciliees = {}
			return handler.logs
		finally:
			root_logger.removeHandler(handler)


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
	utils.log_print("Reconstitution des minutes")
	current_minute = []
	current_minute_number = 0
	minutes = {}
	# Puis on rassemble les minutes
	for idx, ((dossier, ident, image), classe) in enumerate(pages_classees):
		current_image = {"répertoire": dossier, "id": ident, "image_path": image, "classe": classe}
		current_minute.append(current_image)
		if ident == pages_classees[-1][0][1]:
			minutes[current_minute_number] = current_minute
			break
		if classe in ["page_4", "page_autre"] and pages_classees[idx + 1][1] == "page_1":
			minutes[current_minute_number] = current_minute
			current_minute = []
			current_minute_number += 1
	return minutes
	# utils.save_as_dict(minutes, out_dir)

def classification_images(images, page_classifier_model, page_classifier_vocab, workers):
	"""
	Cette fonction classe toutes les images à l'aide d'un Random Forest
	:param page_classifier_vocab: le chemin vers le vocabulaire (mapping classes/labels)
	:param page_classifier_model: le chemin vers le modèle de classification
	:param images: la liste d'images
	:return:
	"""
	# On commence par classer toutes les images du dossier
	utils.log_print("Classification des images")
	images_name_list = []
	pages_classees =  []
	page_classifier = PC.PageClassifier(build_vocab=False,
											 model=page_classifier_model,
											 vocab=page_classifier_vocab)
	data = []
	for image in tqdm.tqdm(images):
		dossier, ident = utils.get_name_from_path(image)
		# On vérifie s'il n'y a pas de problème de disparition d'image
		check_image_consistency(ident, images_name_list)
		data.append((dossier, ident, image, page_classifier))
		# images_name_list.append(ident)
		# current_page_type = page_classifier.predict(image=image)
		# pages_classees.append(((dossier, ident, image), current_page_type))
		# if image == images[-1]:
		# 	utils.log_print("Dossier terminé")
	pages_classees = []
	with mp.Pool(processes=workers) as pool:
		for (dossier, ident, image), current_page_type in pool.starmap(predict, data):
			pages_classees.append(((dossier, ident, image), current_page_type))
	pages_classees.sort(key=lambda x:int(x[0][1]))
	utils.log_print("Pages classées.", print_message=True)
	return pages_classees

def predict(dossier, ident, image, page_classifier):
	return (dossier, ident, image), page_classifier.predict(image=image)

def check_minute_consistency(minute_list):
	try:
		classes = [int(item["classe"].split("_")[-1]) for item in minute_list if item["classe"] not in ["page_autre", "page_manuscrite_suivie"]]
	except ValueError:
		print([item["classe"] for item in minute_list])
		return False, {}
	try:
		if minute_list[-2]["classe"] in ["page_autre", "page_manuscrite_suivie"] and classes == [1, 2, 4]:
			minute_list[-2]["classe"] = "page_3"
			return True, minute_list
	except IndexError:
		return False, {}
	if classes == [1, 2, 3, 4]:
		print("Minute correctement ordonnée")
		return True, minute_list
	else:
		print("Quelque chose ne va pas avec la minute")
		[shutil.copy(image["image_path"], f"debug/") for image in minute_list]
		return False, {}


def check_image_consistency(current_image, images_name_list):
	"""
	Cette fonction vérifie s'il y a un problème au sein des fichiers et si une image est manquante,
	fondé sur la liste des images qui doit être une liste suivie d'entier
	:param images_name_list: l'ensemble des images
	:param current_image: l'image en cours
	:return:
	"""
	if len(images_name_list) != 0 and current_image - images_name_list[-1] != 1:
		utils.log_print(f"Il manque probablement une image.")
		utils.log_print(f"Image courante: {current_image}. \n"
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
		 workers:int=1,
		 focus=None):
	images = glob.glob(f"{images_dir}/*.jpg")
	if target:
		images = [item for item in images if item == target]
	if start_after:
		images = [item for item in images  if int(item.split("/")[-1].replace(".jpg", "")) > start_after]
		images.sort(key=lambda x: int(x.split("/")[-1].split(".jpg")[0]))
	else:
		target = None
	try:
		images.sort(key=lambda x: int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	except:
		images.sort(key= lambda x: int(x.split("/")[-1].split(".jpg")[0]))
	images_number = len(images)
	images_basedir = images_dir.replace("/", "_")
	minutes_dir = f"results/{images_basedir}_minutes.json"
	if os.path.isfile(minutes_dir) and not target:
		minutes = utils.load_json_to_dict(minutes_dir)
	else:
		pages_classees = classification_images(images=images,
							  page_classifier_model="src/Page_Classifier/models/PageClassifier_RF_2.joblib",
							  page_classifier_vocab="src/Page_Classifier/models/vocab_RF_2.joblib",
											   workers=workers)
		minutes = regroupement_minutes(pages_classees=pages_classees)
		utils.serialize_dict(minutes, minutes_dir)
	for ident, minute in minutes.items():
		conformant, updated_minute = check_minute_consistency(minute)
		if conformant is True:
			minutes[ident] = updated_minute
		else:
			minutes[ident] = {}
	minutes_number = len(minutes)
	utils.log_print("Starting.")
	minute_annotee = {}
	minute_reconciliee = {}
	# minute_annotee = utils.load_json_to_dict("results/results.json")
	# minute_reconciliee = utils.load_json_to_dict("results/results_reconciliated.json")
	# utils.convert_to_csv(minute_reconciliee, "results/results.csv")
	# exit(0)
	# exit(0)
	# # exit(0)
	# previous_pages = None
	# import src.Information_Extractor.reconciliation as reconciliation
	# for idx, minute in minute_annotee.items():
	# 	reconciliator = reconciliation.Reconciliator(minute_list=minute, previous_minute=previous_pages)
	# 	reconciliator.reconciliate_minute()
	# 	reconc = reconciliator.reconciliated_minute
	# 	minute_reconciliee[idx] = reconc
	# #
	# utils.serialize_dict(minute_reconciliee, "results/results_reconciliated.json")
	# utils.convert_to_csv(minute_reconciliee, "results/results.csv")
	# exit(0)

	# Trouvé dans https://www.insee.fr/fr/statistiques/3536630 (fichier de l'INSEE)
	list_of_surnames = [name.lower() for name in
							 pd.read_csv("src/Information_Extractor/databases/french_surnames.csv",
										 delimiter="\t")["NOM"].tolist()]
	# Idem: https://www.insee.fr/fr/statistiques/8595130
	list_of_names = {name: gender for gender, name in
						  pd.read_csv("src/Information_Extractor/databases/french_names.csv",
									  delimiter=",").values.tolist()}
	french_lexicon = set(
		[utils.remove_accents(word).lower() for word in utils.txt_to_list("src/resources/french_lexicon.txt") if
		 not word.isupper()])

	with open("src/resources/liste_presidents.txt", "r") as input_presidents:
		liste_presidents = [item.replace("\n", "") for item in input_presidents.readlines()]
	with open("src/resources/liste_jures.txt", "r") as input_presidents:
		liste_jures = [item.replace("\n", "") for item in input_presidents.readlines()]

	with open("src/resources/rangs_militaires.txt", "r") as rangs:
		rangs_militaires = [item.replace("\n", "") for item in rangs.readlines()]

	with open("src/Information_Extractor/models/charge_identification/labels_dict.json", "r") as output_json:
		charge_identification_labels = json.load(output_json)

	df = pd.read_csv("src/resources/professions_categories.csv", delimiter="\t")
	df = df.dropna()
	professions_et_categories_sociopro = df["Profession"].tolist()
	dictionnaire_professions_categories = dict(sorted(df.values.tolist()))

	databases_dict = {"list_of_surnames": list_of_surnames,
								"list_of_names": list_of_names,
								"french_lexicon": french_lexicon,
								"charge_identification_labels": charge_identification_labels,
								"rangs_militaires": rangs_militaires,
								"liste_jures": liste_jures,
								"liste_presidents": liste_presidents,
					  "professions_et_categories_sociopro": professions_et_categories_sociopro,
					  "dictionnaire_professions_categories": dictionnaire_professions_categories}

	minute_reconciliee = {}
	minute_log = {}
	if focus:
		minutes = {k:v for k, v in minutes.items() if k==focus}
	if workers != 1:
		torch.set_num_threads(1)
		torch.set_num_interop_threads(1)
		with mp.Pool(processes=workers) as pool:
			data = [({k:v}, images_dir, device, retranscribe, databases_dict) for k, v in minutes.items()]
			for minute_n, annotations, reconciliation, log in tqdm.tqdm(pool.starmap(single_minute_workflow, data)):
				minute_annotee = {**minute_annotee, **annotations}
				minute_reconciliee[minute_n] = reconciliation
				minute_log[minute_n] = log
	else:
		for idx, minute in minutes.items():
			minute_n, annotations, reconciliation, log = single_minute_workflow({idx:minute},
																				images_dir=images_dir,
																				device=device,
																				retranscribe=retranscribe,
																				databases_dict=databases_dict)
			minute_annotee = {**minute_annotee, **annotations}
			minute_reconciliee[minute_n] = reconciliation
			minute_log[minute_n] = log
	utils.serialize_dict(minute_log, f"results/log_{images_basedir}.json")
	utils.serialize_dict(minute_annotee, f"results/results_{images_basedir}.json")
	utils.serialize_dict(minute_reconciliee, f"results/results_reconciliated_{images_basedir}.json")
	utils.convert_to_csv(minute_reconciliee, f"results/results_{images_basedir}.csv")
	return images_number, minutes_number

def single_minute_workflow(minute:dict,
						   images_dir:str,
						   device:str,
						   retranscribe:bool,
						   databases_dict={}):
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
		pass
	resegment = False
	debug = False
	utils.log_print("Initiating.")
	pipeline = Pipeline(yolo_models=yolo_models,
						debug=debug,
						use_party=False,
						resegment=resegment,
						retranscribe=retranscribe,
						device=device,
						images_dir=images_dir,
						current_minute = minute,
						databases_dict=databases_dict)
	logger = pipeline.workflow(minute)
	minute_number = list(minute.keys())[0]
	return minute_number, pipeline.minutes, pipeline.minutes_reconciliees, logger

if __name__ == '__main__':
	arguments = argparse.ArgumentParser()
	arguments.add_argument("-i", "--images", help="Input folder")
	arguments.add_argument("-db", "--debug", help="Debug mode", default=False)
	arguments.add_argument("-d", "--device", help="Device", default="cpu")
	arguments.add_argument("-w", "--workers", help="Workers", default=1)
	arguments.add_argument("-f", "--focus", help="Focus on specific minute", default=None)
	arguments.add_argument("-t", "--target", help="Target one specific file", default=None)
	arguments.add_argument("-sa", "--start_after", help="Start after given image index", default=0)
	arguments.add_argument("-rs", "--resegment", help="Launch new segmentation", default=False)
	arguments.add_argument("-rt", "--retranscribe", help="Launch new transcription", default=False)
	arguments.add_argument("-up", "--use_party", help="Use party to confirm key OCR predictions", default=True)
	arguments = arguments.parse_args()
	images_dir = arguments.images
	target = arguments.target
	focus = int(arguments.focus) if arguments.focus is not None else None
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
		 workers=workers,
										 focus=focus)
	end_time = time.time()
	elapsed_time = end_time - start_time
	ratio_images = nombre_images / elapsed_time
	ratio_minutes = elapsed_time / nombre_minutes
	utils.log_print(f"Fait en: {elapsed_time} secondes: {ratio_images} image par seconde et {ratio_minutes} secondes pour une minute.")


