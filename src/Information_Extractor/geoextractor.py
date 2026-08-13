import copy
import json
import re
import unicodedata

import src.utils.utils as utils
import src.Information_Extractor.similarity as similarity
import logging

logger = logging.getLogger(__name__)


class GeoExtractor():
	def __init__(self):
		with open("src/Information_Extractor/databases/referentiel_communes.json", "r") as input_json:
			self.geodict = json.load(input_json)
		with open("src/Information_Extractor/databases/correspondance_departements.json", "r") as input_json:
			self.departments_dict = json.load(input_json)
		with open("src/resources/liste_pays_2026.txt", "r") as input_file:
			self.coutries_list = [item.replace("\n", "") for item in input_file.readlines()]
		with open("src/Information_Extractor/databases/arrondissements_paris.json", "r") as input_json:
			self.arrondissement_dict = json.load(input_json)
		self.filtered_geodict = {}

	def correct_department(self, departement: str) -> tuple[str|None, list|None, bool]:
		"""
		Fonction qui permet de corriger le département à partir d'une base de connaissances
		:param departement:
		:return:
		"""
		departement = utils.clean_small_string(departement)
		if departement is None or len(departement) < 2:
			return None, None, False
		if departement in self.coutries_list:
			return departement, departement, True

		if departement in self.departments_dict:
			return self.departments_dict[departement], departement, False
		else:
			liste_des_departements = list(self.departments_dict.keys())
			matching_department, distance = similarity.find_closest_word_in_list(liste_des_departements,
																				 departement,
																				 replacement_mapping={"-": " "})
			matching_country, distance_country = similarity.find_closest_word_in_list(self.coutries_list,
																					  departement,
																					  replacement_mapping={"()": ""})
			logger.info(f"Chaîne de caractères: {departement}")
			logger.info(f"Département identifié: {matching_department}, distance: {distance}")
			logger.info(f"Pays le plus similaire: {matching_country}, distance: {distance_country}")
			if distance_country < distance:
				logger.info("Pays etranger identifié.")
				return departement, matching_country, True
			actual_departement = self.departments_dict[matching_department]
			# Si la distance est trop grande, il s'agit probablement d'une erreur de transcription. On ne filtre pas
			# Problème avec une distance absolue: pénalise les chaînes de caractères longues.
			if distance > 5 or distance == len(departement):
				logger.info(f"Département actuel: {departement}")
				return None, departement, False

			logger.info(f"Département actuel: {actual_departement}")
			return actual_departement, matching_department, False

	def filter_geodict_by_department(self, clean_departement: list):
		"""
		Cette fonction filtre le dictionnaire contenant les positions géographiques des communes françaises
		par département.
		:param departement: le département tel qu'il apparaît dans le minutier
		:return:
		"""
		self.filtered_geodict = copy.deepcopy(self.geodict)
		if clean_departement is None:
			return
		if isinstance(clean_departement, str):
			logger.warning(f"{clean_departement} devrait être une liste.")
			clean_departement = [clean_departement]
		clean_departement = [unicodedata.normalize('NFC', dpt).replace("’", "'") for dpt in clean_departement]
		logger.info(f"On filtre la base de données géographique en ne retenant que {clean_departement}.")
		for key, value in self.geodict.items():
			# Si la clé actuelle ne correspond pas aux départements correspondants, on supprime du dictionnaire.
			if value["département"]:
				normalized = unicodedata.normalize('NFC', value["département"])
				if normalized not in clean_departement:
					del self.filtered_geodict[key]
			else:
				logger.info(f'On conserve {value["département"]}')

	def paris(self, arrondissement):
		arrondissement_regexp = re.compile(r"\d{1,2}")
		try:
			search = re.search(arrondissement_regexp, arrondissement)
		except TypeError:
			return {
				"nom_actuel": "Paris",
				"nom_1999": "Paris",
				"nom_1801": "Paris",
				"departement": "Seine",
				"arrondissement": None,
				"lat": "48.829839",
				"lon": "2.44162",
				"etranger": False,
				"hors_metropole": False,
				"pays": "France"
			}
		try:
			arrondissement_extrait = search.group()
		# Si on ne trouve pas l'arrondissement, on va mettre le point au milieu du Bois de Vincennes: suffisamment
		# proche de Paris pour être interprétable visuellement, suffisamment éloigné pour ne pas être considéré
		# comme un point sur un des arrondissement, mais un point dans Paris sans + de précision
		except AttributeError:
			return {
				"nom_actuel": "Paris",
				"nom_1999": "Paris",
				"nom_1801": "Paris",
				"departement": "Seine",
				"arrondissement": None,
				"lat": "48.829839",
				"lon": "2.44162",
				"etranger": False,
				"hors_metropole": False,
				"pays": "France"
			}
		try:
			corresponding_entry = next(item for item in self.arrondissement_dict if
									   int(item["numero_arrondissement"]) == int(arrondissement_extrait))
		except StopIteration:
			return {
				"nom_actuel": "Paris",
				"nom_1999": "Paris",
				"nom_1801": "Paris",
				"departement": "Seine",
				"arrondissement": None,
				"lat": "48.829839",
				"lon": "2.44162",
				"etranger": False,
				"hors_metropole": False,
				"pays": "France"
			}
		coordinates = corresponding_entry['geo_point_2d']
		return {
			"nom_actuel": "Paris",
			"nom_1999": "Paris",
			"nom_1801": "Paris",
			"departement": "Seine",
			"arrondissement": arrondissement_extrait,
			"lon": coordinates["lon"],
			"lat": coordinates["lat"],
			"etranger": False,
			"hors_metropole": False,
			"pays": "France"
		}

	def database_retrieval(self, ville, arrondissement, departement):
		"""
		Cette fonction récupère les coordonnées géographique d'une ville, en se servant de l'information de département
		pour filtrer et éventuellement de l'arrondissement (pour Paris)
		:param ville: la ville à localiser
		:param arrondissement: l'arrondissement (utilisé pour Paris)
		:param departement: le département servant à filtrer la recherche
		:return: un dictionnaire de la forme:
							{"lon": 4.55039,
							"lat": -45,
							"nom_actuel": nom_actuel_de_la_ville
							}
		"""
		match = False
		if departement is not None:
			departement = departement.replace("l'", "")
			actual_departement, departement_corrige, is_country = self.correct_department(departement)
		else:
			logger.error("Le département n'a pas été identifié et l'extraction des coordonnées géographiques "
						 "n'est pas possible.")
			return {
				"lat": None,
				"lon": None,
				"nom_actuel": None,
				"departement": None,
				"etranger": None,
				"hors_metropole": None,
				"pays": None,
			}
		if is_country is True:
			return {
				"lat": None,
				"lon": None,
				"nom_actuel": None,
				"departement": None,
				"pays": departement_corrige,
				"hors_metropole": False,
				"etranger": True,
			}
		# TODO: Ajouter le Sénégal, le Maroc, Madagascar, la Cochinchine, le Tonkin et utiliser une base de donnée pour tous les pays.
		if ville in ["Constantine", "Oran", "Alger"] or departement_corrige in ["Constantine", "Oran", "Alger", "Maroc",
																				"Madagascar", "Tonkin", "Cochinchine",
																				"Cochinchine française",
																				"Sénégal-Niger", "Soudan français"]:
			if ville == "Constantine" or departement_corrige == "Constantine":
				latitude = 36.35
				longitude = 6.60
			elif ville == "Alger" or departement_corrige == "Alger":
				latitude = 36.75
				longitude = 3.04
			elif ville == "Oran" or departement_corrige == "Oran":
				latitude = 35.70
				longitude = 0.63
			elif departement_corrige == "Tonkin":
				latitude = 21.03
				longitude = 105.83
			elif departement_corrige in ["Sénégal", "Sénégal-Niger"]:
				latitude = 14.50
				longitude = -14.45
			elif departement_corrige == "Maroc":
				latitude = 33.97
				longitude = -6.85
			elif departement_corrige == "Soudan français":
				latitude = 12.65
				longitude = -8.00
			elif departement_corrige in ["Cochinchine", "Cochinchine française"]:
				latitude = 10.79
				longitude = 106.67
			elif departement_corrige == "Madagascar":
				latitude = -18.93
				longitude = 47.51
			return {
				"lat": latitude,
				"lon": longitude,
				"nom_actuel": ville,
				"departement": departement_corrige,
				"pays": "France",
				"etranger": False,
				"hors_metropole": True
			}
		if ville == "Paris":
			return self.paris(arrondissement)
		if ville:
			ville = utils.expand_placename_abreviations(ville)
		# On va réduire la taille du dictionnaire pour rendre la recherche + efficace et plus précise
		# On va nettoyer le département
		self.filter_geodict_by_department(actual_departement)
		if self.filtered_geodict == {}:
			logger.error(f"Le dictionnaire géographique filtré est vide. Département actuel: {actual_departement}")
		# On regarde la liste des villes
		for key, row in self.filtered_geodict.items():
			if ville == row["nom_1801"]:
				latitude = row["latitude"]
				longitude = row["longitude"]
				current_name = row["nom_actuel"]
				commune_correspondante = row
				match = True
			elif ville == row["nom_1999"]:
				latitude = row["latitude"]
				longitude = row["longitude"]
				current_name = row["nom_actuel"]
				commune_correspondante = row
				match = True
			elif ville == row["nom_actuel"]:
				latitude = row["latitude"]
				longitude = row["longitude"]
				current_name = row["nom_actuel"]
				commune_correspondante = row
				match = True
		if match == False:
			liste_des_communes_1801 = [item["nom_1801"] for key, item in self.filtered_geodict.items()]
			liste_des_communes_1999 = [item["nom_1999"] for key, item in self.filtered_geodict.items()]
			liste_des_communes_actuelles = [item["nom_actuel"] for key, item in self.filtered_geodict.items()]
			if ville:
				closest_1999, distance_1999 = similarity.find_closest_word_in_list(liste_des_communes_1999, ville)
			else:
				logger.error("Coordonnées non trouvées.")
				return {
					"lat": None,
					"lon": None,
					"nom_actuel": ville,
					"departement": departement_corrige,
					"pays": "France",
					"etranger": False,
					"hors_metropole": False
				}
			closest_1801, distance_1801 = similarity.find_closest_word_in_list(liste_des_communes_1801, ville)
			closest_actuel, distance_actuel = similarity.find_closest_word_in_list(liste_des_communes_actuelles, ville)
			# try:
			distances = [distance_actuel, distance_1999, distance_1801]
			try:
				min_distance = min(distances)
			except TypeError as e:
				logger.error(f"Type error: {e}")
				logger.error(f"Ville: {ville}")
				logger.error(f"Distance: {distance_1999}")
				logger.error(f"Distances: {distances}")
				logger.error(f"Closest 1999: {closest_1999}")
				logger.error(f"Closest 1801: {closest_1801}")
				logger.error(f"Closest actuel: {closest_actuel}")
				return {
					"lat": None,
					"lon": None,
					"nom_actuel": ville,
					"departement": departement_corrige,
					"pays": "France",
					"etranger": False,
					"hors_metropole": False
				}
			if distances.index(min_distance) == 0:
				current_feature = "nom_actuel"
			elif distances.index(min_distance) == 1:
				current_feature = "nom_1999"
			else:
				current_feature = "nom_1801"
			closest_match = [closest_actuel, closest_1999, closest_1801][distances.index(min_distance)]
			commune_correspondante = next(
				(item for item in self.filtered_geodict.values() if item[current_feature] == closest_match))

			current_name = closest_match
			longitude = commune_correspondante["longitude"]
			latitude = commune_correspondante["latitude"]
		return {
			"lat": latitude,
			"lon": longitude,
			"nom_actuel": current_name,
			"nom_1999": commune_correspondante["nom_1999"],
			"nom_1801": commune_correspondante["nom_1801"],
			"departement": departement_corrige,
			"pays": "France",
			"etranger": False,
			"hors_metropole": False
		}
