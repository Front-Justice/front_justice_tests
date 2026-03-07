import copy
import json
import re

import src.utils.utils as utils

class GeoExtractor():
	def __init__(self):
		with open("src/Information_Extractor/databases/referentiel_communes.json", "r") as input_json:
			self.geodict = json.load(input_json)
		with open("src/Information_Extractor/databases/correspondance_departements.json", "r") as input_json:
			self.departments_dict = json.load(input_json)
		with open("src/Information_Extractor/databases/arrondissements_paris.json", "r") as input_json:
			self.arrondissement_dict = json.load(input_json)
		self.filtered_dict = {}

	def correct_department(self, departement):
		"""
		Cette fonction corrige un département en utilisant une liste pré-établie
		:param departement: la chaîne de caractère contenant le nom du département
		:return: le département correct
		"""
		if departement is None:
			return
		if departement in self.departments_dict:
			matching_department = departement
		else:
			print(departement)
			liste_des_departements = list(self.departments_dict.keys())
			matching_department, distance = utils.find_closest_word_in_list(liste_des_departements, departement)
			corresponding_departments = self.departments_dict[matching_department]
			# Si la distance est trop grande, il s'agit probablement d'une erreur de transcription. On ne filtre pas
			if distance > 4:
				return departement
		return matching_department

	def filter_geodict_by_arrondissement(self, arrondissement):
		"""
		Cette fonction permet de filtrer le dictionnaire par arrondissement.
		Nous n'avons pas de base de données des arrondissements pour 1914-1918, nous allons donc tricher en utilisant
		le département de l'arrondissement concerné.
		:param arrondissement: l'arrondissement identifié
		:return:
		"""
		self.filtered_dict = copy.deepcopy(self.geodict)
		if arrondissement is None:
			return
		match = False
		for key, row in self.filtered_dict.items():
			if arrondissement == row["nom_1801"]:
				departement_correspondant = row["nom_actuel"]
				match = True
			elif arrondissement == row["nom_1999"]:
				departement_correspondant = row["nom_actuel"]
				match = True
		if match == False:
			liste_des_communes_1801 = [item["nom_1801"] for key, item in self.geodict.items()]
			liste_des_communes_1999 = [item["nom_1999"] for key, item in self.geodict.items()]
			closest_1999, distance_1999 = utils.find_closest_word_in_list(liste_des_communes_1999, arrondissement)
			closest_1801, distance_1801 = utils.find_closest_word_in_list(liste_des_communes_1801, arrondissement)
			if distance_1801 < distance_1999:
				closest_match = closest_1801
				commune_correspondante = next((item for item in self.geodict.values() if item["nom_1801"] == closest_match))
			else:
				closest_match = closest_1999
				commune_correspondante = next((item for item in self.geodict.values() if item["nom_1999"] == closest_match))
			departement_correspondant = commune_correspondante["département"]
		# On a identifié le département, on va maintenant filtrer.
		print(f"Arrondissement prédit: {arrondissement}")
		print(f"Commune correspondante dans la base: {commune_correspondante}")
		print(f"Département: {departement_correspondant}")
		self.filter_geodict_by_department(departement_correspondant)

	def filter_geodict_by_department(self, departement):
		"""
		Cette fonction filtre le dictionnaire contenant les positions géographiques des communes françaises
		par département.
		:param departement: le département tel qu'il apparaît dans le minutier
		:return:
		"""
		self.filtered_dict = copy.deepcopy(self.geodict)
		if departement is None:
			return
		if departement in self.departments_dict:
			corresponding_departments = self.departments_dict[departement]
		else:
			print(departement)
			liste_des_departements = list(self.departments_dict.keys())
			matching_department, distance = utils.find_closest_word_in_list(liste_des_departements, departement)
			corresponding_departments = self.departments_dict[matching_department]
			# Si la distance est trop grande, il s'agit probablement d'une erreur de transcription. On ne filtre pas
			if distance > 4:
				return

		for key, value in self.geodict.items():
			# Si la clé actuelle ne correspond pas aux départements correspondants, on supprime du dictionnaire.
			if value["département"] and value["département"] not in corresponding_departments:
				del self.filtered_dict[key]

	def paris(self, arrondissement):
		arrondissement_regexp = re.compile(r"\d{1,2}")
		try:
			search = re.search(arrondissement_regexp, arrondissement)
		except TypeError:
			return {
				"arrondissement": None,
				"lat": "48.829839",
				"lon": "2.44162"
			}
		try:
			arrondissement_extrait = search.group()
		# Si on ne trouve pas l'arrondissement, on va mettre le point au milieu du Bois de Vincennes: suffisamment
		# proche de Paris pour être interprétable visuellement, suffisamment éloigné pour ne pas être considéré
		# comme un point sur un des arrondissement, mais un point dans Paris sans + de précision
		except AttributeError:
			return {
				"arrondissement": None,
				"lat": "48.829839",
				"lon": "2.44162"
			}

		corresponding_entry = next(item for item in self.arrondissement_dict if int(item["numero_arrondissement"]) == int(arrondissement_extrait))
		coordinates = corresponding_entry['geo_point_2d']
		return {
			"arrondissement": arrondissement_extrait,
			"lon": coordinates["lon"],
			"lat": coordinates["lat"]
		}


	def retrieve_coordinates(self, ville, arrondissement, departement):
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
		if ville == "Paris":
			return self.paris(arrondissement)
		# On va réduire la taille du dictionnaire pour rendre la recherche + efficace et plus précise
		if departement is None:
			return None
		else:
			self.filter_geodict_by_department(departement)

			# On regarde la liste des villes
			for key, row in self.filtered_dict.items():
				if ville == row["nom_1801"]:
					latitude = row["latitude"]
					longitude = row["longitude"]
					current_name = row["nom_actuel"]
					match = True
				elif ville == row["nom_1999"]:
					latitude = row["latitude"]
					longitude = row["longitude"]
					current_name = row["nom_actuel"]
					match = True
			if match == False:
				liste_des_communes_1801 = [item["nom_1801"] for key, item in self.geodict.items()]
				liste_des_communes_1999 = [item["nom_1999"] for key, item in self.geodict.items()]
				closest_1999, distance_1999 = utils.find_closest_word_in_list(liste_des_communes_1999, ville)
				closest_1801, distance_1801 = utils.find_closest_word_in_list(liste_des_communes_1801, ville)
				if distance_1801 < distance_1999:
					closest_match = closest_1801
					commune_correspondante = next((item for item in self.geodict.values() if item["nom_1801"] == closest_match))
				else:
					closest_match = closest_1999
					commune_correspondante = next((item for item in self.geodict.values() if item["nom_1999"] == closest_match))
				current_name = closest_match
				longitude = commune_correspondante["longitude"]
				latitude = commune_correspondante["latitude"]

			result = {
				"lat": latitude,
				"lon": longitude,
				"nom_actuel": current_name
			}
		return result