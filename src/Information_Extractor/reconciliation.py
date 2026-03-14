import json
import re

import pandas as pd


class Reconciliator:
	def __init__(self, minute_list):
		self.minute_list = minute_list
		self.reconciliated_minute = {}
		# Trouvé dans https://www.insee.fr/fr/statistiques/3536630 (fichier de l'INSEE)
		self.list_of_surnames = [name.lower() for name in pd.read_csv("src/Information_Extractor/databases/french_surnames.csv",
																	  delimiter="\t")["NOM"].tolist()]
		# Idem: https://www.insee.fr/fr/statistiques/8595130
		self.list_of_names = [name.lower() if isinstance(name, str) else name for name in pd.read_csv("src/Information_Extractor/databases/french_names.csv",
																	  delimiter=";")["prenom"].tolist()]

		self.certitude_prenom_du_soldat = []
		self.prenom_du_soldat = []
		self.nom_du_soldat = None
		self.certitude_nom_du_soldat = None
		self.description_physique = None

	def reconciliate_minute(self):
		self._reconciliate_nom_soldat()
		self._reconciliate_prenom_soldat()
		# self._copy_unici()
		self._produce_dict()

	def _copy_unici(self):
		"""
		Cette fonction récupèrer l'information qui ne se répète pas.
		:return:
		"""
		self.description_physique = self.minute_list[0]["extractions"]["soldat"]["description_physique"]


	def _reconciliate_prenom_soldat(self):
		"""
		Le nom du soldat est présent page 1, 2, 3.
		Une façon de gérer la réconciliation est d'aller chercher une liste de noms français
		Une autre façon: entraîner un petit réseau sur cette même liste
		pour trouver la probabilité qu'une chaîne soit vraisemblable.
		Scores: 0.9 quand il y a accord entre les deux transcriptions, 0.7 quand il y a désaccord (mais le nom est présent
		dans la liste de noms), 0.1 dans les autres cas
		:return: None
		"""
		if len(self.minute_list) == 1:
			self.prenom_du_soldat = self.minute_list[0]['extractions']['soldat']['identite']['prenom']['extracted']
		prenoms_page_1 = self.minute_list[0]['extractions']['soldat']['identite']['prenom']['extracted']
		prenoms_page_2 = self.minute_list[1]['extractions']['soldat']['identite']['prenom']['extracted']
		delimiter = re.compile(r"[.;\s\-]+")
		try:
			liste_prenoms_page_1 = re.split(delimiter, prenoms_page_1)
		except TypeError:
			liste_prenoms_page_1 = None
		try:
			liste_prenoms_page_2 = re.split(delimiter, prenoms_page_2)
		except TypeError:
			liste_prenoms_page_2 = None
		if liste_prenoms_page_1 is None and liste_prenoms_page_2:
			self.prenom_du_soldat = liste_prenoms_page_2
			return
		elif liste_prenoms_page_2 is None and liste_prenoms_page_1:
			self.prenom_du_soldat = liste_prenoms_page_1
			return
		else:
			self.prenom_du_soldat = None
			return

		if len(liste_prenoms_page_1) != len(liste_prenoms_page_2):
			print(liste_prenoms_page_2)
			print(liste_prenoms_page_1)
			print("Oups")
			exit(0)


		# On va zipper pour comparer un à un les multiples prénoms
		liste_comparaison_prenoms = list(zip(liste_prenoms_page_1, liste_prenoms_page_2))
		prenoms_correct = []
		print("---")
		for prenom_1, prenom_2 in liste_comparaison_prenoms:
			try:
				prenom_1 = prenom_1.lower()
			except AttributeError:
				pass
			try:
				prenom_2 = prenom_2.lower()
			except AttributeError:
				pass
			if prenom_1 == prenom_2:
				self.prenom_du_soldat.append(prenom_1)
				self.certitude_prenom_du_soldat.append(0.9)
				print("cas 1")
			else:
				if prenom_1 in self.list_of_names and prenom_2 not in self.list_of_names:
					self.prenom_du_soldat.append(prenom_1)
					self.certitude_prenom_du_soldat.append(0.7)
					print("cas 2")
				elif prenom_2 in self.list_of_names and prenom_1 not in self.list_of_names:
					self.prenom_du_soldat.append(prenom_2)
					self.certitude_prenom_du_soldat.append(0.7)
					print("cas 3")
				elif prenom_2 in self.list_of_names and prenom_1 in self.list_of_names:
					self.prenom_du_soldat.append([prenom_1, prenom_2])
					self.certitude_prenom_du_soldat.append(0.1)
					print("cas 4")
				else:
					self.prenom_du_soldat.append([prenom_1, prenom_2])
					self.certitude_prenom_du_soldat.append(0.1)
					print("cas 5")
		print(self.prenom_du_soldat)


	def _reconciliate_nom_soldat(self):
		"""
		Le nom du soldat est présent page 1, 2, 3.
		Une façon de gérer la réconciliation est d'aller chercher une liste de noms français
		Une autre façon: entraîner un petit réseau sur cette même liste
		pour trouver la probabilité qu'une chaîne soit vraisemblable.
		Scores: 0.9 quand il y a accord entre les deux transcriptions, 0.7 quand il y a désaccord (mais le nom est présent
		dans la liste de noms), 0.1 dans les autres cas
		:return: None
		"""
		# Attention, ne fonctionnera pas s'il y a plus ou moins de 4 pages dans la minute. Passer
		# Par la classification de la page.
		nom_page_1 = self.minute_list[0]['extractions']['soldat']['identite']['nom']['extracted'].upper()
		try:
			nom_page_2_a = self.minute_list[1]['extractions']['soldat']['identite']['nom_1']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_2_a = None
		try:
			nom_page_2_b = self.minute_list[1]['extractions']['soldat']['identite']['nom_2']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_2_b = None
		try:
			nom_page_3 = self.minute_list[2]['extractions']['identite']['nom']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_3 = None
		try:
			nom_page_4a = self.minute_list[3]['extractions']['identite']['nom_1']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_4a = None
		try:
			nom_page_4b = self.minute_list[3]['extractions']['identite']['nom_2']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_4b = None
		try:
			nom_page_4c = self.minute_list[3]['extractions']['identite']['nom_1']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_4c = None
		liste_noms = [nom_page_1, nom_page_2_a, nom_page_2_b, nom_page_3, nom_page_4a, nom_page_4b, nom_page_4c]
		if all([item == liste_noms[0] for item in liste_noms[1:]]):
			self.nom_du_soldat = nom_page_1
			self.certitude_nom_du_soldat = 0.9
		else:
			dictionnary = {}
			presences = [(item, item.lower() in self.list_of_surnames) for item in liste_noms if item]
			print(presences)
			for item, presence in presences:
				if presence is True:
					try:
						dictionnary[item] += 1
					except KeyError:
						dictionnary[item] = 1
			print(dictionnary)
			# https://www.geeksforgeeks.org/python/python-get-key-with-maximum-value-in-dictionary/
			dict_sorted_by_freq = sorted(dictionnary, key=dictionnary.get, reverse=True)
			# On va vérifier qu'on n'ait pas d'égalité des fréquences

			all_values = dictionnary.values()
			try:
				max_value = max(all_values)
			except ValueError:
				dictionnary = {}
				# Dans le cas où le dictionnaire est vide
				for item, _ in presences:
					try:
						dictionnary[item] += 1
					except KeyError:
						dictionnary[item] = 1
				dict_sorted_by_freq = sorted(dictionnary, key=dictionnary.get, reverse=True)
				all_values = dictionnary.values()
				max_value = max(all_values)
				if all([item == liste_noms[0] for item in liste_noms[1:] if item]):
					self.nom_du_soldat = dict_sorted_by_freq[0]
					self.certitude_nom_du_soldat = 0.5
				else:
					if list(all_values).count(max_value) != 1:
						print("Égalité entre deux noms")
						self.nom_du_soldat = [item for item, frec in dictionnary.items() if frec == max_value]
						self.certitude_nom_du_soldat = 0.1
					else:
						print("Pas d'égalité")
						print(list(all_values))
						print(list(all_values).count(max_value))
						self.nom_du_soldat = dict_sorted_by_freq[0]
						self.certitude_nom_du_soldat = 0.3
				print(self.nom_du_soldat)
				print(self.certitude_nom_du_soldat)
				print("OKKKK")
				print(dictionnary)
				print("---")
				return
			if list(all_values).count(max_value) != 1:
				print("égalité")
				self.nom_du_soldat = [item for item, frec in dictionnary.items() if frec == max_value]
				self.certitude_nom_du_soldat = 0.2
			else:
				self.nom_du_soldat = dict_sorted_by_freq[0]
				self.certitude_nom_du_soldat = 0.9
			print(self.nom_du_soldat)
			print(self.certitude_nom_du_soldat)


	def _produce_dict(self):
		self.reconciliated_minute = {
			"soldat":
				{
					"identite": {
						"nom":
							{"nom": self.nom_du_soldat,
							 "certitude": self.certitude_nom_du_soldat},
						"prenom":
							{"prenom": self.prenom_du_soldat,
							 "certitude": self.certitude_prenom_du_soldat}
						},
					"description_physique": self.description_physique
				}
		}
