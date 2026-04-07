import copy
import json
import re
import src.utils.utils as utils
import pandas as pd


class Reconciliator:
	def __init__(self, minute_list, previous_minute):
		self.minute_list = minute_list
		self.reconciliated_minute = {}
		# Trouvé dans https://www.insee.fr/fr/statistiques/3536630 (fichier de l'INSEE)
		self.list_of_surnames = [name.lower() for name in pd.read_csv("src/Information_Extractor/databases/french_surnames.csv",
																	  delimiter="\t")["NOM"].tolist()]
		# Idem: https://www.insee.fr/fr/statistiques/8595130
		self.list_of_names = [name.lower() if isinstance(name, str) else name for name in pd.read_csv("src/Information_Extractor/databases/french_names.csv",
																	  delimiter=";")["prenom"].tolist()]
		self.previous_minute  = previous_minute

		self.french_lexicon =  set([utils.remove_accents(word).lower() for word in utils.txt_to_list("src/resources/french_lexicon.txt") if
				 not word.isupper()])

		self.certitude_prenom_du_soldat = []
		self.prenom_du_soldat = []
		self.nom_du_soldat = None
		self.certitude_nom_du_soldat = None
		self.description_physique = None
		self.ville_residence = None
		self.ville_naissance = None
		self.magistrats = None
		self.numero_ordre = None
		self.numero_jugement = None
		self.lieu_jugement = None
		self.date_proces = None
		self.images_path = None
		self.annotations = None
		self.profession = None
		self.date_proces_orig = None
		self.date_naissance = None
		self.age = None

	def reconciliate_minute(self):
		self._add_images_path()
		self._reconciliate_nom_soldat()
		self._reconciliate_prenom_soldat()
		self._reconciliate_lieu_residence()
		self._reconciliate_lieu_naissance()
		self._retrieve_annotations()
		self._retrieve_profession()
		self._reconciliate_trial_date()
		self._reconciliate_date_naissance()
		self._copy_unici()
		self._produce_dict()
		self._remove_baseline_and_bbox()
		print("---")


	def _reconciliate_date_naissance(self):
		try:
			date_page_1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["date_naissance"]["extracted"]["when"]
		except TypeError:
			date_page_1 = None
		try:
			self.date_naissance = self.minute_list[0]["extractions"]["soldat"]["identite"]["date_naissance"]["extracted"]["when"]
		except TypeError:
			self.date_naissance = None
		age_soldat = self.minute_list[1]["extractions"]["soldat"]["identite"]["age"]["extracted"]
		if self.date_proces:
			try:
				age_theorique = utils.calcule_age(date_naissance=date_page_1, date_proces=self.date_proces)
			except AttributeError:
				age_theorique = None
			print(age_theorique)
			print(age_soldat)
			if age_theorique == age_soldat:
				self.age_soldat = age_soldat
			else:
				self.age_soldat = age_soldat
		else:
			self.age_soldat = age_soldat

	def _reconciliate_trial_date(self):
		try:
			date_page_1 = self.minute_list[0]["extractions"]["date_proces"]["normalized"]["when"]
		except (TypeError, KeyError):
			date_page_1 = None
		try:
			date_a_page_4 = self.minute_list[3]["extractions"]["date_proces_1"]["normalized"]["when"]
		except (TypeError, KeyError, IndexError):
			date_a_page_4 = None
		try:
			date_b_page_4 = self.minute_list[3]["extractions"]["date_proces_2"]["normalized"]["when"]
		except (TypeError, KeyError, IndexError):
			date_b_page_4 = None
		if date_page_1 == date_a_page_4 == date_b_page_4:
			self.date_proces = date_page_1
		else:
			filtered_dates = [date for date in [date_page_1, date_a_page_4, date_b_page_4] if date]
			orig_filtered_dates = copy.copy(filtered_dates)
			self.date_proces_orig = orig_filtered_dates
			try:
				previous_date = self.previous_minute["informations_proces"]["date_du_proces"]
				filtered_dates = [item for item in filtered_dates if utils.is_anterior_or_equal(previous_date, item)]
			except (AttributeError, TypeError):
				pass
			print(filtered_dates)
			# On va filtrer par précision: le / dit la précision du
			# dates_by_Precision = [len(date.split("/")) for date in filtered_dates]
			# filtered_dates_by_precision = [date for date in filtered_dates if len(date.split("/")) == max(dates_by_Precision)]
			print(filtered_dates)
			if filtered_dates == []:
				self.date_proces = None
				return
			if all([item == filtered_dates[0] for item in filtered_dates[1:]]):
				self.date_proces = filtered_dates[0]
			else:
				# Si la taille est de 2, on a 2 options possibles distinctes, on peut pas trancher sur les fréquences
				if len(filtered_dates) == 2:
					self.date_proces = filtered_dates
				else:
					dictionnary = {}
					for date in filtered_dates:
						try:
							dictionnary[date] += 1
						except KeyError:
							dictionnary[date] = 1
					dict_sorted_by_freq = sorted(dictionnary, key=dictionnary.get, reverse=True)
					print(dict_sorted_by_freq)
					exit(0)
					# self.date_proces =


	def _retrieve_profession(self):
		try:
			profession_page_1 = self.minute_list[0]["extractions"]["soldat"]["profession"]["extracted"].lower()
		except AttributeError:
			try:
				self.profession = self.minute_list[1]["extractions"]["soldat"]["profession"]["extracted"].lower()
			except AttributeError:
				self.profession = None
				return
		try:
			profession_page_2 = self.minute_list[1]["extractions"]["soldat"]["profession"]["extracted"].lower()
		except (AttributeError, IndexError):
			self.profession = self.minute_list[0]["extractions"]["soldat"]["profession"]["extracted"].lower()
			return

		if profession_page_1 == profession_page_2:
			self.profession = profession_page_1
			return
		else:
			lexicality_1 = utils.compute_lexicality(profession_page_1, words=self.french_lexicon)
			lexicality_2 = utils.compute_lexicality(profession_page_2, words=self.french_lexicon)

		if lexicality_1 > lexicality_2:
			self.profession = profession_page_1
		elif lexicality_1 < lexicality_2:
			self.profession = profession_page_2
		else:
			self.profession = [profession_page_1, profession_page_2]

	def _retrieve_annotations(self):
		self.annotations = []
		for page in self.minute_list:
			if "extractions" in page and "annotations_ajouts" in page["extractions"]:
				self.annotations.append(page["extractions"]["annotations_ajouts"])

	def _add_images_path(self):
		self.images_path = [item["image_path"] for item in self.minute_list]


	def _remove_baseline_and_bbox(self):
		self.reconciliated_minute = utils.delete_key("baseline", self.reconciliated_minute)
		self.reconciliated_minute = utils.delete_key("bbox", self.reconciliated_minute)

	def _copy_unici(self):
		"""
		Cette fonction récupèrer l'information qui ne se répète pas.
		:return:
		"""
		self.description_physique = self.minute_list[0]["extractions"]["soldat"]["description_physique"]
		try:
			self.magistrats = self.minute_list[0]["extractions"]["magistrats"]
		except KeyError:
			self.magistrats = None
		self.numero_ordre = self.minute_list[0]["extractions"]["numero_ordre"]
		self.numero_jugement = self.minute_list[0]["extractions"]["numero_jugement"]
		self.lieu_jugement = self.minute_list[0]["extractions"]["lieu_jugement"]
		try:
			self.date_crime_ou_delit = self.minute_list[0]["extractions"]["date_du_crime_ou_delit"]["normalized"]
		except TypeError:
			self.date_crime_ou_delit = None


	def _reconciliate_lieu_residence(self):
		try:
			nom_ville_transcrit_p1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["extracted"]
		except KeyError:
			nom_ville_transcrit_p1 = None
		try:
			nom_ville_identifie_p1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1801"]
		except KeyError:
			try:
				nom_ville_identifie_p1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1999"]
			except KeyError:
				nom_ville_identifie_p1 = None
		try:
			distance_p1 = utils.levensthein_distance(nom_ville_transcrit_p1, nom_ville_identifie_p1)
		except TypeError:
			print("Page 1 sans annotations")
			self.lieu_residence = None
			return
		self.lieu_residence = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_residence"]


		try:
			nom_ville_transcrit_p2 = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["extracted"]
		except (KeyError, IndexError, TypeError):
			nom_ville_transcrit_p2 = None
		try:
			nom_ville_identifie_p2 = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1801"]
		except (KeyError, IndexError, TypeError):
			try:
				nom_ville_identifie_p2 = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1999"]
			except (KeyError, IndexError, TypeError):
				nom_ville_identifie_p2 = None
		try:
			distance_p2 = utils.levensthein_distance(nom_ville_transcrit_p2, nom_ville_identifie_p2)
		except TypeError:
			self.lieu_residence = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]
			print("Page 2 sans annotations")
			return
		# Si la distance est plus grande c'est possiblement à cause d'une erreur sur le département
		# Autre option à envisager, faire la correction au niveau du département, extraire les villes à nouveau
		if distance_p1 < distance_p2:
			print("Page 1 choisie.")
			self.lieu_residence = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]
		else:
			print("Page 2 choisie.")
			try:
				self.lieu_residence = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_residence"]
			except KeyError:
				self.lieu_residence = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]
		print(f'Page 1: {self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_residence"]}')
		print(f'Page 2: {self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_residence"]}')


	def _reconciliate_lieu_naissance(self):
		try:
			nom_ville_transcrit_p1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["extracted"]
		except (KeyError, TypeError):
			nom_ville_transcrit_p1 = None
		try:
			nom_ville_identifie_p1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1801"]
		except (KeyError, TypeError):
			try:
				nom_ville_identifie_p1 = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1999"]
			except (KeyError, TypeError):
				nom_ville_identifie_p1 = None
		try:
			distance_p1 = utils.levensthein_distance(nom_ville_transcrit_p1, nom_ville_identifie_p1)
		except TypeError:
			print("Page 1 sans annotations")
			self.lieu_naissance = None
			# self.lieu_naissance = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_naissance"]
			return


		try:
			nom_ville_transcrit_p2 = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["extracted"]
		except (KeyError, IndexError, TypeError):
			nom_ville_transcrit_p2 = None
		try:
			nom_ville_identifie_p2 = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1801"]
		except (KeyError, IndexError, TypeError):
			try:
				nom_ville_identifie_p2 = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1999"]
			except (KeyError, IndexError, TypeError):
				nom_ville_identifie_p2 = None
		try:
			distance_p2 = utils.levensthein_distance(nom_ville_transcrit_p2, nom_ville_identifie_p2)
		except TypeError:
			self.lieu_naissance = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]
			print("Page 2 sans annotations")
			return
		# Si la distance est plus grande c'est possiblement à cause d'une erreur sur le département
		# Autre option à envisager, faire la correction au niveau du département, extraire les villes à nouveau
		if distance_p1 < distance_p2:
			print("Page 1 choisie.")
			self.lieu_naissance = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]
		else:
			print("Page 2 choisie.")
			try:
				self.lieu_naissance = self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_naissance"]
			except KeyError:
				self.lieu_naissance = self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]
		print(f'Page 1: {self.minute_list[0]["extractions"]["soldat"]["identite"]["lieu_naissance"]}')
		print(f'Page 2: {self.minute_list[1]["extractions"]["soldat"]["identite"]["lieu_naissance"]}')


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
			return
		try:
			prenoms_page_1 = self.minute_list[0]['extractions']['soldat']['identite']['prenom']['extracted']
		except KeyError:
			prenoms_page_1 = None
		try:
			prenoms_page_2 = self.minute_list[1]['extractions']['soldat']['identite']['prenom']['extracted']
		except KeyError:
			prenoms_page_2 = None
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
		try:
			nom_page_1 = self.minute_list[0]['extractions']['soldat']['identite']['nom']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_1 = None
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
			"metadata":
				{
				"images": self.images_path,
				"greffier": self.magistrats["greffier"]["extracted"]["persName"]
			},
			"magistrats": self.magistrats,
			"informations_proces": {"numero_ordre": self.numero_ordre,
									"numero_jugement": self.numero_jugement,
									"lieu_jugement": self.lieu_jugement,
									"date_du_proces": {"date_reconciliee": self.date_proces,
										   "date_originelle": self.date_proces_orig}},
			"soldat":
				{
					"identite": {
						"nom":
							{"nom": self.nom_du_soldat,
							 "certitude": self.certitude_nom_du_soldat},
						"prenom":
							{"prenom": self.prenom_du_soldat,
							 "certitude": self.certitude_prenom_du_soldat},
						"age": self.age_soldat,
						"date_naissance": self.date_naissance
						},
					"lieu_residence": self.lieu_residence,
					"lieu_naissance": self.lieu_naissance,
					"description_physique": self.description_physique,
					"profession": self.profession
				},
			"accusation": {
				"date_du_crime_ou_delit": self.date_crime_ou_delit
			},
			"actualisations_du_jugement": self.annotations
		}
