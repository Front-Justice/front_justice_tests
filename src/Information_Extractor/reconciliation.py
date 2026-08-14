import copy
import logging
import math
from datetime import datetime
import json
import re
import shutil

import src.utils.utils as utils
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Reconciliator:
	"""
	Classe de réconciliation des informations extraites.
	"""
	def __init__(self, minute_list, previous_minute):
		self.minute_list = minute_list
		self.reconciliated_minute = {}
		# Trouvé dans https://www.insee.fr/fr/statistiques/3536630 (fichier de l'INSEE)
		self.list_of_surnames = [name.lower() for name in pd.read_csv("src/Information_Extractor/databases/french_surnames.csv",
																	  delimiter="\t")["NOM"].tolist()]
		# Idem: https://www.insee.fr/fr/statistiques/8595130
		self.list_of_names = {name.lower() if isinstance(name, str) else name:gender for gender, name in
							   pd.read_csv("src/Information_Extractor/databases/french_names.csv",
										   delimiter=";").values.tolist()}
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
		self.jours_guerre = None
		self.annee_guerre = None
		self.mois_guerre = None
		self.trimestre_guerre = None
		self.decision_tribunal = None
		self.plusieurs_soldats = False
		self.chef_accusation_extrait = None
		self.chef_accusation_normalise = None
		self.images_path = None
		self.annotations = None
		self.profession_normalisee = None
		self.date_proces_orig = None
		self.date_naissance = None
		self.naissance_hors_metropole = None
		self.naissance_etranger = None
		self.residence_hors_metropole = None
		self.residence_etranger = None
		self.questions = None
		self.condamnation_pecuniaire = None
		self.age = None
		self.appendices_number = None
		self.annotations_page_1 = None
		self.annotations_page_2 = None
		self.annotations_page_3 = None
		self.annotations_page_4 = None

	def _filter_pages(self):
		"""
		Cette fonction va créer les attributs permettant d'accéder aux pages spécifiques, indépendamment de
		leur position dans laminute (il peut y avoir des pages annexes intercalées).
		:return:
		"""
		self.annotations_page_1 = utils.filter_pages(self.minute_list, "page_1")
		self.annotations_page_2 = utils.filter_pages(self.minute_list, "page_2")
		self.annotations_page_3 = utils.filter_pages(self.minute_list, "page_3")
		self.annotations_page_4 = utils.filter_pages(self.minute_list, "page_4")

	def reconciliate_minute(self):
		if self._check_minute_consistency() is False:
			self.reconciliated_minute = {}
			return
		self._check_multiple_soldiers()
		self._filter_pages()
		self._count_number_appendices()
		self._add_images_path()
		self._reconciliate_nom_soldat()
		self._reconciliate_prenom_soldat()
		self._reconciliate_lieu_residence()
		self._reconciliate_lieu_naissance()
		self._retrieve_annotations()
		self._retrieve_profession()
		self._retrieve_categorie_sociopro()
		self._reconciliate_questions()
		self._reconciliate_trial_date()
		self._reconciliate_date_naissance()
		self._reconciliate_condamnation()
		self._copy_unici()
		self._produce_dict()
		# self._remove_baseline_and_bbox()
		print("---")

	def _reconciliate_condamnation(self):
		try:
			condamnation = self.annotations_page_3["extractions"]["decision_tribunal"]["decision_normalisee"]
		except KeyError:
			condamnation = None
		try:
			peine = self.annotations_page_3["extractions"]["decision_tribunal"]["peine"]["extracted"]["type"]
		except KeyError:
			peine = None
		try:
			sursis = self.annotations_page_3["extractions"]["decision_tribunal"]["sursis"]
		except KeyError:
			sursis = None

		if condamnation == "UNK" and (peine != None or sursis is True):
			logger.info("La condamnation n'a pas été extraite, mais la peine ou le sursis sont identifiés. Le soldat"
						"a donc été condamné.")
			self.condamnation = "condamnation"
		elif condamnation == "UNK" and peine == None and sursis is False:
			logger.info("La condamnation n'a pas été extraite, la peine n'est identifiée, il n'y a pas de sursis."
						" Le soldat a probablement été acquitté.")
			self.condamnation = "acquittement"
		else:
			self.condamnation = condamnation

	def _check_multiple_soldiers(self):
		try:
			if any(page["extractions"] == {"commentaire": "Plusieurs soldats"} for page in self.minute_list):
				self.plusieurs_soldats = True
		except KeyError:
			pass

	def _count_number_appendices(self):
		self.appendices_number = len([item for item in self.minute_list if item["classe"] == "page_autre"])


	def _check_minute_consistency(self):
		try:
			classes = [int(item["classe"].split("_")[-1]) for item in self.minute_list if item["classe"] not in ["page_autre", "page_manuscrite_suivie"]]
		except ValueError:
			logger.error(f'Erreur sur la minute, vérifier la classification: {[item["classe"] for item in self.minute_list]}')
			return False
		try:
			if self.minute_list[-2]["classe"] in ["page_autre", "page_manuscrite_suivie"] and classes == [1, 2, 4]:
				logger.info(f'Classification originale: {[item["classe"] for item in self.minute_list]}')
				self.minute_list[-2]["classe"] == "page_3"
				logger.info(f'Classification reconstruite: {[item["classe"] for item in self.minute_list]}')
				return True
		except KeyError:
			return False
		if classes == [1, 2, 3, 4]:
			logger.info("Minute correctement ordonnée")
			return True
		else:
			logger.info("Quelque chose ne va pas avec la minute")
			[shutil.copy(image["image_path"], f"debug/") for image in self.minute_list]
			return False

	def _reconciliate_questions(self):
		try:
			questions_page_2 = self.annotations_page_2["extractions"]["questions"]["extracted"]
		except  (KeyError, TypeError, IndexError):
			questions_page_2 = ""
		try:
			questions_page_3 = self.annotations_page_3["extractions"]["questions"]["extracted"]
		except (KeyError, TypeError, IndexError):
			questions_page_3 = ""

		self.questions = f"{questions_page_2} {questions_page_3}".strip()


	def _reconciliate_date_naissance(self):

		try:
			prediction = self.annotations_page_2["extractions"]["soldat"]["prediction"]
		except  (KeyError, TypeError):
			prediction = None
		try:
			entites = self.annotations_page_2["extractions"]["soldat"]["entites"]
		except  (KeyError, TypeError):
			entites = None
		try:
			date_page_1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["date_naissance"]["extracted"]["when"]
		except  (KeyError, TypeError):
			date_page_1 = None
		try:
			self.date_naissance = self.annotations_page_1["extractions"]["soldat"]["identite"]["date_naissance"]["extracted"]["when"]
		except  (KeyError, TypeError):
			self.date_naissance = None
		try:
			age_soldat = self.annotations_page_2["extractions"]["soldat"]["identite"]["age"]["extracted"]
		except (IndexError, TypeError, KeyError):
			logging.error("L'âge n'a pas été identifié en page 2.")
			logging.error(f"Prediction: {prediction}")
			logging.error(f"Entités: {entites}")
			age_soldat = None
		if self.date_proces and date_page_1:
			try:
				age_theorique = utils.calcule_age(date_naissance=date_page_1, date_proces=self.date_proces)
			except (AttributeError, TypeError, KeyError):
				logging.error(f"Le calcul de l'âge a échoué: date de naissance: {date_page_1}; date du procès: {self.date_proces}")
				age_theorique = None
			if age_theorique == age_soldat:
				logging.info(f"L'âge théorique concorde avec l'âge du soldat: {age_soldat} ans. Les informations sont correctes.")
				self.age_soldat = age_soldat
			elif age_theorique != age_soldat and age_soldat:
				try:
					if 18 < int(age_soldat) < 80:
						logging.info(f"L'âge extrait du soldat est vraisemblable: {age_soldat}.")
						self.age_soldat = age_soldat
					else:
						logging.error(f"L'âge extrait du soldat est invraisemblable. On prend l'âge théorique: {age_theorique}")
						self.age_soldat = age_theorique
				except ValueError:
					logging.error(f"L'âge du soldat n'a pas été correctement extrait: {age_soldat}. On prend l'âge calculé.")
					logging.error(f"Prediction: {prediction}")
					logging.error(f"Entités: {entites}")
					self.age_soldat = age_theorique
			elif not age_soldat and age_theorique:
				logging.info(f"Âge non identifié. Âge calculé: {age_theorique}")
				self.age_soldat = age_theorique
			else:
				self.age_soldat = None
		else:
			logging.info(f"L'âge du soldat ne peut être recalculé. On garde l'âge identifié: {age_soldat}.")
			self.age_soldat = age_soldat
		try:
			self.age_soldat = int(self.age_soldat)
		except (ValueError, TypeError):
			logging.error(f"L'âge du soldat n'a pas été correctement extrait: {self.age_soldat}.")
			self.age_soldat = None
		if self.age_soldat and date_page_1:
			annee_naissance = date_page_1.split("/")[-1]
			verif_age = 1912 < int(self.age_soldat) + int(annee_naissance) < 1921
			if verif_age is True:
				logging.info("Âge du soldat correspondant avec les dates de la guerre.")
			else:
				logging.error(f"Âge du soldat discordant avec les années de guerre: {self.age_soldat} et {annee_naissance}.")
				self.age_soldat = None


	def _reconciliate_trial_date(self):
		try:
			date_page_1 = self.annotations_page_1["extractions"]["date_proces"]["normalized"]["when"]
		except (TypeError, KeyError, IndexError):
			date_page_1 = None
		try:
			date_a_page_4 = self.annotations_page_4["extractions"]["date_proces_1"]["normalized"]["when"]
		except (TypeError, KeyError, IndexError):
			date_a_page_4 = None
		try:
			date_b_page_4 = self.annotations_page_4["extractions"]["date_proces_2"]["normalized"]["when"]
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
			# On va filtrer par précision: le / dit la précision du
			# dates_by_Precision = [len(date.split("/")) for date in filtered_dates]
			# filtered_dates_by_precision = [date for date in filtered_dates if len(date.split("/")) == max(dates_by_Precision)]
			if filtered_dates == []:
				self.date_proces = None
				return
			# Cas le plus simple, toutes les trois dates s'accordent
			if all([item == filtered_dates[0] for item in filtered_dates[1:]]):
				logging.info(f"Toutes les dates s'accordent: {filtered_dates[0]}")
				self.date_proces = filtered_dates[0]
			else:
				logging.info(f"Il y a désaccord dans les dates retenues: {filtered_dates}")
				# Si la taille est de 2, on a 2 options possibles distinctes, on peut pas trancher sur les fréquences
				if len(filtered_dates) == 2:
					logging.info(f"Deux dates retenues: {filtered_dates[0], filtered_dates[1]}")
					self.date_proces = self.reconciliate_date(filtered_dates[0], filtered_dates[1])
					logging.info(f"Date conservées: {self.date_proces}")

				# Si la taille est de 3
				else:
					# On regarde la précision des dates récupérées
					full_precision_date = [item for item in filtered_dates if len(item.split("/")) == 3]
					if len(full_precision_date) == 1:
						logging.info(f"Une date précise: {full_precision_date[0]}")
						self.date_proces = full_precision_date[0]
					elif len(full_precision_date) == 2:
						logging.info(f"Deux dates retenues: {filtered_dates[0], filtered_dates[1]}")
						self.date_proces = self.reconciliate_date(full_precision_date[0], full_precision_date[1])
						logging.info(f"Date conservées: {self.date_proces}")
					else:
						dictionnary = {}
						for date in filtered_dates:
							try:
								dictionnary[date] += 1
							except KeyError:
								dictionnary[date] = 1
						dict_sorted_by_freq = sorted(dictionnary, key=dictionnary.get, reverse=True)
						logging.info(f"La date la plus fréquente est: {dict_sorted_by_freq[0]} parmi {filtered_dates}")
						self.date_proces = dict_sorted_by_freq[0]
		if self.date_proces:
			# On va compter les jours, mois, trimestre et année de guerre.
			debut_guerre = "1914-09-03"
			precision_date = len(self.date_proces.split("/"))
			if len(self.date_proces.split("/")) == 2:
				self.date_proces = self.date_proces.split("/")[0] + "/" + self.date_proces.split("/")[1] + "/19" + self.date_proces.split("/")[2]
			if precision_date == 3:
				# https://stackoverflow.com/a/8419655
				d1 = datetime.strptime(debut_guerre, "%Y-%m-%d")
				d2 = datetime.strptime(self.date_proces, "%d/%m/%Y")
				self.jours_guerre = abs((d2 - d1).days)
				self.annee_guerre = math.ceil(self.jours_guerre / 365)
				self.mois_guerre = math.ceil(self.jours_guerre / 30)
				self.trimestre_guerre = math.ceil(self.jours_guerre / 90)

			# Dns le cas d'une précision au mois, on accepte 1 mois/trimestre d'écart
			elif precision_date == 2:
				self.date_proces = f"01/{self.date_proces}"
				d1 = datetime.strptime(debut_guerre, "%Y-%m-%d")
				d2 = datetime.strptime(self.date_proces, "%d/%m/%Y")
				jours_guerre = abs((d2 - d1).days)
				self.jours_guerre = None
				self.annee_guerre = math.ceil(jours_guerre / 365)
				self.mois_guerre = math.ceil(jours_guerre / 30)
				self.trimestre_guerre = math.ceil(jours_guerre / 90)

			# Dns le cas d'une précision à l'année, on né calcule que la durée de la guerre
			elif precision_date == 1:
				self.date_proces = f"01/01/{self.date_proces}"
				d1 = datetime.strptime(debut_guerre, "%Y-%m-%d")
				try:
					d2 = datetime.strptime(self.date_proces, "%d/%m/%Y")
				except ValueError:
					self.jours_guerre = None
					self.annee_guerre = None
					self.mois_guerre = None
					self.trimestre_guerre = None
					return
				jours_guerre = abs((d2 - d1).days)
				self.jours_guerre = None
				self.annee_guerre = math.ceil(jours_guerre / 365)
				self.mois_guerre = None
				self.trimestre_guerre = None



	def _retrieve_profession(self):
		try:
			profession_page_1_extraite = self.annotations_page_1["extractions"]["soldat"]["profession"]["extracted"].lower()
		except (AttributeError, IndexError, TypeError, KeyError):
			profession_page_1_extraite = None
		try:
			profession_page_1_normalisee = self.annotations_page_1["extractions"]["soldat"]["profession"]["normalized"].lower()
		except (AttributeError, IndexError, TypeError, KeyError):
			profession_page_1_normalisee = None

		try:
			profession_page_2_normalisee = self.annotations_page_2["extractions"]["soldat"]["profession"][
				"normalized"].lower()
		except (AttributeError, IndexError, TypeError, KeyError):
			profession_page_2_normalisee = None
		try:
			profession_page_2_extraite = self.annotations_page_2["extractions"]["soldat"]["profession"]["extracted"].lower()
		except (AttributeError, IndexError, TypeError, KeyError):
			profession_page_2_extraite = None

		if profession_page_1_normalisee == profession_page_2_normalisee:
			self.profession_normalisee = profession_page_1_normalisee
		else:
			self.profession_normalisee = [item for item in [profession_page_1_normalisee, profession_page_2_normalisee] if item]
			
		if profession_page_1_extraite == profession_page_2_extraite:
			self.profession_extraite = profession_page_1_extraite
		else:
			self.profession_extraite = [item for item in [profession_page_1_extraite, profession_page_2_extraite] if item]


	def _retrieve_categorie_sociopro(self):
		try:
			categorie_sociopro_p1 = self.annotations_page_1["extractions"]["soldat"]["profession"]["categorie_socioprofessionnelle"]
		except (KeyError, TypeError):
			categorie_sociopro_p1 = None

		try:
			categorie_sociopro_p2 = self.annotations_page_2["extractions"]["soldat"]["profession"]["categorie_socioprofessionnelle"]
		except (KeyError, TypeError):
			categorie_sociopro_p2 = None

		if categorie_sociopro_p1 == categorie_sociopro_p2 != None:
			self.categorie_sociopro = categorie_sociopro_p1
		elif categorie_sociopro_p1 == categorie_sociopro_p2 == None:
			self.categorie_sociopro = "UNK"
		elif categorie_sociopro_p1 != categorie_sociopro_p2 != None:
			self.categorie_sociopro = "UNK"
		elif categorie_sociopro_p1 == None and categorie_sociopro_p2 != None:
			self.categorie_sociopro = categorie_sociopro_p2
		elif categorie_sociopro_p1 != None and categorie_sociopro_p2 == None:
			self.categorie_sociopro = categorie_sociopro_p1
		else:
			self.categorie_sociopro = "ERROR"

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
		try:
			self.decision_tribunal = self.annotations_page_3["extractions"]["decision_tribunal"]
			self.decision_tribunal["decision_normalisee"] = self.condamnation
		except (TypeError, KeyError, IndexError):
			self.decision_tribunal = None
		try:
			self.description_physique = self.annotations_page_1["extractions"]["soldat"]["description_physique"]
		except (TypeError, KeyError, IndexError):
			self.description_physique = None
		try:
			self.magistrats = self.annotations_page_1["extractions"]["magistrats"]
		except (TypeError, KeyError, IndexError):
			self.magistrats = None
		try:
			self.numero_ordre = self.annotations_page_1["extractions"]["numero_ordre"]
		except (TypeError, KeyError, IndexError):
			self.numero_ordre = None
		try:
			self.numero_jugement = self.annotations_page_1["extractions"]["numero_jugement"]
		except (TypeError, KeyError, IndexError):
			self.numero_ordre = None
		try:
			self.lieu_jugement = self.annotations_page_1["extractions"]["lieu_jugement"]
		except (TypeError, KeyError, IndexError):
			self.lieu_jugement = None
		try:
			self.date_crime_ou_delit = self.annotations_page_1["extractions"]["date_du_crime_ou_delit"]["normalized"]
		except (TypeError, KeyError, IndexError):
			self.date_crime_ou_delit = None
		try:
			self.chef_accusation_extrait = self.annotations_page_1["extractions"]["chef_accusation"]["extracted"]
		except (TypeError, KeyError, IndexError):
			self.chef_accusation_extrait = None
		try:
			self.chef_accusation_normalise = self.annotations_page_1["extractions"]["chef_accusation"]["normalized"]
		except (TypeError, KeyError, IndexError):
			self.chef_accusation_normalise = None
		try:
			self.antecedents = self.annotations_page_1["extractions"]["antécédents"]["extracted"] if isinstance(self.annotations_page_1["extractions"]["antécédents"]["extracted"], list) else []
		except (TypeError, KeyError, IndexError):
			self.antecedents = None
		try:
			self.situation_maritale = self.annotations_page_1["extractions"]["soldat"]["identite"]["situation_maritale"]["situation"]["extracted"]
		except (TypeError, KeyError, IndexError):
			self.situation_maritale = None
		try:
			self.enfants = self.annotations_page_1["extractions"]["soldat"]["identite"]["situation_maritale"]["enfants"]["extracted"]
		except (TypeError, KeyError, IndexError) as e:
			self.enfants = None


		try:
			self.rang = self.annotations_page_1["extractions"]["soldat"]["identite"]["rang"]["extracted"]
		except (TypeError, KeyError, IndexError):
			self.rang = None
		try:
			self.rang_normalise = self.annotations_page_1["extractions"]["soldat"]["identite"]["rang"]["normalized"]
		except (TypeError, KeyError, IndexError):
			self.rang_normalise = None
		try:
			self.affectation = self.annotations_page_1["extractions"]["soldat"]["identite"]["affectation"]["extracted"]
		except (TypeError, KeyError, IndexError):
			self.affectation = None
		try:
			self.numero_matricule = self.annotations_page_1["extractions"]["soldat"]["identite"]["matricule"]["extracted"]
		except (TypeError, KeyError, IndexError):
			self.numero_matricule = None

		try:
			self.condamnation_pecuniaire = self.annotations_page_4["extractions"]["dernier_paragraphe"]["extracted"]["value"]
		except (TypeError, KeyError, IndexError):
			self.condamnation_pecuniaire = None
		if self.condamnation_pecuniaire:
			# Si la conversion ne fonctionne pas, on va récupérer la somme du tableau.
			try:
				float(self.condamnation_pecuniaire)
			except ValueError:
				try:
					self.condamnation_pecuniaire = self.annotations_page_4["extractions"]["tableau_frais"]["extracted"]["frais_totaux"]["totaux_transcrits"]
				except (TypeError, KeyError):
					self.condamnation_pecuniaire = "UNK"

	def reconciliate_date(self, date_a: str, date_b: str):
		"""
		Réconcilie deux dates, en passant à une précision inférieure si les jours divergent mais que mois et année sont identiques.
		:param date_a: date sous la forme jj/mm/aaaa
		:param date_b: date sous la forme jj/mm/aaaa
		:return: la date réconciliée
		"""

		if utils.is_anterior_or_equal(date_a, "03/09/1914"):
			logger.warning(f"La date {date_a} est antérieure au début de la guerre, erreur.")
			return date_b
		if utils.is_anterior_or_equal(date_b, "03/09/1914"):
			logger.warning(f"La date {date_b} est antérieure au début de la guerre, erreur.")
			return date_a
		splitted_a = date_a.split("/")
		splitted_b = date_b.split("/")
		# Dans le cas de dates de longueur identique (jj/mm/aaaa ou mm/aaaa)
		if len(splitted_a) == len(splitted_b):
			# Si les mois correspondent, on retourne la date avec précision au mois
			if splitted_a[1:] == splitted_b[1:]:
				reconciliated_date = "/".join(splitted_a)
			else:
				reconciliated_date = splitted_a[-1]
		# Dans le cas contraire
		else:
			if len(splitted_a) > len(splitted_b):
				return date_a
			else:
				return date_b
			# Si les mois + années coincident, on retourne la date avec précision au mois
			if splitted_a[-2:] == splitted_b[-2:]:
				reconciliated_date = "/".join(splitted_a)
			else:
				reconciliated_date = splitted_a[-1]
		return reconciliated_date


	def _reconciliate_lieu_residence(self):
		try:
			nom_ville_transcrit_p1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["extracted"]
		except (KeyError, TypeError):
			nom_ville_transcrit_p1 = None
		try:
			nom_ville_identifie_p1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1801"]
		except (KeyError, TypeError):
			nom_ville_identifie_p1 = None
		if nom_ville_identifie_p1 is None:
			try:
				nom_ville_identifie_p1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1999"]
			except (KeyError, TypeError):
				nom_ville_identifie_p1 = None
		try:
			distance_p1 = utils.levensthein_distance(nom_ville_transcrit_p1, nom_ville_identifie_p1)
		except TypeError:
			distance_p1 = None


		try:
			nom_ville_transcrit_p2 = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["extracted"]
		except (KeyError, IndexError, TypeError):
			nom_ville_transcrit_p2 = None
		try:
			nom_ville_identifie_p2 = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1801"]
		except (KeyError, IndexError, TypeError):
			nom_ville_identifie_p2 = None
		if nom_ville_identifie_p2 is None:
			try:
				nom_ville_identifie_p2 = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_residence"]["ville"]["nom_1999"]
			except (KeyError, IndexError, TypeError):
				nom_ville_identifie_p2 = None
		try:
			distance_p2 = utils.levensthein_distance(nom_ville_transcrit_p2, nom_ville_identifie_p2)
		except TypeError:
			distance_p2 = None
		# Si la distance est plus grande c'est possiblement à cause d'une erreur sur le département
		# Autre option à envisager, faire la correction au niveau du département, extraire les villes à nouveau
		if distance_p1 is None:
			try:
				self.lieu_residence = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_residence"]
			except (KeyError, TypeError):
				self.lieu_residence = "UNK"
			return
		elif distance_p2 is None:
			self.lieu_residence = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_residence"]
			return
		if distance_p1 < distance_p2:
			self.lieu_residence = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_residence"]
		else:
			try:
				self.lieu_residence = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_residence"]
			except (KeyError, TypeError):
				self.lieu_residence = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_residence"]



	def _reconciliate_lieu_naissance(self):
		try:
			nom_ville_transcrit_p1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["extracted"]
		except (KeyError, TypeError):
			nom_ville_transcrit_p1 = None
		try:
			nom_ville_identifie_p1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1801"]
		except (KeyError, TypeError):
			nom_ville_identifie_p1 = None
		if nom_ville_identifie_p1 is None:
			try:
				nom_ville_identifie_p1 = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1999"]
			except (KeyError, TypeError):
				nom_ville_identifie_p1 = None
		try:
			distance_p1 = utils.levensthein_distance(nom_ville_transcrit_p1, nom_ville_identifie_p1)
		except TypeError:
			distance_p1 = None


		try:
			nom_ville_transcrit_p2 = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["extracted"]
		except (KeyError, IndexError, TypeError):
			nom_ville_transcrit_p2 = None
		try:
			nom_ville_identifie_p2 = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1801"]
		except (KeyError, IndexError, TypeError):
			nom_ville_identifie_p2 = None
		if nom_ville_identifie_p2 is None:
			try:
				nom_ville_identifie_p2 = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_naissance"]["ville"]["nom_1999"]
			except (KeyError, IndexError, TypeError):
				nom_ville_identifie_p2 = None
		try:
			distance_p2 = utils.levensthein_distance(nom_ville_transcrit_p2, nom_ville_identifie_p2)
		except TypeError:
			distance_p2 = None
		# Si la distance est plus grande c'est possiblement à cause d'une erreur sur le département
		# Autre option à envisager, faire la correction au niveau du département, extraire les villes à nouveau
		if distance_p1 is None:
			try:
				self.lieu_naissance = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_naissance"]
			except (KeyError, TypeError):
				self.lieu_naissance = "UNK"
			return
		elif distance_p2 is None:
			self.lieu_naissance = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_naissance"]
			return
		if distance_p1 < distance_p2:
			self.lieu_naissance = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_naissance"]
		else:
			try:
				self.lieu_naissance = self.annotations_page_2["extractions"]["soldat"]["identite"]["lieu_naissance"]
			except (KeyError, TypeError):
				self.lieu_naissance = self.annotations_page_1["extractions"]["soldat"]["identite"]["lieu_naissance"]


	def _reconciliate_prenom_soldat(self):
		"""
		Le nom du soldat est présent page 1, 2, 3.
		:return: None
		"""
		if len(self.minute_list) == 1:
			try:
				self.prenom_du_soldat = self.annotations_page_1['extractions']['soldat']['identite']['prenom']['extracted']
			except (TypeError, KeyError, IndexError):
				self.prenom_du_soldat = None
			return
		try:
			prenoms_page_1 = self.annotations_page_1['extractions']['soldat']['identite']['prenom']['extracted']
		except (KeyError, TypeError):
			prenoms_page_1 = None
		try:
			prenoms_page_2 = self.annotations_page_2['extractions']['soldat']['identite']['prenom']['extracted']
		except (TypeError, KeyError, IndexError):
			prenoms_page_2 = None
		delimiter = re.compile(r"[.;\s\-]+")
		try:
			liste_prenoms_page_1 = re.split(delimiter, prenoms_page_1)
		except (TypeError, KeyError, IndexError):
			liste_prenoms_page_1 = None
		try:
			liste_prenoms_page_2 = re.split(delimiter, prenoms_page_2)
		except (TypeError, KeyError, IndexError):
			liste_prenoms_page_2 = None
		if liste_prenoms_page_1 is None and liste_prenoms_page_2:
			self.prenom_du_soldat = liste_prenoms_page_2
			return
		elif liste_prenoms_page_2 is None and liste_prenoms_page_1:
			self.prenom_du_soldat = liste_prenoms_page_1
			return


		# On va zipper pour comparer un à un les multiples prénoms
		try:
			liste_comparaison_prenoms = list(zip(liste_prenoms_page_1, liste_prenoms_page_2))
		except TypeError:
			self.prenom_du_soldat = None
			return
		prenoms_correct = []
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
			else:
				if prenom_1 in self.list_of_names and prenom_2 not in self.list_of_names:
					self.prenom_du_soldat.append(prenom_1)
					self.certitude_prenom_du_soldat.append(0.7)
				elif prenom_2 in self.list_of_names and prenom_1 not in self.list_of_names:
					self.prenom_du_soldat.append(prenom_2)
					self.certitude_prenom_du_soldat.append(0.7)
				elif prenom_2 in self.list_of_names and prenom_1 in self.list_of_names:
					self.prenom_du_soldat.extend([prenom_1, prenom_2])
					self.certitude_prenom_du_soldat.append(0.1)
				else:
					self.prenom_du_soldat.extend([prenom_1, prenom_2])
					self.certitude_prenom_du_soldat.append(0.1)
		if isinstance(self.prenom_du_soldat, list):
			self.prenom_du_soldat = " ".join(self.prenom_du_soldat)


	def _reconciliate_nom_soldat(self):
		"""
		Le nom du soldat est présent page 1, 2, 3.
		Une façon de gérer la réconciliation est d'aller chercher une liste de noms français
		Scores: 0.9 quand il y a accord entre les deux transcriptions, 0.7 quand il y a désaccord (mais le nom est présent
		dans la liste de noms), 0.1 dans les autres cas
		:return: None
		"""
		# Attention, ne fonctionnera pas s'il y a plus ou moins de 4 pages dans la minute. Passer
		# Par la classification de la page.
		try:
			nom_page_1 = self.annotations_page_1['extractions']['soldat']['identite']['nom']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_1 = None
		try:
			nom_page_2_a = self.annotations_page_2['extractions']['soldat']['identite']['nom_1']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_2_a = None
		try:
			nom_page_2_b = self.annotations_page_2['extractions']['soldat']['identite']['nom_2']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_2_b = None
		try:
			nom_page_3 = self.annotations_page_3['extractions']['identite']['nom']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_3 = None
		try:
			nom_page_4a = self.annotations_page_4['extractions']['identite']['nom_1']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_4a = None
		try:
			nom_page_4b = self.annotations_page_4['extractions']['identite']['nom_2']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_4b = None
		try:
			nom_page_4c = self.annotations_page_4['extractions']['identite']['nom_1']['extracted'].upper()
		except (TypeError, AttributeError, KeyError, IndexError):
			nom_page_4c = None
		liste_noms = [nom_page_1, nom_page_2_a, nom_page_2_b, nom_page_3, nom_page_4a, nom_page_4b, nom_page_4c]
		if all([item == liste_noms[0] for item in liste_noms[1:]]):
			logging.info(f"Tous les noms du soldat correspondent: {liste_noms[0]}")
			self.nom_du_soldat = nom_page_1
			self.certitude_nom_du_soldat = 1
		else:
			dictionnary = {}
			# On va vérifier pour chaque nom si on le retrouve dans la liste d'autorité
			presences = [(item, 1 if item.lower() in self.list_of_surnames else 0.5) for item in liste_noms if item]
			for item, coef in presences:
				try:
					dictionnary[item] += coef
				except KeyError:
					dictionnary[item] = coef
			# https://www.geeksforgeeks.org/python/python-get-key-with-maximum-value-in-dictionary/
			# On classe par fréquence
			logger.info(f"Noms du soldat: {liste_noms}")
			logger.info(f"Noms du soldat classés par fréquence pondérée: {dictionnary}")
			dict_sorted_by_freq = sorted(dictionnary, key=dictionnary.get, reverse=True)

			all_values = dictionnary.values()
			max_value = max(all_values)
			# Dans le cas où on a 2 noms qui apparaissent autant l'un que l'autre
			if list(all_values).count(max_value) != 1:
				logger.warning(f"Impossible de trier le nom du soldat par fréquence: "
							   f"{[item for item, frec in dictionnary.items() if frec == max_value]}")
				self.nom_du_soldat = [item for item, frec in dictionnary.items() if frec == max_value]
				self.certitude_nom_du_soldat = 0.2
			else:
				logger.info(f"Nom du soldat identifié: {dict_sorted_by_freq[0]}")
				self.nom_du_soldat = dict_sorted_by_freq[0]
				self.certitude_nom_du_soldat = 0.9


	def _produce_dict(self):
		try:
			greffier = self.magistrats["greffier"]["extracted"]["persName"]
		except (TypeError, KeyError):
			greffier = None
		if self.plusieurs_soldats is False:
			self.reconciliated_minute = {
				"metadata":
					{
					"images": self.images_path,
					"greffier": greffier,
					"nombre_pages_annexes": self.appendices_number,
					"plusieurs_soldats": self.plusieurs_soldats
				},
				"informations_proces": {"numero_ordre": self.numero_ordre,
										"numero_jugement": self.numero_jugement,
										"lieu_jugement": self.lieu_jugement,
										"date_du_proces": {"date_reconciliee": self.date_proces,
										"date_originelle": self.date_proces_orig,
									   "jours_de_guerre": self.jours_guerre,
									   "mois_de_guerre": self.mois_guerre,
									   "trimestres_de_guerre": self.trimestre_guerre,
									   "annees_de_guerre": self.annee_guerre},
				"magistrats": self.magistrats},
				"soldat":
					{
						"situation_militaire":
							{"rang_extrait": self.rang,
							"rang_normalise": self.rang_normalise,
						"affectation": self.affectation,
							 "matricule": self.numero_matricule},
						"identite": {
							"nom":
								{"extracted": self.nom_du_soldat,
								 "certitude": self.certitude_nom_du_soldat},
							"prenom":
								{"extracted": self.prenom_du_soldat,
								 "certitude": self.certitude_prenom_du_soldat},
							"age": self.age_soldat,
							"date_naissance": self.date_naissance,
							"lieu_residence": self.lieu_residence,
							"lieu_naissance": self.lieu_naissance,
							"profession_extraite": self.profession_extraite,
							"profession_normalisee": self.profession_normalisee,
							"categorie_socioprofessionnelle": self.categorie_sociopro,
							"famille":
							{"situation_maritale": self.situation_maritale,
							 "enfants": self.enfants},
							},
						"description_physique": self.description_physique,
						"antecedents": self.antecedents
					},
				"accusation": {
					"date_du_crime_ou_delit": self.date_crime_ou_delit,
					"chef_accusation_extrait": self.chef_accusation_extrait,
					"chef_accusation_normalise": self.chef_accusation_normalise,
					"questions_posees": self.questions
				},
				"decision_tribunal": {
					"frais": self.condamnation_pecuniaire,
					"jugement": self.decision_tribunal
				},
				"actualisations_du_jugement": self.annotations
			}
		else:
			self.reconciliated_minute = {
				"metadata":
					{
						"images": self.images_path,
						"greffier": greffier,
						"nombre_pages_annexes": self.appendices_number,
						"plusieurs_soldats": self.plusieurs_soldats
					},
				"informations_proces": {"numero_ordre": self.numero_ordre,
										"numero_jugement": self.numero_jugement,
										"lieu_jugement": self.lieu_jugement,
										"date_du_proces": {"date_reconciliee": self.date_proces,
														   "date_originelle": self.date_proces_orig,
									   "jours_de_guerre": self.jours_guerre,
									   "mois_de_guerre": self.mois_guerre,
									   "trimestres_de_guerre": self.trimestre_guerre,
									   "annees_de_guerre": self.annee_guerre},
										"magistrats": self.magistrats}
			}
