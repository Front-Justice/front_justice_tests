import logging

import src.utils.utils as utils
import re
from text_to_num import text2num
import src.date.parse_date as date
import regex
from src.utils.utils import OCRLine, OCRRecord
import src.Information_Extractor.similarity as similarity
import logging

logger = logging.getLogger(__name__)


def extraire_entite_baseline(entities_list: list,
							 nom_entite: str,
							 target_lines: OCRRecord,
							 image_path: str):
	"""
	Cette fonction extrait d'un dictionnaire contenant les entités reconnues par le NER
	une entité en particulier.
	:param dictionnaire:
	:param nom_entite:
	:return: une liste de dictionnaire de la forme: [{"extracted": "entite",
														"baseline": baseline}]
	"""
	dictionnaire = utils.entities_to_dict(entities_list)
	try:
		entites_extraites = dictionnaire[nom_entite]
	except KeyError:
		return None
	if len(entites_extraites) == 1:
		target_baseline = utils.get_baseline_from_string(line=target_lines,
														 target_string=entites_extraites[0]["string"],
														 show_image=False,
														 image_path=image_path)
		return [{"extracted": entites_extraites[0]['string'], "baseline": target_baseline}]
	else:
		extractions = []
		for entite in entites_extraites:
			target_baseline = utils.get_baseline_from_string(line=target_lines,
															 target_string=entite["string"],
															 show_image=False,
															 image_path=image_path)
			extractions.append({"extracted": entite['string'], "baseline": target_baseline})
		return extractions


def traiter_taille(chaine_cible: str) -> float:
	taille = utils.get_string_between_two_words(chaine_cible, "d'un mètre", "milimètres")
	if not taille or taille == "" or set(taille) == {" "}:
		taille = None
	else:
		try:
			taille = 1000 + int(taille)
		except ValueError:
			try:
				taille = 1000 + text2num(taille, "fr")
			except ValueError:
				taille = "UNK"
	return taille


def extraire_taille(lignes_description_physique,
					lignes_description_du_soldat_as_string):
	"""
		Cette fonction extrait l'information sur les cheveux de l'inculpé.
		:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
		:return: Un dictionnaire de la forme
		{
			"extracted": 1690,
			"matching": "un mètre 690 millimètres"
			"baseline": [[1416,3746],
						[2582,3735]],
			"prediction": "Taille d'un mètre 690 millimètres, cheveux Chatains, front ordinaire"
		}
		La baseline correspond à l'information extraite uniquement.
		"""
	chaine_taille = "Taille d'un mètre milimètres"
	ligne_taille, debug, index_taille = utils.match_line_by_substring(lignes_description_physique, chaine_taille,
																	  return_index=True)
	# On récupère la chaîne avant millimètres
	taille, matching_millimetre = utils.approximate_word_split(ligne_taille.prediction, "millimètres", sensibility=0.8,
															   return_word=True)
	# Puis la chaîne après mètre
	try:
		taille, matching_metre = utils.approximate_word_split(taille[0], "mètre", return_word=True)
	except TypeError:
		taille, matching_metre = utils.approximate_word_split(ligne_taille.prediction, "mètre", return_word=True)
	taille = taille[1]
	prediction_taille = ligne_taille.prediction
	prediction_taille = utils.nfc_normalize(prediction_taille)
	starting_index = prediction_taille.find(f"un {matching_metre}")
	try:
		ending_index = prediction_taille.find(f"{matching_millimetre}") + len(matching_millimetre) + 2
	except TypeError:
		ending_index = len(ligne_taille.prediction)
	matching_string = ligne_taille.prediction[starting_index:ending_index]
	specific_baseline = utils.get_baseline_from_string(ligne_taille, matching_string)
	if taille == "" or set(taille) == {" "}:
		taille = None
	else:
		try:
			taille = 1000 + int(taille)
		except ValueError:
			try:
				taille = 1000 + text2num(taille, "fr")
			except ValueError:
				taille = "UNK"
	if taille == "":
		taille = None
	matching_spans = utils.retrieve_substring_span(lignes_description_du_soldat_as_string, matching_string)
	return {"extracted": taille,
			"matching": matching_string,
			"matching_spans": matching_spans,
			"baseline": specific_baseline,
			"prediction": ligne_taille.prediction}


def extraire_matricule(lignes_description_physique,
					   lignes_description_du_soldat_as_string):
	dict_matricule = {}
	numero_matricule = "n^o m^le 00000"
	numero_matricule_2 = "numero matricule 00000"
	ligne_matricule, debug, index_matricule = utils.match_line_by_substring(lignes_description_physique,
																			numero_matricule, return_index=True)
	# Si l'index est 0, c'est qu'il a identifié la première ligne qui contient des chiffres.
	# Le numéro de matricule n'y est jamais indiqué.
	if index_matricule == 0:
		ligne_matricule, debug, index_matricule = utils.match_line_by_substring(lignes_description_physique,
																				numero_matricule_2, return_index=True)
		if index_matricule == 0:
			return None
	utils.log_print("Ligne trouvée.")
	# https://maxhalford.github.io/blog/fuzzy-regex-matching-in-python/
	regexp_matricule = r"[Nn]um[eé]ro matricule (\d+\.?\s{,3}\d+)|[Nn]?\^?o?\s?m\^?l?e? (\d+\.?\s{,3}\d+)|[nN]?\^?o?\s?m\^?l?e? [ao]u corps (\d+\.?\s{,3}\d+)"
	fuzzy_pattern = f'({regexp_matricule}){{e<=4}}'
	try:
		numero_matricule = regex.search(fuzzy_pattern, ligne_matricule.prediction.lower(),
										regex.BESTMATCH).group(1)
		numero = re.compile(r"(\d+\.?\s{,3}\d+)")
		numero_extrait = re.search(numero, numero_matricule).group(1)
	except AttributeError:
		numero_extrait = None
	if numero_matricule:
		baseline_matricule = utils.get_baseline_from_string(ligne_matricule,
															numero_matricule)
		dict_matricule["extracted"] = numero_extrait
		dict_matricule["matching"] = numero_matricule
		dict_matricule["prediction"] = ligne_matricule.prediction
		dict_matricule["baseline"] = baseline_matricule
		try:
			dict_matricule["matching_spans"] = [lignes_description_du_soldat_as_string.find(numero_extrait),
												lignes_description_du_soldat_as_string.find(numero_extrait) + len(
													numero_extrait)]
		except TypeError:
			dict_matricule["matching_spans"] = [lignes_description_du_soldat_as_string.find(numero_matricule),
												lignes_description_du_soldat_as_string.find(numero_matricule) + len(
													numero_matricule)]
	else:
		dict_matricule = {
			"extracted": None,
			"matching": None,
			"prediction": ligne_matricule.prediction
		}
	return dict_matricule


def extraire_age_soldat(lignes_identite_soldat: OCRRecord) -> dict:
	out_dict = {}
	lines_as_text = " ".join([line.prediction for line in lignes_identite_soldat])
	regexp_age = re.compile(r"(\d+) ans")
	age = re.search(regexp_age, lines_as_text)
	# Si on trouve déjà quelque chose
	if not age:
		regexp_age = re.compile(r"([^\s]+)\sans[\s,.]")
		age = re.search(regexp_age, lines_as_text)
	try:
		full_age = age.group()
	except AttributeError:
		return {"extracted": "UNK"}
	age_digits = re.search(regexp_age, lines_as_text).group(1)
	try:
		age_as_int = int(age_digits)
	except ValueError:
		age_as_int = age_digits
	corresponding_line, *_ = utils.match_line_by_substring(corresponding_lines=lignes_identite_soldat,
														   string_to_match=full_age)

	box_age = utils.get_baseline_from_string(line=corresponding_line,
											 target_string=full_age)
	out_dict = {"extracted": age_as_int,
				"baseline": box_age,
				"prediction": full_age}
	return out_dict


def extraire_cheveux(lignes_description_physique,
					 lignes_description_du_soldat_as_string):
	"""
	Cette fonction extrait l'information sur les cheveux de l'inculpé.
	:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
	:return: Un dictionnaire de la forme
	{
		"extracted": "Chatain-focie",
		"matching": "cheveux Chatain-focie"
		"baseline": [[1416,3746],
					[2582,3735]],
		"prediction": "Taille d'un mètre 670 millimètres, cheveux Chatain-focie, front moyen"
	}
	La baseline correspond à l'information extraite uniquement.
	"""
	chaine_cheveux = "cheveux"
	ligne_cheveux, debug, index_cheveux = utils.match_line_by_substring(lignes_description_physique, chaine_cheveux,
																		return_index=True)

	cheveux, matching_cheveux = utils.approximate_word_split(ligne_cheveux.prediction, "cheveux",
															 sensibility=0.7, return_word=True)
	front, matching_front = utils.approximate_word_split(ligne_cheveux.prediction, "front",
														 sensibility=0.7, return_word=True)

	starting_index = ligne_cheveux.prediction.find(matching_cheveux)

	# Dans certains cas l'information est sur une ligne isolée et le front apparaît sur une autre ligne
	try:
		ending_index = ligne_cheveux.prediction.find(f"{matching_front}") + len(matching_front)
	except TypeError:
		matching_front = ""
		ending_index = len(ligne_cheveux.prediction)
	matching_string = ligne_cheveux.prediction[starting_index:ending_index]
	specific_baseline = utils.get_baseline_from_string(ligne_cheveux, matching_string)
	extracted_cheveux = utils.strip_punctuation(
		matching_string.replace(matching_cheveux, "").replace(matching_front, ""))
	if extracted_cheveux == "":
		extracted_cheveux = None
		matching_spans = None
	else:
		matching_spans = utils.retrieve_substring_span(lignes_description_du_soldat_as_string, extracted_cheveux)
	return {"extracted": extracted_cheveux,
			"matching": matching_string.replace(matching_front, ""),
			"baseline": specific_baseline,
			"matching_spans": matching_spans,
			"prediction": ligne_cheveux.prediction}


def extraire_feature(entities_list,
					 lignes: OCRRecord,
					 feature: str,
					 image_path: str) -> dict:
	"""
	Cette fonction extrait une feature précise d'un résultat de NER, et retrouve la ligne de base
	qui contient
	:param entities_list:
	:param lignes:
	:param feature:
	:return: Un dictionnaire de la forme:
	'''
	{
		"extracted": feature,
		"baseline": target_baseline,
		"certainty": certainty
	}
	'''
	"""
	try:
		entite_et_baseline = extraire_entite_baseline(
			entities_list=entities_list,
			nom_entite=feature,
			target_lines=lignes,
			image_path=image_path
		)
	except IndexError:
		return {
		"extracted": None,
		"baseline": None,
		"certainty": None
	}
	if entite_et_baseline and len(entite_et_baseline) == 1:
		certainty = 0.8
	elif not entite_et_baseline:
		certainty = 0.5
	else:
		certainty = None

	if entite_et_baseline:
		extracted_feature, target_baseline_extracted_feature = (entite_et_baseline[0]["extracted"],
																entite_et_baseline[0]["baseline"])
	else:
		extracted_feature, target_baseline_extracted_feature = None, None
	return {
		"extracted": utils.clean_small_string(extracted_feature),
		"baseline": target_baseline_extracted_feature,
		"certainty": certainty
	}



def extraire_greffier(lignes_zone_magistrat: list, image_path, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le greffier à partir de la ligne complète:
	"M. Arnould, Off. d'Adm^n Greffier près ledit Conseil;"
	:param ligne_greffier: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes
	"""
	greffier_dict = {}
	ligne_greffier, _ = utils.match_line_by_substring(lignes_zone_magistrat, "pres ledit conseil")
	baseline = ligne_greffier.baseline
	prediction = ligne_greffier.prediction
	greffier, form_greffier = utils.check_word_in_sentence(prediction, "Greffier")
	if greffier is True:
		debut_de_chaine = prediction.split(form_greffier)[0]
	else:
		utils.log_print("Ligne mal reconnue")
		return {"greffier": None}

	# TODO: gérer le problème avec maréchal des logis

	commis, form_commis = utils.check_word_in_sentence(debut_de_chaine, target_word="commis", sensibility=0.7)
	if commis is True:
		nom_et_statut = debut_de_chaine.replace(form_commis, "").replace("M.", "")
	else:
		commis_abrege, form_commis_abrege = utils.check_word_in_sentence(debut_de_chaine, "c^is")
		if commis_abrege is True:
			nom_et_statut = debut_de_chaine.replace(form_commis_abrege, "").replace("M.", "")
		else:
			nom_et_statut = debut_de_chaine.replace("M.", "")
	nom_et_fonction_du_greffier = extraire_nom_et_fonction(nom_et_statut, pipeline=ner_pipeline)
	baseline = utils.get_baseline_from_string(line=ligne_greffier,
											  target_string=nom_et_fonction_du_greffier['persName'],
											  image_path=image_path)
	if commis is True:
		nom_et_fonction_du_greffier["commis"] = True
	else:
		nom_et_fonction_du_greffier["commis"] = False

	greffier_dict["extracted"] = nom_et_fonction_du_greffier
	greffier_dict["baseline"] = baseline
	greffier_dict["prediction"] = prediction

	return greffier_dict


def extraire_date_naissance(entity_dict, lignes:OCRRecord, image_path):
	"""
	Cette fonction extrait la date de naissance et les baselines
	:param entity_dict:
	:param lignes:
	:return: Le dictionnaire produit.
	"""
	date_naissance = extraire_feature(
		entity_dict,
		lignes,
		"date_naissance",
				image_path=image_path
	)

	if date_naissance["extracted"] is not None:
		date_naissance_corrigee = utils.correct_date(date_naissance["extracted"])
		date_naissance["normalized"] = date_naissance["extracted"]
		try:
			date_normalisee = date.process_date(date_naissance_corrigee)
		except TypeError:
			logging.error(f"Erreur d'extraction de la date de naissance {date_naissance_corrigee}.")
			date_naissance["extracted"] = None
			date_normalisee = date_naissance["extracted"]
		date_naissance["extracted"] = date_normalisee
	else:
		logging.error(f"Date de naissance non identifiée: {entity_dict}; {lignes.join_transcription()}")

	return date_naissance


def extraire_lieu_naissance(entity_dict, lignes, image_path, geoextractor):
	"""
	Cette fonction extrait le lieu de naissance et les baselines
	:param entity_dict:
	:param lignes:
	:return: Le dictionnaire produit.
	"""

	lieu_naissance = {}
	lieu_naissance["departement"] = extraire_feature(
		entity_dict,
		lignes,
		"departement_naissance",
		image_path=image_path
	)


	lieu_naissance["ville"] = extraire_feature(
		entity_dict,
		lignes,
		"ville_naissance",
		image_path=image_path
	)

	lieu_naissance["arrondissement"] = extraire_feature(
		entity_dict,
		lignes,
		"arrondissement_naissance",
		image_path=image_path
	)
	ville = lieu_naissance["ville"]["extracted"]
	arrondissement = lieu_naissance["arrondissement"]["extracted"]
	if arrondissement and "dudit" in arrondissement:
		lieu_naissance["arrondissement"]["extracted"] = ville
	departement = lieu_naissance["departement"]["extracted"]
	logger.info("Récupération des coordonnées du lieu de naissance.")
	if departement is None:
		logger.warning(f"Pas de lieu de naissance identifié. Texte: {lignes.join_transcription()}. Entités: {entity_dict}")
	result = geoextractor.database_retrieval(ville=ville, arrondissement=arrondissement, departement=departement)
	logger.info(f"Lieu de naissance: {result}")
	lieu_naissance["etranger"] = result["etranger"]
	lieu_naissance["hors_metropole"] = result["hors_metropole"]
	lieu_naissance["pays"] = result["pays"]
	lieu_naissance["coordonnées"] = {
		"lon": result["lon"],
		"lat": result["lat"]
	}
	try:
		lieu_naissance["departement"]["corrected"] = result["departement"]
	except (KeyError, TypeError):
		lieu_naissance["departement"]["corrected"] = None
	try:
		lieu_naissance["ville"]["nom_actuel"] = result["nom_actuel"]
		lieu_naissance["ville"]["nom_1999"] = result["nom_1999"]
		lieu_naissance["ville"]["nom_1801"] = result["nom_1801"]
	except KeyError:
		pass
	except TypeError:
		lieu_naissance["ville"] = None
		lieu_naissance["coordonnées"] = {
			"lon": result["lon"],
			"lat": result["lat"]
		}
		return lieu_naissance
	return lieu_naissance


def extraire_lieu_residence(entity_dict, lignes, geoextractor, image_path, lieu_naissance):
	"""
	Cette fonction extrait le lieu de naissance et les baselines
	:param entity_dict:
	:param lignes:
	:return: Le dictionnaire produit.
	"""

	lieu_residence = {}

	lieu_residence["departement"] = extraire_feature(
		entity_dict,
		lignes,
		"departement_residence",
		image_path=image_path
	)

	lieu_residence["ville"] = extraire_feature(
		entity_dict,
		lignes,
		"ville_residence",
		image_path=image_path
	)

	lieu_residence["arrondissement"] = extraire_feature(
		entity_dict,
		lignes,
		"arrondissement_residence",
		image_path=image_path
	)

	adresse = extraire_feature(
		entity_dict,
		lignes,
		"adresse_residence",
		image_path=image_path
	)

	ville = lieu_residence["ville"]["extracted"]
	arrondissement = lieu_residence["arrondissement"]["extracted"]
	if arrondissement and "dudit" in arrondissement:
		lieu_residence["arrondissement"]["extracted"] = ville

	if ville and utils.similarite_ratcliff(ville, "y domicilié") > .8:
		logger.info(f"La ville de résidence est la même que la ville de naissance ('{ville}').")
		return lieu_naissance
	departement = lieu_residence["departement"]["extracted"]

	# On considère que les cas où le département n'est pas indiqué correspondent aux cas où il est le même que
	# le département de naissance.
	logger.info("Récupération des coordonnées du lieu de résidence.")
	if departement is None:
		logger.info("Le département de résidence n'est pas identifié. On cherche le département de naissance.")
		departement = lieu_naissance["departement"]["extracted"]

	if departement is None:
		logger.warning(f"Pas de lieu de résidence identifié. Texte: {lignes.join_transcription()}. Entités: {entity_dict}")
	result = geoextractor.database_retrieval(ville=ville, arrondissement=arrondissement, departement=departement)
	logger.info(f"Lieu de résidence: {result}")
	try:
		lieu_residence["departement"]["corrected"] = result["departement"]
	except (TypeError, KeyError):
		lieu_residence["departement"]["corrected"] = None
	if result is None:
		lieu_residence["coordonnées"] = None
	else:
		try:
			lieu_residence["ville"]["nom_actuel"] = result["nom_actuel"]
			lieu_residence["ville"]["nom_1999"] = result["nom_1999"]
			lieu_residence["ville"]["nom_1801"] = result["nom_1801"]
		except KeyError:
			lieu_residence["coordonnées"] = None
		except TypeError:
			lieu_residence["coordonnées"] = None
	lieu_residence["etranger"] = result["etranger"]
	lieu_residence["hors_metropole"] = result["hors_metropole"]
	lieu_residence["pays"] = result["pays"]
	lieu_residence["coordonnées"] = {
		"lon": result["lon"],
		"lat": result["lat"]
	}
	if adresse["extracted"]:
		lieu_residence["adresse"] = adresse

	return lieu_residence


def extraire_sit_maritale(entity_dict, image_path, lignes):
	"""
	Cette fonction extrait la situation maritale
	:param entity_dict:
	:param lignes:
	:return: Le dictionnaire produit.
	"""

	situation_maritale = {}
	situation_courante = extraire_feature(
		entity_dict,
		lignes,
		"situation_maritale",
		image_path=image_path
	)

	enfants = extraire_feature(
		entity_dict,
		lignes,
		"enfants",
		image_path=image_path
	)
	if enfants["extracted"] is not None:
		try:
			nombre_enfants = utils.approximate_word_split(
				sentence=enfants["extracted"],
				word="enfants",
				sensibility=0.6
			)[0]
			if utils.check_substring_in_sentence(sentence=nombre_enfants,
												 target_substring="sans",
												 max_distance=1):
				enfants["predicted"] = enfants["extracted"]
				nombre_enfants = 0
			else:
				nombre_enfants = utils.clean_small_string(nombre_enfants)
				try:
					nombre_enfants = int(nombre_enfants)
				except ValueError:
					try:
						nombre_enfants = text2num(nombre_enfants, lang="fr")
					except ValueError:
						nombre_enfants = None
					# except PanicException:
					# 	nombre_enfants = None
					except:
						nombre_enfants = None
			enfants["extracted"] = nombre_enfants
		except TypeError:
			enfants["extracted"] = None

	situation_maritale["enfants"] = enfants

	if not situation_courante["extracted"]:
		return situation_courante
	situations = ["marié", "veuf", "célibataire"]
	situations_as_dict = {}
	for situation in situations:
		situations_as_dict[situation] = utils.similarite_ratcliff(situation_courante["extracted"].lower(), situation)
	reverse_dict = {score: label for label, score in situations_as_dict.items()}
	max_score = max(reverse_dict.keys())
	correct_label = reverse_dict[max_score]

	situation_courante["predicted"] = situation_courante["extracted"]
	situation_courante["extracted"] = correct_label
	situation_maritale["situation"] = situation_courante

	return situation_maritale


def extraire_nom_et_fonction(prediction: str, pipeline, debug: bool = False):
	"""
	Cette fonction extrait le nom du magistrat qui siège au procès. On considère que la phrase est composée
	de deux éléments: le nom et la fonction
	:param prediction: La ligne à traiter
	:param pipeline: la pipeline de NER
	:param debug:
	:return:
	"""
	result = pipeline(f"[2] {prediction}")
	persName = " ".join(item['word'] for item in result if item["entity_group"] == "nom_du_soldat")
	role = " ".join(item['word'] for item in result if item["entity_group"] == "rang")
	role = utils.strip_punctuation(role)
	persName = utils.strip_punctuation(persName)
	return {"persName": persName,
			"role": role,
			"certainty": 0}

def predire_genre(prenoms, dict_genre):
	liste_prenoms = list(dict_genre.keys())
	liste_genres = []
	for prenom in prenoms:
		matching, distance = similarity.find_closest_word_in_list(word_list=liste_prenoms, target_word=prenom, load_file=False)
		# On ne va prendre que des prénoms qu'on est sûrs de reconnaître. Plus de rappel, mais moins de précision.
		if distance > 1:
			continue
		else:
			current_genre = dict_genre[matching]
			liste_genres.append(current_genre)
	masculin = liste_genres.count(2)
	feminin = liste_genres.count(1)
	if masculin > feminin:
		return "M"
	# Dans le doute on conserve le genre masculin
	elif masculin == feminin:
		return "M"
	else:
		logger.info(f"La personne jugée semble être une femme: {prenoms}.")
		return "F"

def extraire_commissaire(lignes_zone_magistrat: list, image_path:str, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le commissaire à partir de la ligne complète:
	"M. Le Clerc, S.Eicmtu^e Slsttut de Commissaire du Gouvernement;"
	:param ligne_commissaire: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes: informations de nom, substitut, baseline, prediction kraken
	"""
	commissaire_dict = {}
	ligne_commissaire, _ = utils.match_line_by_substring(lignes_zone_magistrat,
														 "commissaire du gouvernement")

	prediction = ligne_commissaire.prediction
	commissaire, form_commissaire = utils.check_word_in_sentence(prediction, "commissaire")
	if commissaire is True:
		debut_de_chaine = prediction.split(form_commissaire)[0]
	else:
		utils.log_print("Ligne mal reconnue")
		return {"commissaire": None}

	substitut, form_substitut = utils.check_word_in_sentence(debut_de_chaine,
															 target_word="substitut",
															 sensibility=0.5)
	if substitut is True:
		nom_et_statut = debut_de_chaine.split(form_substitut)[0].replace("M.", "")
	else:
		# Parfois l'indication de substitut est mise en fin de ligne. On relance sur toute la ligne.
		substitut, form_substitut = utils.check_word_in_sentence(prediction,
																 target_word="substitut",
																 sensibility=0.5)
		nom_et_statut = debut_de_chaine.replace("M.", "")
	nom_et_fonction_du_commissaire = extraire_nom_et_fonction(nom_et_statut, pipeline=ner_pipeline)
	baseline = utils.get_baseline_from_string(line=ligne_commissaire,
											  target_string=nom_et_statut,
											  image_path=image_path)
	if substitut is True:
		nom_et_fonction_du_commissaire["substitut"] = True
	else:
		nom_et_fonction_du_commissaire["substitut"] = False
	commissaire_dict["extracted"] = nom_et_fonction_du_commissaire
	commissaire_dict["baseline"] = baseline
	commissaire_dict["prediction"] = prediction

	return commissaire_dict


def extraire_general(lignes_zone_magistrat: OCRRecord,
					 image_path):
	"""
	Cette fonction extrait les informations concernant le militaire gradé nommant les magistrats:
	"tous nommés par le (1) Général Commandant la 2^e Armée"
	:param ligne_grade: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes: informations de nom, substitut, baseline, prediction kraken
	"""
	ligne_grade, _ = utils.match_line_by_substring(lignes_zone_magistrat, "nommés par le (1) général")
	prediction = ligne_grade.prediction
	prediction = utils.nfc_normalize(prediction)
	grade, form_grade = utils.check_word_in_sentence(prediction, "général")
	if grade is True:
		grade_extrait = f"{form_grade} {prediction.split(form_grade)[1].strip()}"
	else:
		utils.log_print("Ligne mal reconnue")
		return {"grade": None}
	baseline = utils.get_baseline_from_string(line=ligne_grade,
											  target_string=grade_extrait,
											  image_path=image_path)
	nom_et_fonction_du_grade = {"extracted": grade_extrait}
	nom_et_fonction_du_grade["baseline"] = baseline
	nom_et_fonction_du_grade["prediction"] = prediction
	return nom_et_fonction_du_grade
