import src.utils.utils as utils
import re
from text_to_num import text2num
import regex

from src.utils.utils import OCRLine, OCRRecord


def extraction_geographique(lieu:str, dictionnaire_informations:dict, ner_pipeline):
	"""
	Cette fonction extrait et formatte des informations géographiques.
	:param lieu: la chaîne à interroger
	:param informations: le dictionnaire à nourir
	:param ner_pipeline: le moteur de NER
	:return:
	"""
	dictionnaire_informations["extracted"] = {}
	split_departement = utils.approximate_split(lieu, "département")
	if split_departement:
		dictionnaire_informations["extracted"]["departement"] = utils.full_clean_string(split_departement[-1])
		split_arrondissement_1 = utils.approximate_split(split_departement[0], "arrd^t", sensibility=0.5)
		split_arrondissement_2 = utils.approximate_split(split_departement[0], "arrondissement", sensibility=0.85)
	else:
		dictionnaire_informations["extracted"]["departement"] = None
		split_arrondissement_1 = utils.approximate_split(lieu, "arrd^t", sensibility=0.5)
		split_arrondissement_2 = utils.approximate_split(lieu, "arrondissement", sensibility=0.7)
	if split_arrondissement_1:
		split_arrondissement = split_arrondissement_1
		arrondissement = utils.full_clean_string(split_arrondissement_1[-1])
	elif split_arrondissement_2:
		split_arrondissement = split_arrondissement_2
		arrondissement = utils.full_clean_string(split_arrondissement_2[-1])
	else:
		arrondissement = None
	dictionnaire_informations["extracted"]["arrondissement"] = arrondissement

	try:
		if arrondissement:
			ville = split_arrondissement[0]
		else:
			ville = split_departement[0]
	except TypeError:
		# On considère que le premier lieu est un nom de ville.
		try:
			ville = [item['word'] for item in ner_pipeline(lieu) if item['entity_group'] == "LOC"][0]
		except IndexError:
			ville = None
	if ville is not None:
		ville = utils.full_clean_string(ville)
	print(ner_pipeline(lieu))
	ner_extractions = ner_pipeline(lieu)
	lieux = [item['word'] for item in ner_extractions if item['entity_group'] == "LOC"]
	dictionnaire_informations["extracted"]["NER"] = lieux
	dictionnaire_informations["extracted"]["ville"] = ville
	return dictionnaire_informations

def extraire_taille(lignes_description_physique):
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
	taille, matching_millimetre = utils.approximate_split(ligne_taille.prediction, "millimètres", sensibility=0.8,
														  return_word=True)
	# Puis la chaîne après mètre
	try:
		taille, matching_metre = utils.approximate_split(taille[0], "mètre", return_word=True)
	except TypeError:
		taille, matching_metre = utils.approximate_split(ligne_taille.prediction, "mètre", return_word=True)
	taille = taille[1]
	prediction_taille = ligne_taille.prediction
	prediction_taille = utils.nfc_normalize(prediction_taille)
	starting_index = prediction_taille.find(f"un {matching_metre}")
	try:
		ending_index = prediction_taille.find(f"{matching_millimetre}") + len(matching_millimetre)
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
	return {"extracted": taille,
			"matching": matching_string,
		   "baseline": specific_baseline,
		   "prediction": ligne_taille.prediction}

def extraire_matricule(lignes_description_physique):
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
	print("Ligne trouvée.")
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
	else:
		dict_matricule = {
			"extracted": None,
			"matching": None,
			"prediction": ligne_matricule.prediction
		}
	return dict_matricule

def extraire_cheveux(lignes_description_physique):
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

	cheveux, matching_cheveux = utils.approximate_split(ligne_cheveux.prediction, "cheveux",
														sensibility=0.7, return_word=True)
	front, matching_front = utils.approximate_split(ligne_cheveux.prediction, "front",
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
	extracted_cheveux = utils.strip_punctuation(matching_string.replace(matching_cheveux, "").replace(matching_front, ""))
	if extracted_cheveux == "":
		extracted_cheveux = None
	return {"extracted": extracted_cheveux,
			"matching":  matching_string.replace(matching_front, ""),
			"baseline": specific_baseline,
			"prediction": ligne_cheveux.prediction}

def extraire_front(lignes_description_physique):
	"""
	Cette fonction extrait l'information sur les cheveux de l'inculpé.
	:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
	:return: Un dictionnaire de la forme
	{
		"extracted": "ordinaire",
		"matching": "front ordinaire"
		"baseline": [[1416,3746],
					[2582,3735]],
		"prediction": "Taille d'un mètre 550 millimètres, cheveux châtain, front ordinaire"
	}
	La baseline correspond à l'information extraite uniquement.
	"""
	chaine_front = "front"
	ligne_front, debug, index_front = utils.match_line_by_substring(lignes_description_physique, chaine_front,
																	return_index=True)

	front, matching_front = utils.approximate_split(ligne_front.prediction, "front",
													sensibility=0.7, return_word=True)

	try:
		starting_index = ligne_front.prediction.find(matching_front)
	except TypeError:
		return {"extracted": None,
			"matching":  None,
			"baseline": ligne_front.baseline,
			"prediction": ligne_front.prediction}

	matching_string = ligne_front.prediction[starting_index:]
	specific_baseline = utils.get_baseline_from_string(ligne_front, matching_string)
	extracted_front = utils.strip_punctuation(matching_string.replace(matching_front, ""))
	if extracted_front == "":
		extracted_front = None
	return {"extracted": extracted_front,
			"matching":  matching_string,
			"baseline": specific_baseline,
			"prediction": ligne_front.prediction}


def extraire_visage(lignes_description_physique):
	"""
	Cette fonction extrait l'information sur le visage de l'inculpé.
	:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
	:return: Un dictionnaire de la forme
	{
		"extracted": "ordinaire",
		"matching": "front ordinaire"
		"baseline": [[1416,3746],
					[2582,3735]],
		"prediction": "Taille d'un mètre 550 millimètres, cheveux châtain, front ordinaire"
	}
	La baseline correspond à l'information extraite uniquement.
	"""
	chaine_visage = "visage"
	ligne_visage, debug, index_visage = utils.match_line_by_substring(lignes_description_physique, chaine_visage,
																	  return_index=True)

	visage, matching_visage = utils.approximate_split(ligne_visage.prediction, "visage",
													sensibility=0.8, return_word=True)

	try:
		starting_index = ligne_visage.prediction.find(matching_visage)
	except TypeError:
		return {"extracted": None,
				"prediction": ligne_visage.prediction}

	matching_string = ligne_visage.prediction[starting_index:]
	specific_baseline = utils.get_baseline_from_string(ligne_visage, matching_string)
	extracted_visage = utils.strip_punctuation(matching_string.replace(matching_visage, ""))
	if extracted_visage == "":
		extracted_visage = None
	return {"extracted": extracted_visage,
			"matching":  matching_string,
			"baseline": specific_baseline,
			"prediction": ligne_visage.prediction}


def extraire_yeux(lignes_description_physique):
	"""
	Cette fonction extrait l'information sur les yeux de l'inculpé.
	:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
	:return: Un dictionnaire de la forme
	{
		"extracted": "gris bleux",
		"matching": "yeux gris bleux, "
		"baseline": [[1416,3746],
					[2582,3735]],
		"prediction": "yeux gris bleux, nez ordinoire, visage lonq et plein"
	}
	La baseline correspond à l'information extraite uniquement.
	"""
	chaine_yeux = "yeux"
	ligne_yeux, debug, index_yeux = utils.match_line_by_substring(lignes_description_physique, chaine_yeux,
																  return_index=True)

	yeux, matching_yeux = utils.approximate_split(ligne_yeux.prediction, "yeux",
													sensibility=0.8, return_word=True)
	nez, matching_nez = utils.approximate_split(ligne_yeux.prediction, "nez",
													sensibility=0.8, return_word=True)

	try:
		starting_index = ligne_yeux.prediction.find(matching_yeux)
	except TypeError:
		return {"extracted": None,
				"matching":  None,
				"baseline": ligne_yeux.baseline,
				"prediction": ligne_yeux.prediction}


	# Dans certains cas l'information est sur une ligne isolée et le yeux apparaît sur une autre ligne
	try:
		ending_index = ligne_yeux.prediction.find(f"{matching_nez}")
	except TypeError:
		matching_yeux = ""
		ending_index = len(ligne_yeux.prediction)

	matching_string = ligne_yeux.prediction[starting_index:ending_index]
	specific_baseline = utils.get_baseline_from_string(ligne_yeux, matching_string)
	extracted_yeux = utils.strip_punctuation(matching_string.replace(matching_yeux, ""))
	if extracted_yeux == "":
		extracted_yeux = None
	return {"extracted": extracted_yeux,
			"matching":  matching_string,
			"baseline": specific_baseline,
			"prediction": ligne_yeux.prediction}

def extraire_affectation_soldat(lignes_description_physique):
	derniere_ligne = lignes_description_physique[-1]
	if utils.check_word_in_sentence(derniere_ligne.prediction,
									target_word=['Inculpé', 'Accusé'],
									sensibility=0.6,
									debug=False)[0] is True:
		ligne_affectation = None
	elif len(derniere_ligne.prediction) < 10:
		ligne_affectation = None
	else:
		ligne_affectation = derniere_ligne.prediction
	return {"prediction": ligne_affectation,
			"extracted": ligne_affectation}

def extraire_marques_particulieres(lignes_description_physique, matricule) -> dict:
	"""
		Cette fonction extrait l'information sur les marques particulières du soldat
		:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
		:return: Un dictionnaire de la forme
		{
			"extracted": "RàS",
			"baseline": [[1416,3746],
						[2582,3735]],
			"prediction": "Renseignements physionomiques complémentaires:  Numero matricule: 17.615."
		}
		La baseline correspond à l'information extraite uniquement.
		"""
	chaine_marques = "Marques particulières"
	clean_regexp = re.compile("^\s?:?\s?")
	ligne_marques, debug, index_marques = utils.match_line_by_substring(lignes_description_physique, chaine_marques,
																		return_index=True)
	print(ligne_marques.prediction)
	try:
		particulieres, matching_particulieres = utils.approximate_split(ligne_marques.prediction,
																		  "particulières",
																		  sensibility=0.9,
																		  return_word=True)
	except TypeError:
		return {"extracted": None,
				"baseline": ligne_marques.baseline,
				"prediction": ligne_marques.prediction}

	try:
		corresp_string = particulieres[-1]
	except TypeError:
		return {"extracted": None,
				"baseline": ligne_marques.baseline,
				"prediction": ligne_marques.prediction}
	clean = re.sub(clean_regexp, "", corresp_string)

	# On vérifie si ce n'est pas l'information de matricule qui est indiqué (comme souvent), et on l'enlève
	try:
		clean = clean.lower().replace(matricule["matching"], "")
	except TypeError:
		pass
	clean = clean.strip()
	clean = utils.clean_small_string(clean)
	if clean == "":
		clean = "RàS"
	elif utils.check_neant(clean) is True:
		clean = "RàS"

	return {"extracted": clean,
			"baseline": ligne_marques.baseline,
			"prediction": ligne_marques.prediction}

def extraire_renseignements_complementaires(lignes_description_physique, matricule):
	"""
		Cette fonction extrait l'information sur les marques particulières
		:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
		:return: Un dictionnaire de la forme
		{
			"extracted": "RàS",
			"baseline": [[1416,3746],
						[2582,3735]],
			"prediction": "Renseignements physionomiques complémentaires:  Numero matricule: 17.615."
		}
		La baseline correspond à l'information extraite uniquement.
		"""
	chaine_renseignements = "Renseignements physionomiques complémentaires"
	clean_regexp = re.compile("^\s?:?\s?")

	ligne_renseignements, debug, index_renseignements = utils.match_line_by_substring(lignes_description_physique, chaine_renseignements,
																					  return_index=True)
	print(ligne_renseignements.prediction)
	try:
		complementaire, matching_complementaire = utils.approximate_split(ligne_renseignements.prediction,
																		  "complémentaires",
																		  sensibility=0.9,
																		  return_word=True)
	except TypeError:
		return {"extracted": None,
				"baseline": ligne_renseignements.baseline,
				"prediction": ligne_renseignements.prediction}

	try:
		corresp_string = complementaire[-1]
	except TypeError:
		return {"extracted": None,
				"baseline": ligne_renseignements.baseline,
				"prediction": ligne_renseignements.prediction}
	clean = re.sub(clean_regexp, "", corresp_string)

	# On vérifie si ce n'est pas l'information de matricule qui est indiqué (comme souvent), et on l'enlève
	try:
		clean = clean.lower().replace(matricule["matching"], "")
	except TypeError:
		pass
	clean = clean.strip()
	clean = utils.clean_small_string(clean)
	if clean == "":
		clean = "RàS"
	elif utils.check_neant(clean) is True:
		clean = "RàS"

	return {"extracted": clean,
			"baseline": ligne_renseignements.baseline,
			"prediction": ligne_renseignements.prediction}

def extraire_nez(lignes_description_physique):
	"""
	Cette fonction extrait l'information sur le nez de l'inculpé.
	:param lignes_description_physique: La liste de lignes sous la forme d'un dictionnaire {prediction, baseline, cuts}
	:return: Un dictionnaire de la forme
	{
		"extracted": "grand",
		"matching": "nez grand "
		"baseline": [[1416,3746],
					[2582,3735]],
		"prediction": "yeux noirs, nez grand visage ovale"
	}
	La baseline correspond à l'information extraite uniquement.
	"""
	chaine_nez = "nez"
	ligne_nez, debug, index_nez = utils.match_line_by_substring(lignes_description_physique, chaine_nez,
																return_index=True)

	nez, matching_nez = utils.approximate_split(ligne_nez.prediction, "nez",
													sensibility=0.8, return_word=True)
	visage, matching_visage = utils.approximate_split(ligne_nez.prediction, "visage",
													sensibility=0.8, return_word=True)

	try:
		starting_index = ligne_nez.prediction.find(matching_nez)
	except TypeError:
		return {"extracted": None,
				"matching": None,
				"baseline": ligne_nez.baseline,
				"prediction": ligne_nez.prediction}
	# Dans certains cas l'information est sur une ligne isolée et le nez apparaît sur une autre ligne
	try:
		ending_index = ligne_nez.prediction.find(f"{matching_visage}")
	except TypeError:
		matching_nez = ""
		ending_index = len(ligne_nez.prediction)

	matching_string = ligne_nez.prediction[starting_index:ending_index]
	specific_baseline = utils.get_baseline_from_string(ligne_nez, matching_string)
	extracted_nez = utils.strip_punctuation(matching_string.replace(matching_nez, ""))
	if extracted_nez == "":
		extracted_nez = None
	return {"extracted": extracted_nez,
			"matching":  matching_string,
			"baseline": specific_baseline,
			"prediction": ligne_nez.prediction}



def extraire_greffier(lignes_zone_magistrat:list, ner_pipeline):
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
	ligne_greffier = ligne_greffier.prediction
	greffier, form_greffier = utils.check_word_in_sentence(ligne_greffier, "Greffier")
	if greffier is True:
		debut_de_chaine = ligne_greffier.split(form_greffier)[0]
	else:
		print("Ligne mal reconnue")
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
	if commis is True:
		nom_et_fonction_du_greffier["commis"] = True
	else:
		nom_et_fonction_du_greffier["commis"] = False

	greffier_dict["extracted"] = nom_et_fonction_du_greffier
	greffier_dict["baseline"] = baseline
	greffier_dict["prediction"] = ligne_greffier

	return greffier_dict



def extraire_nom_et_fonction(prediction:str, pipeline, debug:bool=False):
	"""
	Cette fonction extrait le nom du magistrat qui siège au procès. On considère que la phrase est composée
	de deux éléments: le nom et la fonction
	:param prediction: La ligne à traiter
	:param pipeline: la pipeline de NER
	:param debug:
	:return:
	"""
	# result = pipeline(prediction.lower())
	result = pipeline(prediction)
	if debug:
		print("---")
		print(prediction)
		print(result)
	try:
		persName_NER = prediction[result[0]['start']: result[0]['end']].strip() if result[0]["entity_group"] == "PER" else None
	except IndexError:
		print(f"La phrase suivante: {prediction} n'a pas mené à reconnaissance d'entité. "
			  f"Une erreur en amont (zonage, OCR) est possible.")
		return {"persName": "UNK",
			"role": "UNK",
			"certainty": 0}

	# La fonction s'extrait après le nom identifié
	role_NER = prediction[result[0]['end']:] if result[0]["entity_group"] == "PER" else None

	# Extraction simple: le premier mot. Pour les noms à particule c'est plus compliqué: aller chercher la virgule?
	match_first_word = re.search(re.compile(r"[^\s,.]+"), prediction)
	spans = match_first_word.span()
	homemade_NER = prediction[spans[0]: spans[1]].strip()
	homemade_role = prediction[spans[1]:]
	if persName_NER == homemade_NER:
		certainty = 1
		persName = persName_NER
		role = role_NER
	elif persName_NER is not None:
		certainty = 0.5
		persName = persName_NER
		role = role_NER
	else:
		certainty = 0.3
		persName = homemade_NER
		role = homemade_role
	role = utils.strip_punctuation(role)
	persName = utils.strip_punctuation(persName)
	return {"persName": persName,
			"role": role,
			"certainty": certainty}


def extraire_situation_maritale(string) -> tuple:
	"""
	Cette fonction permet d'extraire la situation maritale
	:param string:
	:return: Un ensemble de données:
		- la vérification du célibat dans la chaîne de caractère donnée,
		- l'inférence sur le célibat par rapport aux autre données
		- le token du célibat trouvé
		- l'extraction de l'information maritale
		- le token marital correspondant,
		- le nombre d'enfants trouvé.
	"""
	token_marie, token_celibataire, token_enfants, token_veuf = None, None, None, None

	check_veuf, token_veuf = utils.check_word_in_sentence(string, "veuf", sensibility=0.90)

	check_celibataire, token_celibataire = utils.check_word_in_sentence(string,
																		"célibataire", sensibility=0.85)
	if check_celibataire is False:
		marie, token_marie = utils.check_word_in_sentence(string, "marié", sensibility=0.85)
		if marie is True:
			check_enfants, token_enfants = utils.check_word_in_sentence(string, "enfant", sensibility=0.85)
			if check_enfants is True:
				nombre_enfants = utils.split_before_keep_delimiter(string, token_enfants)[0].split(token_marie)[-1]
				nombre_enfants = utils.strip_punctuation(nombre_enfants)
				try:
					nombre_enfants = utils.number_dict[utils.correct_based_on_list(nombre_enfants, list(utils.number_dict.keys()))]
				except KeyError:
					nombre_enfants = nombre_enfants
			else:
				nombre_enfants = None
		else:
			# On peut éventuellement (???) avoir une indication d'enfants sans mariage, OU rater l'information du mariage
			check_enfants, token_enfants = utils.check_word_in_sentence(string, "enfant", sensibility=0.85)
			if check_enfants is True:
				print(token_enfants)
				tokens_enfants_clean = token_enfants.replace('(', '').replace(')', '')
				nombre_enfants = utils.strip_punctuation(tokens_enfants_clean)
				try:
					nombre_enfants = utils.number_dict[utils.correct_based_on_list(nombre_enfants, list(utils.number_dict.keys()))]
				except KeyError:
					nombre_enfants = tokens_enfants_clean
				regexp = re.compile(rf"(sans|\d+)\s*{nombre_enfants}")
				try:
					nombre_enfants = re.search(regexp, string).group(1)
				except AttributeError:
					nombre_enfants = "Unknown"
			else:
				nombre_enfants = None
	else:
		marie = False
		nombre_enfants = None
	if (marie, nombre_enfants) == (False, None):
		celibataire = True
	else:
		celibataire = False
	if check_veuf is True:
		marie = True
	return check_veuf, token_veuf, check_celibataire, celibataire, token_celibataire, marie, token_marie, nombre_enfants


def extraire_commissaire(lignes_zone_magistrat:list, ner_pipeline):
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
	baseline = ligne_commissaire.baseline
	ligne_commissaire = ligne_commissaire.prediction
	commissaire, form_commissaire = utils.check_word_in_sentence(ligne_commissaire, "commissaire")
	if commissaire is True:
		debut_de_chaine = ligne_commissaire.split(form_commissaire)[0]
	else:
		print("Ligne mal reconnue")
		return {"commissaire": None}

	substitut, form_substitut = utils.check_word_in_sentence(debut_de_chaine,
															 target_word="substitut",
															 sensibility=0.5)
	if substitut is True:
		nom_et_statut = debut_de_chaine.split(form_substitut)[0].replace("M.", "")
	else:
		# Parfois l'indication de substitut est mise en fin de ligne. On relance sur toute la ligne.
		substitut, form_substitut = utils.check_word_in_sentence(ligne_commissaire,
																 target_word="substitut",
																 sensibility=0.5)
		nom_et_statut = debut_de_chaine.replace("M.", "")
	nom_et_fonction_du_commissaire = extraire_nom_et_fonction(nom_et_statut, pipeline=ner_pipeline)
	if substitut is True:
		nom_et_fonction_du_commissaire["substitut"] = True
	else:
		nom_et_fonction_du_commissaire["substitut"] = False
	commissaire_dict["extracted"] = nom_et_fonction_du_commissaire
	commissaire_dict["baseline"] = baseline
	commissaire_dict["prediction"] = ligne_commissaire

	return commissaire_dict


def extraire_general(lignes_zone_magistrat:OCRRecord, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le militaire gradé nommant les magistrats:
	"tous nommés par le (1) Général Commandant la 2^e Armée"
	:param ligne_grade: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes: informations de nom, substitut, baseline, prediction kraken
	"""
	ligne_grade, _ = utils.match_line_by_substring(lignes_zone_magistrat, "nommés par le (1) général")
	baseline = ligne_grade.baseline
	ligne_grade = ligne_grade.prediction
	ligne_grade = utils.nfc_normalize(ligne_grade)
	grade, form_grade = utils.check_word_in_sentence(ligne_grade, "général")
	if grade is True:
		grade_extrait = f"{form_grade} {ligne_grade.split(form_grade)[1].strip()}"
	else:
		print("Ligne mal reconnue")
		return {"grade": None}

	nom_et_fonction_du_grade = {"extracted": grade_extrait}
	nom_et_fonction_du_grade["baseline"] = baseline
	nom_et_fonction_du_grade["prediction"] = ligne_grade
	return nom_et_fonction_du_grade

