import src.utils.utils as utils
import re

def extraire_greffier(lignes_zone_magistrat:list, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le greffier à partir de la ligne complète:
	"M. Arnould, Off. d'Adm^n Greffier près ledit Conseil;"
	:param ligne_greffier: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes
	"""
	ligne_greffier, _ = utils.check_substring_in_line(lignes_zone_magistrat, "pres ledit conseil")
	baseline = ligne_greffier['baseline']
	ligne_greffier = ligne_greffier['prediction']
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
	nom_et_fonction_du_greffier["baseline"] = baseline
	nom_et_fonction_du_greffier["prediction_kraken"] = ligne_greffier

	return nom_et_fonction_du_greffier



def extraire_nom_et_fonction(prediction:str, pipeline, debug:bool=False):
	"""
	Cette fonction extrait le nom du magistrat qui siège au procès. On considère que la phrase est composée
	de deux éléments: le nom et la fonction
	:param prediction: La ligne à traiter
	:param pipeline: la pipeline de NER
	:param debug:
	:return:
	"""
	result = pipeline(prediction.lower())
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
	ligne_commissaire, _ = utils.check_substring_in_line(lignes_zone_magistrat,
														 "commissaire du gouvernement")
	baseline = ligne_commissaire['baseline']
	ligne_commissaire = ligne_commissaire['prediction']
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
	nom_et_fonction_du_commissaire["baseline"] = baseline
	nom_et_fonction_du_commissaire["prediction_kraken"] = ligne_commissaire

	return nom_et_fonction_du_commissaire


def extraire_general(lignes_zone_magistrat:list, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le militaire gradé nommant les magistrats:
	"tous nommés par le (1) Général Commandant la 2^e Armée"
	:param ligne_grade: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes: informations de nom, substitut, baseline, prediction kraken
	"""
	ligne_grade, _ = utils.check_substring_in_line(lignes_zone_magistrat, "nommés par le (1) général")
	baseline = ligne_grade['baseline']
	ligne_grade = ligne_grade['prediction']
	ligne_grade = utils.nfc_normalize(ligne_grade)
	grade, form_grade = utils.check_word_in_sentence(ligne_grade, "général")
	if grade is True:
		grade_extrait = f"{form_grade} {ligne_grade.split(form_grade)[1].strip()}"
	else:
		print("Ligne mal reconnue")
		return {"grade": None}

	nom_et_fonction_du_grade = {"rang": grade_extrait}
	nom_et_fonction_du_grade["baseline"] = baseline
	nom_et_fonction_du_grade["prediction_kraken"] = ligne_grade
	return nom_et_fonction_du_grade

