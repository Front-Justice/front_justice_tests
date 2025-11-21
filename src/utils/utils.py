import json
import pickle
import unicodedata
from datetime import datetime
import dateparser
import PIL.Image as Image
import re

import party.fusion
from Levenshtein import distance
from difflib import SequenceMatcher
from shapely.geometry import Polygon
from collections import namedtuple
import time
from yaspin import yaspin
import dateutil


def load(path):
	return Image.open(path)


def split_after_keep_delimiter(target_string: str, delimiter: str) -> list:
	"""
		Cette fonction coupe une phrase selon un délimiteur qui est une expression régulière, et garde le délimiteur.
		La coupe se fait après le délimiteur.
		:param target_string: la chaîne à couper
		:param delimiter: le délimiteur sous la forme d'une chaîne de caractères qui sera compilée après normalisation
		:return: la liste voulue
		"""
	# On commence par normaliser les chaînes de caractères
	delimiter = normalize_string(delimiter)
	delimiter_as_regexp = re.compile(delimiter)
	target_string = normalize_string(target_string)
	out_split = []
	results = re.finditer(delimiter_as_regexp, target_string)
	delimiters = [0]
	for result in results:
		delimiters.append(result.span()[1])
	delimiters.append(len(target_string))

	for position in range(len(delimiters[1:])):
		pos = position + 1
		out_split.append(target_string[delimiters[pos - 1]: delimiters[pos]])
	return out_split


def normalize_string(input_string: str) -> str:
	"""
	Cette fonction applique une normalisation unicode NFC à la chaîne de caractères voulue.
	:param input_string:
	:return:
	"""
	return unicodedata.normalize('NFC', input_string)


def split_before_keep_delimiter(target_string: str, delimiter: str) -> list:
	"""
	Cette fonction coupe une phrase selon un délimiteur qui est une expression régulière, et garde le délimiteur.
	La coupe se fait avant le délimiteur.
	:param target_string: la chaîne à couper
	:param delimiter: le délimiteur sous la forme d'une chaîne de caractères qui sera compilée après normalisation
	:return: la liste voulue
	"""
	# On commence par normaliser les chaînes de caractères
	delimiter = normalize_string(delimiter)
	delimiter_as_regexp = re.compile(delimiter)
	target_string = normalize_string(target_string)

	out_split = []
	results = re.finditer(delimiter_as_regexp, target_string)
	delimiters = [0]
	for result in results:
		delimiters.append(result.span()[0])
	delimiters.append(len(target_string))

	for position in range(len(delimiters[1:])):
		pos = position + 1
		out_split.append(target_string[delimiters[pos - 1]: delimiters[pos]].strip())
	return out_split


def produce_line_function(baseline):
	x_1, y_1, x_2, y_2 = baseline
	a = (y_2 - y_1) / (x_2 - x_1)
	b = y_1 - a * x_1
	return a, b


def point_in_box(coord, box_coord):
	x, y = coord
	if box_coord.xmin <= x <= box_coord.xmax and box_coord.ymin <= y <= box_coord.ymax:
		return True
	else:
		return False


# class Line:
# 	def __init__(self, line):
# 		self.baseline = line['baseline']
# 		self.prediction = line['prediction']
# 		self.cuts = line['cuts']
#
# class OCRannotation:
# 	def __init__(self, annotation):
# 		self.annotation = []
# 		for line in annotation:
# 			self.line = Line(annotation)
# 			self.annotation.append(self.baseline,
# 								   self.prediction,
# 								   self.cuts)

def vertical_order_lines(lines: list[dict]) -> list[dict]:
	"""
	Cette fonction trie les lignes de façon verticale (de haut en bas). Elle suppose un filtre
	préalable des lignes au sein des zones pour être efficace
	:param lines: la liste de dictionnaires (baseline, prediction, cuts)
	:return: la liste ordonnée
	"""
	sorted_list = sorted(lines, key=lambda x: x["baseline"][0][1])
	return sorted_list


def vertical_order_zones(annotations: list[dict]) -> list[dict]:
	"""
	Fonction pour ordonner les zones verticalement (du plus haut au plus bas).
	On ordonne par la deuxième coordonnée de la boîte (y1)
	:param annotations: Les annotations sous la forme d'une liste de dictionnaire:
	[
		{
			'label': 'ligne',
			'coordinates': [[2713, 2242], [3033, 2857]]
		},
		{
			'label': 'ligne',
			'coordinates': [[213, 2236], [2745, 2404]]
		}
	]
	:return: Les mêmes annotations ordonnées
	"""
	sorted_list = sorted(annotations, key=lambda x: x["coordinates"][0][1])
	return sorted_list


def rectanglify(coords):
	return


def horizontal_order_zones(annotations):
	"""
	Fonction pour ordonner les zones horizontalement (de gauche à droite).
	On ordonne par la première coordonnée de la boîte (x1)
	:param annotations: Les annotations sous la forme d'une liste de dictionnaire:
	[
		{
			'label': 'ligne',
			'coordinates': [[2713, 2242], [3033, 2857]]
		},
		{
			'label': 'ligne',
			'coordinates': [[213, 2236], [2745, 2404]]
		}
	]
	:return: Les mêmes annotations ordonnées
	"""
	sorted_list = sorted(annotations, key=lambda x: x["coordinates"][0][0])
	return sorted_list


def check_if_overlap(target, source):  # returns None if rectangles don't intersect
	dx = min(target.xmax, source.xmax) - max(target.xmin, source.xmin)
	dy = min(target.ymax, source.ymax) - max(target.ymin, source.ymin)
	area_source = round((source.xmax - source.xmin) * (source.ymax - source.ymin))
	if (dx >= 0) and (dy >= 0):
		overlap_area = round(dx * dy)
		ratio = round(overlap_area / area_source, 2)
		return ratio
	else:
		return None


def clean_forename(name):
	name = name.replace(",", "").strip()
	return name

def normalize_string_and_strip_spaces(string:str) -> str:
	"""
	Cette fonction neutralise la casse et supprime les espace de début et fin de chaîne
	:param string: le texte à normaliser
	:return: le texte normalisé
	"""
	return string.lower().strip()


def extraction_prenom_du_soldat(prediction, nom_du_soldat, pipeline):
	"""
	Cette fonction utilise un NER pour extraire le prénom du soldat
	:param prediction: La chaîne de caractère
	:param nom_du_soldat:
	:param pipeline:
	:param debug:
	:return:
	"""
	# On nettoie pour faciliter le NER
	result = pipeline(prediction.lower().replace(",", " "))
	words = [prediction[entity['start']:entity['end']] for entity in result]
	try:
		# Si on a un nom, on prend l'entité qui le contient,
		correct_entity = next(entity for entity in words if nom_du_soldat.lower() in entity.lower())
		forename = correct_entity.split(nom_du_soldat)[1].strip()
		certainty = 0.8
	except StopIteration:
		# Si le nom est mal reconnu, on considère que l'entité nommée est la première de la ligne
		correct_entity = words[0]
		forename = correct_entity
		certainty = 0.5
	forename = clean_forename(forename)
	return forename, certainty


def match_lines_in_zones(ocr_prediction: list[dict], zone_as_rectangle: namedtuple, intersect_ratio=0.5):
	"""
	Cette fonction identifie toutes les lignes qui traversent une boîte
	:param ocr_prediction: un objet de classe OCRPrediction. les lignes comme une liste de dictionnaires (baseline, prediction, cuts)
	:param zone_as_rectangle: la boîte
	:param intersect_ratio: la proportion minimale de la ligne comprise dans la boîte
	:return: une liste avec les lignes filtrées
	"""
	corresponding_lines = []
	for idx, line in enumerate(ocr_prediction):
		baseline = line['baseline']

		# Si la ligne de base comprend plus d'un point, on simplifie en prenant les extrémités
		converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
		is_in_box = check_if_line_in_box(box_coord=zone_as_rectangle,
										 baseline=converted_baseline,
										 intersect_ratio=intersect_ratio)
		if is_in_box is True:
			corresponding_lines.append(line)
	return corresponding_lines

def strip_punctuation(string:str, debug=False) -> str:
	"""
	Cette fonction supprime la ponctuation en début et fin de chaîne
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	orig_string = string
	punctuation = "[\(\),;.!?\-:]"
	expression = "^"+ punctuation + "\s{0,}|\s{0,}"+ punctuation + "$"
	punct_regexp = re.compile(expression)
	string = string.strip()
	string = re.sub(punct_regexp, "", string)
	if debug:
		print(f"|{orig_string}| -> |{string}|")
	return string

def extraire_greffier(lignes_zone_magistrat:list, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le greffier à partir de la ligne complète:
	"M. Arnould, Off. d'Adm^n Greffier près ledit Conseil;"
	:param ligne_greffier: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes
	"""
	ligne_greffier = match_line_by_similarity(lignes_zone_magistrat, "pres ledit conseil")
	baseline = ligne_greffier['baseline']
	ligne_greffier = ligne_greffier['prediction']
	print(ligne_greffier)
	greffier, form_greffier = check_word_in_sentence(ligne_greffier, "Greffier")
	if greffier is True:
		debut_de_chaine = ligne_greffier.split(form_greffier)[0]
	else:
		print("Ligne mal reconnue")
		return {"greffier": None}

	# TODO: gérer le problème avec maréchal des logis
	commis, form_commis = check_word_in_sentence(debut_de_chaine, target_word="commis", sensibility=0.7)
	if commis is True:
		nom_et_statut = debut_de_chaine.replace(form_commis, "").replace("M.", "")
	else:
		commis_abrege, form_commis_abrege = check_word_in_sentence(debut_de_chaine, "c^is")
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


def extraire_commissaire(lignes_zone_magistrat:dict, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le commissaire à partir de la ligne complète:
	"M. Le Clerc, S.Eicmtu^e Slsttut de Commissaire du Gouvernement;"
	:param ligne_commissaire: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes: informations de nom, substitut, baseline, prediction kraken
	"""
	ligne_commissaire = match_line_by_similarity(lignes_zone_magistrat, "commissaire du gouvernement")
	baseline = ligne_commissaire['baseline']
	ligne_commissaire = ligne_commissaire['prediction']
	print(ligne_commissaire)
	commissaire, form_commissaire = check_word_in_sentence(ligne_commissaire, "commissaire")
	if commissaire is True:
		debut_de_chaine = ligne_commissaire.split(form_commissaire)[0]
	else:
		print("Ligne mal reconnue")
		return {"commissaire": None}

	substitut, form_substitut = check_word_in_sentence(debut_de_chaine, target_word="substitut", sensibility=0.5)
	if substitut is True:
		nom_et_statut = debut_de_chaine.split(form_substitut)[0].replace("M.", "")
	else:
		# Parfois l'indication de substitut est mise en fin de ligne. On relance sur toute la ligne.
		substitut, form_substitut = check_word_in_sentence(ligne_commissaire, target_word="substitut", sensibility=0.5)
		nom_et_statut = debut_de_chaine.replace("M.", "")
	nom_et_fonction_du_commissaire = extraire_nom_et_fonction(nom_et_statut, pipeline=ner_pipeline)
	if substitut is True:
		nom_et_fonction_du_commissaire["substitut"] = True
	else:
		nom_et_fonction_du_commissaire["substitut"] = False
	nom_et_fonction_du_commissaire["baseline"] = baseline
	nom_et_fonction_du_commissaire["prediction_kraken"] = ligne_commissaire

	return nom_et_fonction_du_commissaire


def extraire_general(lignes_zone_magistrat:dict, ner_pipeline):
	"""
	Cette fonction extrait les informations concernant le militaire gradé nommant les magistrats:
	"tous nommés par le (1) Général Commandant la 2^e Armée"
	:param ligne_grade: la ligne
	:param ner_pipeline:
	:return: un dictionnaire contenant les informations importantes: informations de nom, substitut, baseline, prediction kraken
	"""
	ligne_grade = match_line_by_similarity(lignes_zone_magistrat, "nommés par le (1) général")
	baseline = ligne_grade['baseline']
	ligne_grade = ligne_grade['prediction']
	grade, form_grade = check_word_in_sentence(ligne_grade, "général")
	if grade is True:
		grade_extrait = f"{form_grade} {ligne_grade.split(form_grade)[1].strip()}"
	else:
		print("Ligne mal reconnue")
		return {"grade": None}

	nom_et_fonction_du_grade = {"fonction": grade_extrait}
	nom_et_fonction_du_grade["baseline"] = baseline
	nom_et_fonction_du_grade["prediction_kraken"] = ligne_grade
	return nom_et_fonction_du_grade


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
	role = strip_punctuation(role)
	persName = strip_punctuation(persName)
	return {"persName": persName,
			"role": role,
			"certainty": certainty}


def extract_string_from_cuts(box: list[list[int]], line: dict) -> str:
	"""
	Cette fonction extrait les caractères compris dans une boîte par la comparaison
	entre cette boîte et les polygones individuels de la prédiction
	:param box: Les coordonnées de la boîte [[xmin, ymin], [xmax, ymax]]
	:param line: Un dictionnaire représentant la ligne et
	 contenant la baseline, la prédiction et les cuts, de la forme:
		{
		  "baseline": [
			[215, 3372],
			[3289, 3392]
		  ],
		  "prediction": "A l'effet de juger le nommé, Braillon Eugìne Louis, fils de Cclestin Théophile et",
		  "cuts": [
			[
				[278, 3319], [278, 3412], [278, 3412], [278,3319]
			]
		  ]
		}
	:return: la chaîne de caractères reconstruite à partir des intersections
	"""
	assert len(line['prediction']) == len(line['cuts']), ("Un problème dans les données est apparu. "
														  "La longueur de la prédiction doit être identique "
														  "à celle des cuts")
	out_string = ""
	(xmin, ymin), (xmax, ymax) = box

	# Solution tirée de https://gis.stackexchange.com/a/90063
	polygon_soldat = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
	for char, cut in zip(line['prediction'], line['cuts']):
		cut = Polygon([tuple(coords) for coords in cut])
		intersection = polygon_soldat.intersects(cut)
		if intersection:
			out_string += char
	return out_string


def similarite_ratcliff(string_a, string_b):
	return SequenceMatcher(None, string_a, string_b).ratio()

def check_word_in_sentence(sentence:str, target_word:str, sensibility=0.5) -> (bool, str|None):
	"""
	Cette fonction vérifie si un mot (pouvant présenter des coquilles) est présent dans une phrase
	:param sentence: la phrase cible
	:param target_word: le mot à chercher
	:return: vrai ou faux et le mot identifié (ou None)
	"""
	target_word = target_word.lower()
	split_regexp = re.compile(r'[.!?,.:;\-\s]')
	sentence = re.split(split_regexp, sentence)
	distances = []
	matching_word = []
	for word in sentence:
		word_lower = word.lower()
		dist = similarite_ratcliff(word_lower, target_word)
		if dist > sensibility:
			distances.append(dist)
			matching_word.append(word)
	if len(distances) == 0:
		return False, None
	elif len(distances) > 1:
		print(f"Plus d'un mot trouvé, une erreur est possiblement survenue: {matching_word}."
			  f"On prend le dernier mot identifié.")
		# Dans ce cas on considère le dernier mot identifié, Greffier étant imprimé en fin de ligne.
		return True, matching_word[1]
	else:
		return True, matching_word[0]



def match_line_by_similarity(corresponding_lines:list, string_to_match:str):
	"""
	Cette fonction extrait la ligne qui contient une sous-chaîne la plus proche de la chaîne cible
	:param corresponding_lines: l'ensemble des lignes dans lesquelles chercher
	:param string_to_match: la chaîne à trouver
	:return: la ligne qui contient la chaîne de caractères
	"""
	# On commence par normaliser la chaîne à matcher
	string_to_match = string_to_match.lower()
	distances = []
	if len(corresponding_lines) > 1:
		for idx, ligne in enumerate(corresponding_lines):
			prediction = ligne['prediction']
			prediction = prediction.lower()
			# On identifie la ligne pouvant contenir à l'effet de juger
			dist = similarite_ratcliff(prediction, string_to_match)
			distances.append(dist)
	correct_line_index = distances.index(max(distances))
	name_line = corresponding_lines[correct_line_index]
	return name_line



def levensthein_distance(string_a, string_b):
	return distance(string_a, string_b)


def rectangle_to_baseline(rectangle):
	return [[rectangle.xmin, rectangle.ymin], [rectangle.xmax, rectangle.ymax]]


def check_if_line_in_box(box_coord, baseline, intersect_ratio=.5) -> bool:
	"""
	Cette fonction vérifie si une ligne est comprise pour au moins 50% dans une zone.
	Présuppose des lignes globalement droites (= représentables par des fonctions affines)
	:param box_coord: les coordonnées de la zone
	:param baseline: les points de la ligne
	:param intersect_ratio: la proportion de la ligne comprise dans la zone pour retourner vrai
	:return: Bool
	"""

	# On identifie la fonction qui représente la droite passant par les 2 points extrêmes de la ligne
	a, b = produce_line_function(baseline)

	# On regarde la distance horizontale entre ces deux points
	number_points = 20
	x_distance = round(baseline[-2] - baseline[0])
	steps = x_distance // number_points

	# On crée 20 points le long de la droite. Si la moitié sont dans la zone, on renvoie True
	try:
		n_points = [(item, round(a * item + b)) for item in range(baseline[0], baseline[-2], steps)]
	except ValueError as e:
		print(f"La ligne est verticale, on passe: {baseline}.")
		return False
	number_in = 0
	for point in n_points:
		if point_in_box(coord=point, box_coord=box_coord):
			number_in += 1
	if round(number_points * intersect_ratio) < number_in:
		return True
	else:
		return False


def check_if_missing(list_target, list_source):
	set1 = set(list_target)
	set2 = set(list_source)
	missing = list(sorted(set1 - set2))
	return missing


def pickle_object(obj, path):
	with open(path, "wb") as segmentation_as_file:
		pickle.dump(obj, segmentation_as_file, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_object(path):
	with open(path, "rb") as segmentation_as_file:
		return pickle.load(segmentation_as_file)


def save_as_dict(dictionnary:dict, path:str):
	with open(path, 'w') as f:
		# https://stackoverflow.com/a/36142844 default permet de gérer la sérialisation des objets bizarres (dates...)
		json.dump(dictionnary, f, indent=2, default=str)


def list_depth(lst: list) -> int:
	"""
	Retourne la profondeur maximale d'une liste
	:param lst: la liste à analyser
	:return: un entier
	"""
	return isinstance(lst, list) and max(map(list_depth, lst)) + 1


def load_json_to_dict(path):
	with open(path, 'r') as f:
		return json.load(f)


def get_name_from_path(path):
	basename = path.split('/')[-1].split('.')[0]
	dossier = "_".join(basename.split('_')[:-1])
	ident = basename.split('_')[-1]
	return dossier, int(ident)


def format_coordinates(coords):
	rounded = [round(item) for item in coords]
	return [[rounded[0], rounded[1]], [rounded[2], rounded[3]]]
