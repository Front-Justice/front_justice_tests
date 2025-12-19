import csv
import json
import pickle
import random
import string
import unicodedata
import PIL.Image as Image
import PIL
import re
from thefuzz import fuzz
from Levenshtein import distance
from difflib import SequenceMatcher
from shapely.geometry import Polygon
from collections import namedtuple
import spellchecker

number_dict = {"un": 1,
				"deux": 2,
				"trois": 3,
				"quatre": 4,
				"cinq": 5,
				"six": 6,
				"sept": 7,
				"huit": 8,
				"neuf": 9,
				"dix": 10,
				"onze": 11,
				"douze": 12,
				"treize": 13,
				"quatorze": 14,
				"quinze": 15,
				"seize": 16,
				"dix sept":17,
				"dix huit": 18,
				"dix neuf": 19,
				"vingt": 20,
				"vingt-et-un": 21,
				"vingt deux": 22,
				"vingt trois": 23,
				"vingt quatre": 24,
				"vingt cinq": 25,
				"vingt six": 26,
				"vingt sept": 27,
				"vingt huit": 28,
				"vingt neuf": 29,
				"trente": 30,
				"trente et un": 31,
				"mil": 1000,
				"cent": 100,
			   "enfants": "enfants"}

def load(path):
	return Image.open(path)


def calcule_age(date_naissance:str, date_proces:str) -> int|None:
	"""
	Cette fonction calcule l'âge du soldat étant donné sa date de naissance et la date du procès.
	Note: l'exception se fait en amont.
	:param date_naissance: la date de naissance, format DD/JJ/MMMM
	:param date_proces: la date du procès, format DD/JJ/MMMM
	:return: l'age ou "Inconnu" si il manque une des deux dates.
	"""
	print(date_naissance, date_proces)
	annee_proces = int(date_proces.split("/")[-1])
	annee_naissance = int(date_naissance.split("/")[-1])

	return annee_proces - annee_naissance

def split_after_keep_delimiter(target_string: str, delimiter: str) -> list:
	"""
		Cette fonction coupe une phrase selon un délimiteur qui est une expression régulière, et garde le délimiteur.
		La coupe se fait après le délimiteur.
		:param target_string: la chaîne à couper
		:param delimiter: le délimiteur sous la forme d'une chaîne de caractères qui sera compilée après normalisation
		:return: la liste voulue
		"""
	# On commence par normaliser les chaînes de caractères
	delimiter = nfc_normalize(delimiter)
	delimiter_as_regexp = re.compile(delimiter)
	target_string = nfc_normalize(target_string)
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

def tokenize_sent(sentence:str) -> list:
	punctuation_and_space = re.compile(r'(["\'\-?,!;\.:\s])')
	tokenized = re.split(punctuation_and_space, sentence)
	tokenized = [item for item in tokenized if item != " "]
	return tokenized


def correct_date(date:str) -> str:
	number_dict = {"un": 1,
						"deux": 2,
						"trois": 3,
						"quatre": 4,
						"cinq": 5,
						"six": 6,
						"sept": 7,
						"huit": 8,
						"neuf": 9,
						"dix": 10,
						"onze": 11,
						"douze": 12,
						"treize": 13,
						"quatorze": 14,
						"quinze": 15,
						"seize": 16,
						"vingt": 20,
						"trente": 30,
						"mil": 1000,
						"cent": 100}

	month_dict = {"janvier": "01",
					   "février": "02",
					   "mars": "03",
					   "avril": "04",
					   "mai": "05",
					   "juin": "06",
					   "juillet": "07",
					   "août": "08",
					   "septembre": "09",
					   "octobre": "10",
					   "novembre": "11",
					   "décembre": "12",
					   }
	date = nfc_normalize(date)
	date = date.lower().strip()
	clean_regexp = re.compile(r"(\d+)\^?er?")
	date = re.sub(clean_regexp, r'\g<1>', date)
	date = strip_punctuation(date)

	# On corrige les erreurs fŕequentes
	common_mistakes = {"aout": "août",
					   "dix": "dix ",
					   "vingt": "vingt ",
					   "trente": "trente "}
	for orig, reg in common_mistakes.items():
		date = date.replace(orig, reg)
	splits = re.compile(r"[\s+\-]")
	splitted = re.split(splits, date)
	result = []
	for token in splitted:
		if token in common_mistakes:
			result.append(common_mistakes[token])
		elif token in month_dict or token in number_dict or token in ['de', 'du', 'an', 'et', 'en']:
			result.append(token)
		else:
			matching, corrected = check_word_in_list(list(month_dict.keys()) + list(number_dict.keys()),
													 token,
													 sensibility=0.7 if len(token) > 4 else 0.6)
			if matching:
				result.append(corrected)
			else:
				result.append(token)
	normalized = " ".join([item for item in result if item != ""])
	normalized = normalized.lower()
	return normalized

def correct_based_on_list(sentence, list):
	"""
	Cette fonction corrige une phrase en se fondant sur une liste de mots définie en amont.
	:param sentence: la phrase à corriger
	:param list: la liste de mots importants
	:return: la phrase corrigée
	"""
	splits = re.compile(r"[\s+\-]")
	splitted = re.split(splits, sentence)
	result = []
	for token in splitted:
		matching, corrected = check_word_in_list(list, token, sensibility=0.7 if len(token) > 4 else 0.6)
		if matching:
			result.append(corrected)
		else:
			result.append(token)
	normalized = " ".join([item for item in result if item != ""])
	normalized = normalized.lower()
	print(f"{sentence} -> {normalized}")
	return normalized

def correct_description_soldat(string:str):
	liste_termes_frequents = ['rectiligne',
							  'long',
							  'menton',
							  'visage',
							  'yeux',
							  'front',
							  'canonnier',
							  'artillerie',
							  'cheveux']

def correct_string(string:str) -> str:
	correcteur = spellchecker.spellchecker.SpellChecker(language='fr')
	corrected_string = []
	tokens = tokenize_sent(string)
	for token in tokens:
		corr = correcteur.correction(token)
		if corr:
			corrected_string.append(corr)
	return " ".join(corrected_string)



def nfc_normalize(input_string: str) -> str:
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
	delimiter = nfc_normalize(delimiter)
	delimiter_as_regexp = re.compile(delimiter)
	target_string = nfc_normalize(target_string)

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


def produce_line_function(baseline) -> tuple[int, int]:
	"""
	Cette fonction analyse et récupère les paramètres de la fonction affine y = ax+b par laquelle passe une droite.
	:param baseline: la ligne, sous la forme x_1, x_2, y_1, y_2.
	:return: a, b.
	"""
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
		try:
			correct_entity = words[0]
			forename = clean_forename(correct_entity)
			certainty = 0.5
		except IndexError:
			forename = None
			certainty = None
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

def clean_spaces(string) -> str:
	spaces_regexp = re.compile("\s+")
	return re.sub(spaces_regexp, " ", string)

def full_clean_string(string) -> str:
	"""
	Cette fonction a vocation à nettoyer completement une chaîne de caractères de la ponctuation et des espaces,
	elle supprime aussi
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	string = remove_all_punctuation(string)
	string = strip_stopwords(string)
	string = clean_spaces(string)
	return string

def strip_stopwords(string):
	stopwords = re.compile("^du |^de la |^de |^[àa] |y |de l'")
	clean = re.sub(stopwords, "", string)
	return clean

def remove_all_punctuation(string:str, debug=False) -> str:
	"""
	Cette fonction supprime la ponctuation en début et fin de chaîne
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	orig_string = string
	expression = "[\(\),;.!?\-:]"
	punct_regexp = re.compile(expression)
	string = string.strip()
	string = re.sub(punct_regexp, " ", string)
	string = string.strip()
	if debug:
		print(f"|{orig_string}| -> |{string}|")
	return string

def strip_punctuation(string:str|None, debug=False) -> str|None:
	"""
	Cette fonction supprime la ponctuation en début et fin de chaîne
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	if string is None:
		return None
	orig_string = string
	punctuation = "[\(\),;.!?\-:]"
	expression = "^"+ punctuation + "\s{0,}|\s{0,}"+ punctuation + "$"
	punct_regexp = re.compile(expression)
	string = string.strip()
	string = re.sub(punct_regexp, "", string)
	string = string.strip()
	if debug:
		print(f"|{orig_string}| -> |{string}|")
	return string


def convert_to_csv(extractions:dict, outpath:str):
	extracted_data = [["Numero_image",
					   "Id",
					  "Date du procès",
					   "Institution engagée",
					   "Lieu du procès",
					   "Numéro du jugement",
					   "Numéro d'ordre",
					   "Président du jury",
					   "Juré 1",
					   "Juré 2",
					   "Juré 3",
					   "Juré 4",
					   "Greffier",
					   "Commissaire",
					   "Général nommant",
					   "Date du crime ou du délit",
					  "Nom",
					  "Prénoms",
					  "Date de naissance",
					  "Âge",
					   "Taille",
					   "Cheveux",
					   "Front",
					   "Yeux",
					   "Nez",
					   "Visage",
					   "Ville de naissance",
					   "Arrondissement de naissance",
					   "Département de naissance",
					   "Ville de résidence",
					   "Arrondissement de résidence",
					   "Département de résidence",
					   "Situation maritale",
					   "Enfants",
					   "Profession",
					   "Rang du soldat",
					   "Numéro de matricule",
					   "Chef d'accusation",
					   "Antécédents"]]
	for idx_minute, minute in extractions.items():
		for idx_page, page in enumerate(minute):
			try:
				page['extractions']
			except KeyError:
				continue
			interm = []
			# Image
			print(page['image_path'])
			interm.append(page['image_path'])

			# ID
			interm.append(random_string())
			# Date du procès
			try:
				date_proces = page['extractions']['date_proces']['date_normalisee']['when']
			except TypeError:
				date_proces = "?"
			interm.append(date_proces)

			# Lieu du procès
			try:
				institution = page['extractions']['lieu_jugement']['institution']
			except TypeError:
				institution = "?"
			interm.append(institution)

			try:
				lieu_proces = page['extractions']['lieu_jugement']['siège']
			except TypeError:
				lieu_proces = "?"
			interm.append(lieu_proces)

			# Numéro de jugement
			try:
				numero_jugement = page['extractions']['numero_jugement']['extracted']
			except TypeError:
				numero_jugement = "?"
			except KeyError:
				numero_jugement = "?"
			interm.append(numero_jugement)

			# Numéro d'ordre
			try:
				numero_ordre = page['extractions']['numero_ordre']['extracted']
			except TypeError:
				numero_ordre = "?"
			interm.append(numero_ordre)

			# Président du jury (rôle non extrait)
			president = page['extractions']['magistrats']['president']['extracted']['persName']
			interm.append(president)

			# Jurés (on n'extrait pas les rôles)

			jures = page['extractions']['magistrats']['jures']
			for jure in jures[:4]:
				try:
					extracted_jure = jure['extracted']['persName']
				except TypeError:
					extracted_jure = "?"
				interm.append(extracted_jure)

			# Greffier (on n'extrait pas les rôles)
			greffier = page['extractions']['magistrats']['greffier']['extracted']['persName']
			interm.append(greffier)

			# Commissaire du gouvernement (on n'extrait pas les rôles)
			commissaire = page['extractions']['magistrats']['commissaire']['extracted']['persName']
			interm.append(commissaire)

			# Général
			general = page['extractions']['magistrats']['general']['extracted']
			interm.append(general)

			# Date du crime
			try:
				date_crime = page['extractions']['date_du_crime_ou_delit']['date_normalisee']
			except KeyError:
				date_crime = "?"
			except TypeError:
				date_crime = "?"
			interm.append(json.dumps(date_crime))

			# Nom et prénom du soldat
			try:
				prenoms_soldat = page['extractions']['description_soldat']['nom_du_soldat']['extracted']['forename']['persName']
				nom_soldat = page['extractions']['description_soldat']['nom_du_soldat']['extracted']['surname']['persName']
			except TypeError:
				nom_soldat = "Plusieurs soldats"
				prenoms_soldat = "Plusieurs soldats"
				interm.append(nom_soldat)
				interm.append(prenoms_soldat)
				extracted_data.append(interm)
				continue
			interm.append(nom_soldat)
			interm.append(prenoms_soldat)

			# Date de naissance et âge du soldat
			try:
				date_naissance = page['extractions']['description_soldat']['date_de_naissance']['date_normalisee']['when']
			except TypeError:
				date_naissance = "?"
			try:
				age = page['extractions']['description_soldat']["Âge"]
			except KeyError:
				age = "?"
			interm.append(date_naissance)
			interm.append(age)

			try:
				taille = page['extractions']['description_soldat']["physique"]["taille"]["extracted"]
			except KeyError:
				taille = "?"
			try:
				cheveux = page['extractions']['description_soldat']["physique"]["cheveux"]["extracted"]
			except KeyError:
				cheveux = "?"
			try:
				front = page['extractions']['description_soldat']["physique"]["front"]["extracted"]
			except KeyError:
				front = "?"
			try:
				yeux = page['extractions']['description_soldat']["physique"]["yeux"]["extracted"]
			except KeyError:
				yeux = "?"
			try:
				nez = page['extractions']['description_soldat']["physique"]["nez"]["extracted"]
			except KeyError:
				nez = "?"
			try:
				visage = page['extractions']['description_soldat']["physique"]["visage"]["extracted"]
			except KeyError:
				visage = "?"
			interm.append(taille)
			interm.append(cheveux)
			interm.append(front)
			interm.append(yeux)
			interm.append(nez)
			interm.append(visage)

			# Lieu de naissance
			ville_naissance = page['extractions']['description_soldat']['lieu_naissance']['extracted']['ville']
			arrondissement_naissance = page['extractions']['description_soldat']['lieu_naissance']['extracted']['arrondissement']
			departement_naissance = page['extractions']['description_soldat']['lieu_naissance']['extracted']['departement']
			interm.append(ville_naissance)
			interm.append(arrondissement_naissance)
			interm.append(departement_naissance)

			# Lieu de résidence
			ville_residence = page['extractions']['description_soldat']['lieu_residence']['extracted']['ville']
			arrondissement_residence = page['extractions']['description_soldat']['lieu_residence']['extracted']['arrondissement']
			departement_residence = page['extractions']['description_soldat']['lieu_residence']['extracted']['departement']
			interm.append(ville_residence)
			interm.append(arrondissement_residence)
			interm.append(departement_residence)

			# Femme et enfants
			situation_maritale = page['extractions']['description_soldat']['situation_maritale']
			if situation_maritale['célibataire'] == True:
				situation_maritale = "célibataire"
			elif situation_maritale["marié"] == True:
				situation_maritale = "marié"
			elif situation_maritale["veuf"] == True:
				situation_maritale = "veuf"
			interm.append(situation_maritale)

			enfants = page['extractions']['description_soldat']['situation_maritale']['nombre_enfants']
			interm.append(enfants)

			# Profession
			profession = page['extractions']['description_soldat']['profession']['extracted']
			interm.append(profession)

			# Rang du soldat
			rang = page['extractions']['description_soldat']['rang']['extracted']
			interm.append(rang)

			# Numéro de matricule
			try:
				matricule = page['extractions']['description_soldat']['matricule']['extracted']
			except:
				matricule = None
			interm.append(matricule)

			# Chef d'accusation
			chef_accusation = page['extractions']['chef_accusation']['extracted']
			interm.append(chef_accusation)

			# Antécédent (juste le nombre)
			antecedents = page['extractions']['antécédents']['extracted']
			if antecedents != "Néant":
				antecedents = len(antecedents)
			interm.append(antecedents)
			extracted_data.append(interm)
	with open(outpath, 'w', newline='') as f:
		writer = csv.writer(f, delimiter="$")
		writer.writerows(extracted_data)


def random_string():
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

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

def test_number_of_zones(annotations:list[dict], label:str, number:int) -> bool:
	"""
	Cette fonction permet de vérifier si les annotations YOLO contiennent le nombre de zones attendues
	:param annotations: La liste des annotations (dictionnaire {label, coordinates})
	:param label: le label à vérifier
	:param number: le nombre de zones attendues
	:return:
	"""
	filtered_list = [item for item in annotations if item['label'] == label]
	if len(filtered_list) == number:
		return True
	else:
		return False

def similarite_ratcliff(string_a, string_b):
	return SequenceMatcher(None, string_a, string_b).ratio()

def approximate_split(sentence:str, word:str, sensibility:float=0.5, return_word:bool=False) -> list|tuple[list, str]|None|tuple[None, None]:
	"""
	Cette fonction découpe une phrase selon un mot qui peut être approximatif. Le mot n'est pas retourné.
	:param sentence: la phrase à découper
	:param word: le mot sur lequel s'appuyer
	:param sensibility: la sensibilité a appliquer à la recherche (à trouver par l'expérience)
	:return:
	"""
	sentence = nfc_normalize(sentence)
	word = nfc_normalize(word)
	match, matching_word = check_word_in_sentence(sentence, word, sensibility)
	if match:
		if return_word:
			return sentence.split(matching_word), matching_word
		else:
			return sentence.split(matching_word)
	else:
		if return_word:
			return None, None
		else:
			return None


def check_word_in_list(word_list:list, target_word:str, sensibility=0.7) -> (bool, str|None):
	"""
	Cette fonction vérifie si un mot (pouvant présenter des coquilles) est présent dans une liste de mots
	:param sentence: la phrase cible
	:param target_word: le mot à chercher
	:return: vrai ou faux et le mot identifié (ou None)
	"""
	distances = []
	matching_words = []
	target_word = target_word.lower()
	for word in word_list:
		word_lower = word.lower()
		dist = similarite_ratcliff(word_lower, target_word)
		if dist > sensibility:
			matching_words.append(word)
			distances.append(dist)
	if len(distances) == 0:
		return False, target_word
	max_dist:int = distances.index(max(distances))
	return True, matching_words[max_dist]


def check_word_in_sentence(sentence:str, target_word:str|list, sensibility=0.5) -> tuple[bool, str|None]:
	"""
	Cette fonction vérifie si un mot (pouvant présenter des coquilles) est présent dans une phrase
	:param sentence: la phrase cible
	:param target_word: le mot à chercher
	:return: vrai ou faux et le mot identifié (ou None)
	"""
	sentence = nfc_normalize(sentence)
	split_regexp = re.compile(r'[.!?,.:;\-\s]')
	sentence = re.split(split_regexp, sentence)
	distances = []
	matching_word = []
	if isinstance(target_word, str):
		target_word = [target_word]
	else:
		pass
	for word in sentence:
		word_lower = word.lower()
		for item in target_word:
			item = item.lower().strip()
			item = nfc_normalize(item)
			dist = similarite_ratcliff(word_lower, item)
			if dist > sensibility:
				distances.append(dist)
				matching_word.append(word)
	if len(distances) == 0:
		return False, False
	elif len(distances) > 1:
		print(f"Plus d'un mot trouvé, une erreur est possiblement survenue: {matching_word}."
			  f"On prend le dernier mot identifié.")
		# Dans ce cas on considère le dernier mot identifié, étant imprimé en fin de ligne.
		return True, matching_word[1]
	else:
		return True, matching_word[0]



def check_substring_in_line(corresponding_lines:list, string_to_match:str|list, return_index:bool=False):
	"""
	Cette fonction extrait la ligne qui contient une sous-chaîne la plus proche de la chaîne cible
	:param corresponding_lines: l'ensemble des lignes dans lesquelles chercher
	:param string_to_match: la chaîne à trouver ou une liste de chaines alternatives à identifier
	:return: la ligne qui contient la chaîne de caractères et le zip de 1) la ligne et 2) la similarité avec la requête.
	Peut également retourner l'indice de l'item identifié dans la liste.
	"""
	# On commence par normaliser la chaîne à matcher
	distances = []
	for idx, ligne in enumerate(corresponding_lines):
		prediction = ligne['prediction']
		prediction = prediction.lower()
		prediction = nfc_normalize(prediction)
		# On identifie la ligne pouvant contenir à l'effet de juger
		if isinstance(string_to_match, list):
			pass
		else:
			string_to_match = [string_to_match]
		interm_distances = []
		for item in string_to_match:
			item = item.lower()
			item = nfc_normalize(item)
			if item in prediction:
				interm_distances.append(9999)
			elif len(prediction) < 10:
				interm_distances.append(0)
			else:
				# dist = similarite_ratcliff(prediction, string_to_match)
				dist = fuzz.partial_ratio(prediction, item)
				interm_distances.append(dist)
		distances.append(max(interm_distances))
	correct_line_index = distances.index(max(distances))
	name_line = corresponding_lines[correct_line_index]
	debug_zip = list(zip([item['prediction'] for item in corresponding_lines], distances))
	if return_index is True:
		return name_line, debug_zip, correct_line_index
	else:
		return name_line, debug_zip



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


def get_baseline_from_string(line:dict,
							 target_string:str,
							 loaded_image:Image.Image=None,
							 show_image:bool=False) -> tuple[tuple[int, int], tuple[int, int]] | None:
	"""
	Cette fonction récupère les coordonnées du fragment de baseline qui contient une chaîne de caractère donnée
	:param line: La ligne, dictionnaire {prediction, baseline, cuts}
	:return: La ligne de base qui contient le texte : [[x_1, y_1], [x_2, y_2]]
	"""
	assert all([item in line for item in ['cuts', 'prediction', 'baseline']]), f"La structure de la ligne est incorrecte: {line}"
	cuts = line["cuts"]
	baseline = line["baseline"]
	prediction = line["prediction"]
	prediction = prediction.lower()
	target_string = target_string.lower()
	if target_string not in prediction.lower():
		print(f"La ligne ne contient pas {target_string}")
		return None
	first_char_idx, last_char_idx = (prediction.find(target_string),
									 prediction.find(target_string) + len(target_string) - 1)

	first_cut = cuts[first_char_idx]
	last_cut = cuts[last_char_idx]
	# On extrait l'abscisse minimale et maximale et on ajoute un peu de marge à droite et à gauche
	x_1 = min(item[0] for item in first_cut) - 40
	x_2 = max(item[0] for item in last_cut) + 40
	baseline = [baseline[0], baseline[-1]]
	formatted_baseline = baseline[0][0], baseline[0][1], baseline[1][0], baseline[1][1]
	a, b = produce_line_function(formatted_baseline)

	# On calcule y1 et y2
	y_1 = round(a*x_1 + b)
	y_2 = round(a*x_2 + b)
	target_baseline = [[x_1, y_1], [x_2, y_2]]

	if show_image:
		cropped = loaded_image.crop((x_1, y_1 - 70, x_2, y_2 + 70))
		cropped.show()
	return target_baseline