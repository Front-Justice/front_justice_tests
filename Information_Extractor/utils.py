import PIL.Image as Image
import re
from Levenshtein import distance
from difflib import SequenceMatcher
from shapely.geometry import Polygon
from collections import namedtuple
import time
from yaspin import yaspin


def load(path):
	return Image.open(path)

def split_after_keep_delimiter(string, delimiter):
	print("---")
	print(string)
	print(delimiter)
	print(len(string))
	out_split = []
	results = re.finditer(delimiter, string)
	delimiters = [0]
	for result in results:
		print(result.span())
		delimiters.append(result.span()[1])
	delimiters.append(len(string))
	print(delimiters)

	for position in range(len(delimiters[1:])):
		pos = position + 1
		print(f"appending ({delimiters[pos - 1], delimiters[pos]}) |{string[delimiters[pos - 1]: delimiters[pos]]}|")
		out_split.append(string[delimiters[pos - 1]: delimiters[pos]])
	print(out_split)
	print(len(out_split))
	return out_split

def split_before_keep_delimiter(string, delimiter):
	out_split = []
	results = re.finditer(delimiter, string)
	delimiters = [0]
	for result in results:
		delimiters.append(result.span()[0])
	delimiters.append(len(string))

	for position in range(len(delimiters[1:])):
		pos = position + 1
		out_split.append(string[delimiters[pos - 1]: delimiters[pos]])
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

def vertical_order_zones(annotations:list[dict]) -> list[dict]:
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
	if (dx>=0) and (dy>=0):
		overlap_area = round(dx*dy)
		ratio = round(overlap_area / area_source, 2)
		return ratio
	else:
		return None

def measured_party_inference(party_engine, segmentation, image, objet_transcrit):
	with yaspin(text=f"Transcription de {objet_transcrit}") as sp:
		start = time.time()
		prediction = party_engine.inference(segmentation=segmentation, image=image)
		end = time.time()
	print(f"Transcription de {objet_transcrit} faite en {end - start} secondes")
	return prediction


def extraction_prenom_du_soldat(prediction, nom_du_soldat, pipeline):
	"""
	Cette fonction utilise un NER pour extraire le prénom du soldat
	:param prediction: La chaîne de caractère
	:param nom_du_soldat:
	:param pipeline:
	:param debug:
	:return:
	"""
	result = pipeline(prediction.lower())
	words = [prediction[entity['start']:entity['end']] for entity in result]
	try:
		# Si on a un nom, on prend l'entité qui le contient,
		correct_entity = next(entity for entity in words if nom_du_soldat.lower() in entity.lower())
		forename = correct_entity.replace(nom_du_soldat, '').strip()
		certainty = 0.8
	except StopIteration:
		# Si le nom est mal reconnu, on considère que l'entité nommée est la première de la ligne
		correct_entity = words[0]
		forename = correct_entity
		certainty = 0.5
	return forename, certainty

	print(result)


def match_lines_in_zones(ocr_prediction:list[dict], zone_as_rectangle:namedtuple, intersect_ratio=0.5):
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

def extract_magistrates_names(prediction, pipeline, debug=False):
	if isinstance(prediction, list):
		example = " ".join(prediction)
	result = pipeline(prediction.lower())
	if debug:
		print("---")
		print(prediction)
		print(result)
	persName_NER = prediction[result[0]['start']: result[0]['end']] if result[0]["entity_group"] == "PER" else None
	role_NER = prediction[result[0]['end']:] if result[0]["entity_group"] == "PER" else None

	# Extraction simple: le premier mot. Pour les noms à particule c'est plus compliqué: aller chercher la virgule?
	match_first_word = re.search(re.compile(r"[^\s,.]+"), prediction)
	spans = match_first_word.span()
	homemade_NER = prediction[spans[0]: spans[1]]
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
	return {"persName": persName,
			"role": role,
			"certainty": certainty}

def extract_string_from_cuts(box:list[list[int]], line:dict) -> str:
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
	n_points = [(item , round(a * item + b)) for item in range(baseline[0], baseline[-2], steps)]
	number_in = 0
	for point in n_points:
		if point_in_box(coord=point, box_coord=box_coord):
			number_in += 1
	if round(number_points * intersect_ratio) < number_in:
		return True
	else:
		return False
