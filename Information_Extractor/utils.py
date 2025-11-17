import PIL.Image as Image
import re

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


def vertical_order_zones(annotations:list[dict]) -> list[dict]:
	"""
	Fonction pour ordonner les zones verticalement (du plus haut au plus bas).
	On ordonne par la deuxième coordonnée de la boîte (y1)
	:param annotations: Les annotations sous la forme d'une liste de dictionnaire:
	[
		{'label': 'ligne', 'coordinates': [2713, 2242, 3033, 2857]},
		{'label': 'ligne', 'coordinates': [213, 2236, 2745, 2404]}
	]
	:return: Les mêmes annotations ordonnées
	"""
	sorted_list = sorted(annotations, key=lambda x: x["coordinates"][1])
	return sorted_list

def rectanglify(coords):
	return

def horizontal_order_zones(annotations):
	"""
	Fonction pour ordonner les zones horizontalement (de gauche à droite).
	On ordonne par la première coordonnée de la boîte (x1)
	:param annotations: Les annotations sous la forme d'une liste de dictionnaire:
	[
		{'label': 'ligne', 'coordinates': [2713, 2242, 3033, 2857]},
		{'label': 'ligne', 'coordinates': [213, 2236, 2745, 2404]}
	]
	:return: Les mêmes annotations ordonnées
	"""
	sorted_list = sorted(annotations, key=lambda x: x["coordinates"][0])
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

def process_name(example, pipeline):
	if isinstance(example, list):
		example = " ".join(example)
	print("---")
	print(example)
	result = pipeline(example.lower())
	print(result)
	persName_NER = example[result[0]['start']: result[0]['end']] if result[0]["entity_group"] == "PER" else None
	role_NER = example[result[0]['end']:] if result[0]["entity_group"] == "PER" else None

	# Extraction simple: le premier mot. Pour les noms à particule c'est plus compliqué: aller chercher la virgule?
	match_first_word = re.search(re.compile(r"[^\s,.]+"), example)
	spans = match_first_word.span()
	homemade_NER = example[spans[0]: spans[1]]
	homemade_role = example[spans[1]:]
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


def check_if_line_in_box(box_coord, baseline):
	"""
	Cette fonction vérifie si une ligne est comprise pour au moins 50% dans une zone.
	Présuppose des lignes globalement droites (= représentables par des fonctions affines)
	:param box_coord: les coordonnées de la zone
	:param baseline: les points de la ligne
	:return:
	"""

	# On identifie la fonction qui représente la droite passant par les 2 points extrêmes de la ligne
	a, b = produce_line_function(baseline)

	# On regarde la distance horizontale entre ces deux points
	x_distance = round(baseline[-2] - baseline[0])
	steps = x_distance // 20

	# On crée 20 points le long de la droite. Si la moitié sont dans la zone, on renvoie True
	twenty_points = [(item , round(a * item + b)) for item in range(baseline[0], baseline[-2], steps)]
	number_in = 0
	for point in twenty_points:
		if point_in_box(coord=point, box_coord=box_coord):
			number_in += 1
	if 10 < number_in:
		return True
	else:
		return False
