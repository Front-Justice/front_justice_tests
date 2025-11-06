import PIL.Image as Image
import re

def load(path):
	return Image.open(path)

def split_keep_delimiter(string, delimiter):
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
	b = y_1
	# print(f"Function: y = {round(a, 2)}*x + {b}")
	return a, b

def point_in_box(coord, box_coord):
	x, y = coord
	min_x, min_y, max_x, max_y = box_coord
	if min_x <= x <= max_x and min_y <= y <= max_y:
		return True
	else:
		return False

def check_if_line_in_box(box_coord, baseline):
	a, b = produce_line_function(baseline)
	x_distance = round(baseline[-2] - baseline[0])
	steps = x_distance // 20
	twenty_points = [(item , round(a * item + b)) for item in range(baseline[0], baseline[-2], steps)]
	for point in twenty_points:
		if point_in_box(coord=point, box_coord=box_coord):
			return True
	return False

def convert_coco_coordinates(coords):
	"""
	Cette fonction remplace le format coco (x, y, width, height) par le format
	(droite, haut, gauche, bas)
	:param coords: les coordonnées au format coco
	:return:
	"""
	converted = [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]]
	return converted
