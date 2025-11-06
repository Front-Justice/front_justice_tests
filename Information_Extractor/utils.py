import copy

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
	# print(f"Function: y = {round(a, 2)}*x + {b}")
	return a, b

def point_in_box(coord, box_coord):
	x, y = coord
	min_x, min_y, max_x, max_y = box_coord
	if min_x <= x <= max_x and min_y <= y <= max_y:
		return True
	else:
		return False


def vertical_order_zones(annotations):
	interm = [(item['bbox'][1], item) for item in annotations]
	sorted_list = sorted(interm, key=lambda x: x[0])
	return sorted_list


def horizontal_order_zones(annotations):
	interm = [(item['bbox'][0], item) for item in annotations]
	sorted_list = sorted(interm, key=lambda x: x[0])
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

def check_if_line_in_box(box_coord, baseline):
	"""
	Cette fonction vérifie si une ligne est comprise pour au moins 50% dans une zone
	:param box_coord: les coordonnées de la zone
	:param baseline: les points de la ligne
	:return:
	"""
	a, b = produce_line_function(baseline)
	x_distance = round(baseline[-2] - baseline[0])
	steps = x_distance // 20
	twenty_points = [(item , round(a * item + b)) for item in range(baseline[0], baseline[-2], steps)]
	number_in = 0
	for point in twenty_points:
		if point_in_box(coord=point, box_coord=box_coord):
			number_in += 1
	if 10 < number_in:
		return True
	else:
		return False

def convert_coco_coordinates(coords):
	"""
	Cette fonction remplace le format coco (x, y, width, height) par le format
	(droite, haut, gauche, bas)
	:param coords: les coordonnées au format coco
	:return:
	"""
	converted = [round(coords[0]), round(coords[1]), round(coords[0] + coords[2]), round(coords[1] + coords[3])]
	return converted
