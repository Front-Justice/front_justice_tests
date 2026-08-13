import logging

import cv2
import json
import math
from datetime import datetime, timedelta
import rapidfuzz
import pickle
import random
import string
import unicodedata
import PIL.ImageDraw
import PIL.Image as Image
import re
from typing import Union, Self

import numpy as np
from thefuzz import fuzz
from Levenshtein import distance
from difflib import SequenceMatcher
from shapely.geometry import Polygon
from collections import namedtuple
import pandas as pd
from dataclasses import dataclass
import text_to_num
import fuzzysearch

import src.Vision.KRAKEN as KRAKEN
import kraken.containers as containers
import logging


logger = logging.getLogger(__name__)

@dataclass
class DateRecord:
	"""Classe principale pour la description d'une date"""
	def __init__(self, extracted, predicted, normalized, bbox, baseline, corrected):
		self.extracted: str = extracted
		self.predicted: str = predicted
		self.normalized: dict = normalized
		self.corrected: str = corrected
		self.bbox: list = bbox
		self.baseline: list = baseline

	def to_json(self):
		return {
			"predicted": self.predicted,
			"extracted": self.extracted,
			"corrected": self.corrected,
			"normalized": self.normalized,
			"bbox": self.bbox,
			"baseline": self.baseline
		}


@dataclass
class YOLOZone:
	"""CLasse principale pour la description d'une zone"""

	def __init__(self, label, coordinates, probs):
		self.label: str = label
		self.coordinates: list[list] = coordinates
		self.probs: str = probs

	def __repr__(self):
		return json.dumps({"label": self.label,
						   "coordinates": self.coordinates,
						   "probs": self.probs})


class YOLORecord():
	def __init__(self, record: list[dict]):
		self.record: list = []
		for item in record:
			Zone: YOLOZone = YOLOZone(label=item['label'], coordinates=item['coordinates'], probs=item['probs'])
			self.record.append(Zone)

	def index(self, item):
		return self.record.index(item)

	def __iter__(self):
		return iter(self.record)

	def __len__(self) -> int:
		return len(self.record)

	def __getitem__(self, index: int) -> YOLOZone:
		return self.record[index]

	def __repr__(self):
		return self.record

	def __str__(self):
		return json.dumps([{"label": item.label,
							"probs": item.probs,
							"coordinates": item.coordinates} for item in self.record])

	def filter_zones(self, label) -> list[YOLOZone]:
		return [item for item in self.record if item.label == label]

	def to_json(self):
		"""
		Transforme un objet YOLORecord en dictionnaire.
		:return:
		"""
		dictionnary = []
		for item in self.record:
			dictionnary.append({"label": item.label,
								"coordinates": item.coordinates
								})
		return dictionnary


@dataclass
class OCRLine:
	"""
	Classe principale pour la description d'une ligne en sortie de Kraken ou Party. Contient la baseline, la prédiction,
	les cuts (= les polygones individuels pour chaque caractère prédit) et le polygone le cas échéant
	"""
	image_path: str
	baseline: list
	prediction: str
	cuts: list
	polygon: list | None = None
	prediction_with_deletion: str | None = None


class OCRRecord():
	"""
	Classe principale qui contient les résultats de l'OCR.
	Une liste d'objets OCRLine qui contiennent chacun la baseline,
	la prédiction, les cuts, les polygones et le chemin vers l'image.
	"""

	def __init__(self, record: list[dict] = []):
		self.record: list = []
		for line in record:
			Line: OCRLine = OCRLine(baseline=line['baseline'],
									prediction=line['prediction'],
									prediction_with_deletion=None,
									cuts=line['cuts'],
									polygon=line['polygon'],
									image_path=line['image_path'])
			self.record.append(Line)

	def recreate_record(self, list_of_lines: list[OCRLine]):
		self.record: list = []
		for line in list_of_lines:
			self.record.append(line)

	def join_transcription(self, merge_newlines=True) -> str:
		"""
		Cette fonction retourne le texte fusionné de toutes les lignes d'un OCRRecord.
		:return: Une chaîne de caractères.
		"""
		if merge_newlines is False:
			delimiter = "\n"
		else:
			delimiter = " "
		return delimiter.join([line.prediction for line in self.record])

	def index(self, item):
		return self.record.index(item)

	def __iter__(self):
		return iter(self.record)

	def __len__(self) -> int:
		return len(self.record)

	def __getitem__(self, index: int) -> Union["OCRLine", Self]:
		current_slice = self.record[index]
		if isinstance(current_slice, list):
			new_record = OCRRecord()
			new_record.recreate_record(list_of_lines=current_slice)
		else:
			new_record = current_slice
		return new_record

	def __str__(self, show_cuts=False):
		out_dict = []
		for line in self.record:
			if show_cuts is True:
				out_dict.append({"baseline": line.baseline,
								 "cuts": line.cuts,
								 "prediction": line.prediction,
								 "prediction_with_deletion": line.prediction_with_deletion,
								"image_path": line.image_path})
			else:
				out_dict.append({"baseline": line.baseline,
								 "prediction": line.prediction,
								 "polygon": line.polygon,
								 "prediction_with_deletion": line.prediction_with_deletion,
								"image_path": line.image_path})
		return json.dumps(out_dict)

	def to_json(self):
		"""
		Transforme un objet OCRRecord en dictionnaire.
		:return:
		"""
		dictionnary = []
		for line in self.record:
			dictionnary.append({"prediction": line.prediction,
								"baseline": line.baseline,
								"cuts": line.cuts,
								"polygon": line.polygon,
								 "prediction_with_deletion": line.prediction_with_deletion,
								"image_path": line.image_path})
		return dictionnary

	def from_json(self, path) -> None:
		"""
		Instancie un objet de classe OCRRecord à partir d'un fichier JSON.
		:param path: le chemin vers le fichier
		"""
		lines_as_dict = load_json_to_dict(path)
		if lines_as_dict is None:
			print(f"Error with {path}")
			self.record = None
			return
		self.record = []
		for item in lines_as_dict:
			if "polygon" not in item:
				item["polygon"] = None
			try:
				prediction_with_deletion = item['prediction_with_deletion']
			except KeyError:
				prediction_with_deletion = None
			Line: OCRLine = OCRLine(baseline=item['baseline'],
									prediction=item['prediction'],
									cuts=item['cuts'],
									polygon=item['polygon'],
									prediction_with_deletion=prediction_with_deletion,
									image_path=item["image_path"])
			self.record.append(Line)


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
			   "dix sept": 17,
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


def show_image(path):
	loaded_image = Image.open(path)
	loaded_image.show()


def calcule_age(date_naissance: str, date_proces: str) -> int | None:
	"""
	Cette fonction calcule l'âge du soldat étant donné sa date de naissance et la date du procès.
	Note: l'exception se fait en amont.
	:param date_naissance: la date de naissance, format DD/JJ/MMMM
	:param date_proces: la date du procès, format DD/JJ/MMMM
	:return: l'age ou "Inconnu" si il manque une des deux dates.
	"""
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


def tokenize_sent(sentence: str) -> list:
	punctuation_and_space = re.compile(r'(["\'\-?,!;\.:\s])')
	tokenized = re.split(punctuation_and_space, sentence)
	tokenized = [item for item in tokenized if item != " "]
	return tokenized


def correct_date(date: str) -> str:
	"""
	Cette fonction vise à corriger une date extraite et où seraient présentes des erreurs
	d'HTR:
	:param date: la chaîne de caractères à corriger
	:return: la date corrigée
	"""
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
	le_regexp = re.compile(r"^[Ll]e ")
	date = re.sub(le_regexp, r"", date)
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
													 sensibility=0.7 if len(token) > 4 else 0.57)
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
	return normalized


def correct_description_soldat(string: str):
	liste_termes_frequents = ['rectiligne',
							  'long',
							  'menton',
							  'visage',
							  'yeux',
							  'front',
							  'canonnier',
							  'artillerie',
							  'cheveux']





def nfc_normalize(input_string: str) -> str:
	"""
	Cette fonction applique une normalisation unicode NFC à la chaîne de caractères voulue.
	:param input_string:
	:return:
	"""
	assert isinstance(input_string, str), (f"Input string should be a string. "
										   f"Actually: {type(input_string)}."
										   f"Current string: {input_string}")
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

	# Dans certains cas de ligne verticale, on a une division par 0
	try:
		a = (y_2 - y_1) / (x_2 - x_1)
	except ZeroDivisionError:
		a = (y_2 - y_1) / 0.00001
	b = y_1 - a * x_1
	return a, b


def point_in_box(coord, box_coord):
	"""
	Vérifie si un point se trouve dans une boîte
	:param coord:
	:param box_coord:
	:return:
	"""
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

def split_date(date:str):
	if len(date.split("/")) == 3:
		day, month, year = date.split("/")
	elif len(date.split("/")) == 2:
		month, year = date.split("/")
		day = 1
	elif len(date.split("/")) == 1:
		year = date
		month = 1
		day = 1
	return int(day), int(month), int(year)

def is_anterior_or_equal(date_a:str, date_b:str) -> bool:
	"""
	Cette fonction vérifie si une date a est antérieure à une date b
	:param date_a: une date formattée jj/mm/aaaa
	:param date_b: une date formattée jj/mm/aaaa
	:return:
	"""
	day_a, month_a, year_a = split_date(date_a)
	day_b, month_b, year_b = split_date(date_b)
	date_a_formatted = datetime(year_a, month_a, day_a)
	date_b_formatted = datetime(year_b, month_b, day_b)
	if date_a_formatted <= date_b_formatted is False:
		logger.warning(f"{date_a_formatted} is before {date_b_formatted}, there must be an error. Ignoring the date.")
	return date_a_formatted <= date_b_formatted

def extract_bbox_from_baseline(target_line:OCRLine, image:Image.Image, height_rectangle:int=150):
	"""
	Cette fonction extrait une bounding box autour d'une baseline. Elle affiche l'image pour l'isntant.
	:param target_line: La ligne, instance de classe OCRline
	:param height_rectangle:
	:param image:
	:return:
	"""
	shifted_lines = extend_line(target_line, 50)
	[[x1, y1], [x2, y2]] = [shifted_lines[0], shifted_lines[-1]]
	angle_ligne = get_angle(target_line)
	target_angle_pos = angle_ligne - math.radians(90)
	target_angle_neg = angle_ligne + math.radians(90)
	xa, ya = (x1 + height_rectangle * 0.5 * math.cos(target_angle_pos),
			  y1 + height_rectangle * 0.5 * math.sin(target_angle_pos))
	xb, yb = (x2 + height_rectangle * 0.5 * math.cos(target_angle_pos),
			  y2 + height_rectangle * 0.5 * math.sin(target_angle_pos))
	xd, yd = (x1 + height_rectangle * 0.5 * math.cos(target_angle_neg),
			  y1 + height_rectangle * 0.5 * math.sin(target_angle_neg))
	xc, yc = (x2 + height_rectangle * 0.5 * math.cos(target_angle_neg),
			  y2 + height_rectangle * 0.5 * math.sin(target_angle_neg))
	polygon = [[xa, ya], [xb, yb], [xc, yc], [xd, yd]]
	polygon_extraction(polygon, image)


def opencv_polygon_extraction(polygon, image:Union[Image.Image,np.array], keep_alpha:bool=True, return_image:bool=False, vertical_padding=None):
	# Convertir en np.ndarray si c'est une PIL.Image
	if isinstance(image, Image.Image):
		image = np.array(image)
		if image.ndim == 2:  # Niveaux de gris
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	elif image.ndim == 2:  # Déjà en niveaux de gris (NumPy)
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	# Créer un masque vide (uint8: 0-255)
	mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

	# Remplir le polygone dans le masque
	cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)

	# Calculer la bounding box
	x_coords = [p[0] for p in polygon]
	y_coords = [p[1] for p in polygon]
	x_min, x_max = max(0, min(x_coords) - vertical_padding), min(image.shape[1], max(x_coords) + vertical_padding)
	y_min, y_max = max(0, min(y_coords) - vertical_padding), min(image.shape[0], max(y_coords) + vertical_padding)

	# Recadrer l'image et le masque
	cropped_img = image[y_min:y_max, x_min:x_max]
	cropped_mask = mask[y_min:y_max, x_min:x_max]

	# Ajouter le canal alpha si nécessaire
	if keep_alpha:
		if cropped_img.shape[2] == 3:  # RGB → RGBA
			cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)
		cropped_img[:, :, 3] = cropped_mask  # Appliquer le masque comme alpha


	if return_image:
		return cropped_img
	else:
		if show_image:
			cropped_img = Image.fromarray(cropped_img)
			cropped_img.show()
		return None




def polygon_extraction(polygon, image:Union[Image.Image,np.array], keep_alpha:bool=True, return_image:bool=False, vertical_padding=None):
	"""
	Cette fonction extrait un polygone d'une image et la montre
	https://stackoverflow.com/a/22650239
	:return:
	"""
	# read image as RGB and add alpha (transparency)
	# convert to numpy (for convenience)
	if isinstance(image, Image.Image):
		imArray = np.asarray(image)
	else:
		imArray = image

	# create mask
	maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
	# if isinstance(polygon, np.array):
	# 	polygon = polygon[0].tolist()

	PIL.ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
	mask = np.array(maskIm)

	# assemble new image (uint8: 0-255)
	newImArray = np.empty(imArray.shape, dtype='uint8')

	# colors (three first columns, RGB)
	newImArray[:, :, :3] = imArray[:, :, :3]

	# transparency (4th column)
	if keep_alpha is True:
		newImArray[:, :, 3] = mask * 255

	# back to Image from numpy
	x_coords = [point[0] for point in polygon]
	y_coords = [point[1] for point in polygon]
	x_min, x_max = min(x_coords), max(x_coords)
	y_min, y_max = min(y_coords), max(y_coords)

	if vertical_padding:
		x_min, x_max = x_min - vertical_padding, x_max + vertical_padding
	rectangle_coordinates = (x_min, y_min, x_max, y_max)

	# On enregistre
	if keep_alpha is True:
		mode = "RGBA"
	else:
		mode = "RGB"
	newIm = Image.fromarray(newImArray, mode)
	cropped_img = newIm.crop(rectangle_coordinates)
	if return_image is True:
		return cropped_img
	else:
		cropped_img.show()


def batch_alto_line_to_img_cv2(coordinates, loaded_im, vertical_padding=None, keep_alpha=False):
	# Créer un masque vide (uint8: 0-255)
	mask = np.zeros((loaded_im.shape[0], loaded_im.shape[1]), dtype=np.uint8)

	# Remplir le polygone dans le masque


	cv2.fillPoly(mask, [np.array(coordinates, dtype=np.int32)], 255)
	# Calculer la bounding box
	x_coords = [p[0] for p in coordinates]
	y_coords = [p[1] for p in coordinates]
	x_min, x_max = max(0, min(x_coords) - vertical_padding), min(loaded_im.shape[1], max(x_coords) + vertical_padding)
	y_min, y_max = max(0, min(y_coords) - vertical_padding), min(loaded_im.shape[0], max(y_coords) + vertical_padding)

	# Recadrer l'image et le masque
	cropped_img = loaded_im[y_min:y_max, x_min:x_max]
	cropped_mask = mask[y_min:y_max, x_min:x_max]

	# Ajouter le canal alpha si nécessaire
	if keep_alpha:
		if cropped_img.shape[2] == 3:  # RGB → RGBA
			cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)
		cropped_img[:, :, 3] = cropped_mask  # Appliquer le masque comme alpha

	as_img = Image.fromarray(cropped_img)
	return as_img



def extend_baseline_and_retranscribe(line: OCRLine,
									 image_path: str,
									 ocr_model) -> OCRLine:
	image = Image.open(image_path)
	# draw_lines_on_image(image_path=image_path, baseline=[line.baseline])
	shifted_lines = extend_line(line, 50)
	polygons = KRAKEN.blla.calculate_polygonal_environment(im=image, baselines=[shifted_lines])
	# draw_lines_on_image(image_path=image_path, baseline=[shifted_lines])
	baselinelines = [containers.BaselineLine(id="test",
								  baseline=shifted_lines,
								  boundary=polygons[0])]
	mysegmentation = containers.Segmentation(imagename="test",
											 script_detection=False,
											 lines=baselinelines,
											 regions=None,
											 text_direction='horizontal-lr',
											 type='baselines')
	kraken_ocr = KRAKEN.KRAKEN(segmentation_model=None,
							   ocr_model=ocr_model)
	return kraken_ocr.predict_with_kraken(im=image, segments=mysegmentation)

def find_best_transcription(lines: OCRRecord,
							image_path: str,
							step: int,
							ranges: tuple,
							ocr_model) -> OCRRecord:
	'''
	Cette fonction va prendre un ensemble de lignes et les déplacer de quelques pixels jusqu'à trouver la meilleure transcription
	:return: la meilleure transcription
	'''
	all_records = []
	lexicality_indices = []
	log_print(f"Before shift: {lines.join_transcription()}")
	log_print(f"La glose fait {len(lines)} lignes.")
	orig_transcription = lines.join_transcription()
	words = set([remove_accents(word).lower() for word in txt_to_list("src/resources/french_lexicon.txt") if
				 not word.isupper()])
	lexicality = compute_lexicality(orig_transcription, words)
	all_records.append(lines)
	lexicality_indices.append(lexicality)

	# On essaie d'améliorer l'OCR en déplaçant de quelques pixels chaque ligne dans la direction orthogonale.
	# En faire une fonction
	# utils.draw_lines_on_image(image_path, baseline=[item.baseline for item in as_record])

	# D'abord on va faire les transcriptions en déplacant les baselines peu à peu, puis recalculant les polygones

	image = Image.open(image_path)
	for shift in range(ranges[0], ranges[1], step):
		shifted_lines = shift_lines(lines, shift)
		# draw_lines_on_image(image_path, baseline=shifted_lines)
		polygons = KRAKEN.blla.calculate_polygonal_environment(im=image, baselines=shifted_lines)
		baselinelines = []
		for baseline, polygon in zip(shifted_lines, polygons):
			bll = containers.BaselineLine(id="test",
										  baseline=baseline,
										  boundary=polygon)
			baselinelines.append(bll)

		mysegmentation = containers.Segmentation(imagename="test",
												 script_detection=False,
												 lines=baselinelines,
												 regions=None,
												 text_direction='horizontal-lr',
												 type='baselines')

		kraken_ocr = KRAKEN.KRAKEN(segmentation_model=None,
								   ocr_model=ocr_model)
		transcription = kraken_ocr.predict_with_kraken(im=image, segments=mysegmentation)
		log_print("---")
		log_print(f"Current shift: {shift}")
		log_print(f"Current transcription: {transcription.join_transcription()}")
		all_records.append(transcription)
		transcription = transcription.join_transcription()
		lexicality = compute_lexicality(transcription, words)
		log_print(f"Transcription: {transcription}")
		log_print(f"Lexicalité: {lexicality}")
		lexicality_indices.append(lexicality)

	best_transcription = all_records[lexicality_indices.index(min(lexicality_indices))]
	log_print(f"Best transcription: {best_transcription.join_transcription()}")

	return best_transcription


def txt_to_list(path) -> list:
	with open(path, 'r') as f:
		as_string = f.read().split("\n")
	normalized = [nfc_normalize(item) for item in as_string]
	normalized = [item.lower() for item in normalized]
	return normalized


def remove_accents(string):
	string = nfc_normalize(string)
	string = string.replace("é", "e").replace("à", "a").replace("è", "e").replace("â", "a").replace("ê", "e").replace(
		"ô", "o")
	return string


def compute_lexicality(string: str, words: set) -> float:
	"""
	Cette fonction va appliquer à une chaîne de caractère un indice de lexicalité, et trier par ordre ascendant la liste produite.
	:return: le taux de lexicalité de la phrase
	"""

	split_regexp = re.compile("[.«?!:()–⟦⟧\[\]+,;\-\s+\d+\"'»]")
	normalized_string = nfc_normalize(string)
	normalized_string = remove_accents(normalized_string).lower()
	# nerd = ner(all_lines_as_string)
	# for entity in reversed(nerd):
	# 	if entity['entity_group'] != "PER":
	# 		continue
	# 	start = entity['start']
	# 	end = entity["end"]
	# 	as_list = [char for char in all_lines_as_string]
	# 	del as_list[start:end]
	# 	all_lines_as_string = "".join(as_list)

	splitted = [item for item in re.split(split_regexp, normalized_string) if item not in ["", None]]
	filtered_split = [item.lower() for item in splitted if "^" not in item]
	filtered_split = [item for item in filtered_split if len(item) > 3]
	vocab = set(filtered_split)

	# On identifie tous les mots qui ne sont pas dans le vocabulaire
	comparison = vocab - words
	try:
		error_rate = len(comparison)
	except ZeroDivisionError:
		return 0
	return error_rate

def extend_line(line: OCRLine, pixels: int):
	"""
	Shift line left and right by n pixels
	:param line:
	:param pixels:
	:return: la nouvelle baseline
	"""
	first_point, last_point = line.baseline[0], line.baseline[-1]
	a, b = produce_line_function([first_point[0], first_point[1], last_point[0], last_point[1]])

	# On adapte la taille du shift en fonction de la pente.
	# pixels = pixels / (1 if a == 0 else a)
	extended_point_left_x = first_point[0] - pixels
	extended_point_right_x = last_point[0] + pixels
	extended_point_right_y = a*(extended_point_right_x) + b
	extended_point_left_y = a*(extended_point_left_x) + b
	new_baseline = [[extended_point_left_x, extended_point_left_y], [extended_point_right_x, extended_point_right_y]]
	return new_baseline

def shift_lines(lines_as_record: OCRRecord, pixels_shift: int):
	"""
	Cette fonction vise à déplacer un ensemble de lignes d'un certain nombre de pixels dans une direction donnée, représentée sous la forme d'un angle
	par rapport à l'axe des abcisses.
	:param lines_as_record:
	:param pixels_shift: de combien de pixels il faut déplacer les lignes
	:param angle: la direction du déplacement
	:return: une liste de lignes de base qu'il faudra re-transformer en polygones.
	"""

	new_lines = []
	for line in lines_as_record:
		baseline = line.baseline
		new_baseline = []
		current_angle = get_angle(line)
		for x1, y1 in baseline:
			# On cherche la direction orthogonale
			target_angle = current_angle + math.radians(90)
			# https://stackoverflow.com/a/48525695
			x2, y2 = (x1 + pixels_shift * math.cos(target_angle), y1 + pixels_shift * math.sin(target_angle))
			if x2 < 0:
				x2 = 0
			if y2 < 0:
				y2 = 0
			new_baseline.append([round(x2), round(y2)])
		# log_print("---")
		# log_print(f"Current point: {x1, y1}")
		# log_print(f"Shifted point: {x2, y2}")
		new_lines.append(new_baseline)

	return new_lines


def sort_lines_with_rotation(lines_as_record: OCRRecord, zone: namedtuple):
	"""
	Cette fonction réordonne les lignes d'un ajout en les redressant selon un angle calculé par moyenne des angles de toutes les lignes
	et un centre de rotation qui est le centre de la zone identifiée.
	:param lines_as_record:
	:param zone:
	:return:
	"""
	angle = get_average_angle(lines_as_record)
	rectangle_center = get_center_of_rectangle(zone)
	rotated_lines = []
	for line in lines_as_record:
		# On va redresser la ligne en prenant le centre de la zone comme point de référence et l'angle moyen identifié.
		rotated_point_a = rotate(rectangle_center, line.baseline[0], - angle)
		rotated_point_b = rotate(rectangle_center, line.baseline[-1], - angle)
		rotated_lines.append({"original_line": line, "rotated": [rotated_point_a, rotated_point_b]})
	sorted_lines = sorted(rotated_lines, key=lambda x: x['rotated'][0][1])
	sorted_lines = [item["original_line"] for item in sorted_lines]
	new_record = OCRRecord()
	new_record.recreate_record(sorted_lines)
	return new_record


def vertical_order_lines(lines: OCRRecord) -> OCRRecord:
	"""
	Cette fonction trie les lignes de façon verticale (de haut en bas). Elle suppose un filtre
	préalable des lignes au sein des zones pour être efficace
	:param lines: la liste de dictionnaires (baseline, prediction, cuts)
	:return: la liste ordonnée
	"""
	sorted_list = sorted(lines, key=lambda x: x.baseline[0][1])
	new_record = OCRRecord()
	new_record.recreate_record(sorted_list)
	return new_record


def vertical_order_zones(annotations: YOLORecord) -> YOLORecord:
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
	sorted_list = sorted(annotations, key=lambda x: x.coordinates[0][1])
	return sorted_list


def rectanglify(coords):
	return


def horizontal_order_zones(annotations: YOLORecord) -> YOLORecord:
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
	sorted_list = sorted(annotations, key=lambda x: x.coordinates[0][0])
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


def normalize_string_and_strip_spaces(string: str) -> str:
	"""
	Cette fonction neutralise la casse et supprime les espace de début et fin de chaîne
	:param string: le texte à normaliser
	:return: le texte normalisé
	"""
	return string.lower().strip()


def merge_adjacent_entities(tokens: list[dict]):
	"""
	Cette fonction fusionne les entités adjacentes de même classe, qui ne sont qu'une entité.
	:param entities: une liste d'entité produite par une pipeline.
	:return: la liste avec les entités fusionnées
	"""
	entites = []
	i = 0
	n = len(tokens)
	while i < n:
		token = tokens[i]
		if token["entity"].startswith("B-"):
			# Début d'une nouvelle entité
			entite = {
				"entity": token["entity"][2:],  # Enlever le préfixe "B-"
				"word": token["word"],
				"start": token["start"],
				"end": token["end"]
			}
			# Chercher les tokens suivants de type "I-XXX"
			j = i + 1
			while j < n and tokens[j]["entity"] == f"I-{entite['entity']}":
				entite["word"] += " " + tokens[j]["word"]
				entite["end"] = tokens[j]["end"]
				j += 1
			entites.append(entite)
			i = j
		else:
			i += 1
	return entites


def entities_to_dict(entities: list) -> dict:
	result = {}
	for entity in entities:
		label = entity["entity_group"]
		word = entity["word"]
		spans = [entity["start"], entity["end"]]
		try:
			result[label].append({"string": word,
								  "spans": spans})
		except KeyError:
			result[label] = [{"string": word, "spans": spans}]

	return result


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


def convert_baseline_coordinates_to_alto(coords):
	coords = [[str(point) for point in coord] for coord in coords]
	converted = " ".join([" ".join(item) for item in coords])
	return converted

def convert_alto_coordinates_to_baseline(coords):
	splits = coords.split()
	splits = [int(item) for item in splits]
	converted = [[splits[idx], splits[idx + 1]] for idx in range(0, len(splits) - 1, 2)]
	return converted

def match_lines_in_zones(ocr_prediction: OCRRecord,
						 zone_as_rectangle: namedtuple,
						 intersect_ratio=0.5) -> OCRRecord:
	"""
	Cette fonction identifie toutes les lignes qui traversent une boîte
	:param ocr_prediction: un objet de classe OCRPrediction. les lignes comme une liste de dictionnaires (baseline, prediction, cuts)
	:param zone_as_rectangle: la boîte
	:param intersect_ratio: la proportion minimale de la ligne comprise dans la boîte
	:return: une liste avec les lignes filtrées
	"""
	corresponding_lines = []
	for idx, line in enumerate(ocr_prediction):
		baseline = line.baseline

		# Si la ligne de base comprend plus d'un point, on simplifie en prenant les extrémités
		converted_baseline = [baseline[0][0], baseline[0][1], baseline[-1][0], baseline[-1][1]]
		is_in_box = check_if_line_in_box(box_coord=zone_as_rectangle,
										 baseline=converted_baseline,
										 intersect_ratio=intersect_ratio,
										 record=line)

		if is_in_box is True:
			corresponding_lines.append(line)
	return corresponding_lines


def draw_rectangle(image, rectangle, return_image=False):
	log_print("Attempting to show image")
	draw = PIL.ImageDraw.Draw(image)
	draw.rectangle(rectangle, outline="red", width=10)
	if return_image is False:
		image.show()
	else:
		return image


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
	stopwords = re.compile(r"^du |^de la |^de |^[àa] |y |de l'")
	clean = re.sub(stopwords, "", string)
	return clean


def remove_all_punctuation(string: str, debug=False) -> str:
	"""
	Cette fonction supprime la ponctuation en début et fin de chaîne
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	orig_string = string
	expression = r"[\(\),;.!?\-:]"
	punct_regexp = re.compile(expression)
	string = string.strip()
	string = re.sub(punct_regexp, " ", string)
	string = string.strip()
	if debug:
		log_print(f"|{orig_string}| -> |{string}|")
	return string


def strip_punctuation(string: str | None, debug=False) -> str | None:
	"""
	Cette fonction supprime la ponctuation en début et fin de chaîne
	:param string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	if string is None:
		return None
	orig_string = string
	punctuation = r"[\(\),;.!?\-:]"
	expression = r"^" + punctuation + r"\s{0,}|\s{0,}" + punctuation + r"$"
	punct_regexp = re.compile(expression)
	string = string.strip()
	string = re.sub(punct_regexp, "", string)
	string = string.strip()
	if debug:
		log_print(f"|{orig_string}| -> |{string}|")
	return string


def convert_to_csv(extractions: dict, outpath: str):
	extracted_data = []
	header = ["Numero_minute",
			  "Images concernées",
			  "Erreur de traitement de la minute",
			  "Plusieurs soldats",
			  "Date du procès",
			  "Jours de guerre",
			  "Mois de guerre",
			  "Trimestres de guerre",
			  "Années de guerre",
			  "Institution engagée",
			  "Lieu du procès",
			  "Numéro du jugement",
			  "Numéro d'ordre",
			  "Président du jury (extrait)",
			  "Président du jury (normalisé)",
			  "Juré 1 (extrait)",
			  "Juré 1 (normalisé)",
			  "Juré 2 (extrait)",
			  "Juré 2 (normalisé)",
			  "Juré 3 (extrait)",
			  "Juré 3 (normalisé)",
			  "Juré 4 (extrait)",
			  "Juré 4 (normalisé)",
			  "Greffier",
			  "Commissaire",
			  "Général nommant",
			  "Date du crime ou du délit",
			  "Nom",
			  "Prénoms",
			  "Date de naissance",
			  "age",
			  "Taille",
			  "Cheveux",
			  "Front",
			  "Yeux",
			  "Nez",
			  "Visage",
			  "Renseignements complémentaires",
			  "Marques particulières",
			  "Naissance hors métropole",
			  "Naissance à l'étranger",
			  "Pays de naissance",
			  "Ville de naissance transcrite",
			  "Ville de naissance - nom actuel",
			  "Ville de naissance - nom 1999",
			  "Ville de naissance - nom 1801",
			  "Latitude lieu naissance",
			  "Longitude lieu naissance",
			  "Arrondissement de naissance",
			  "Département de naissance transcrit",
			  "Département de naissance",
			  "Résidence hors métropole",
			  "Résidence à l'étranger",
			  "Pays de résidence",
			  "Ville de résidence transcrite",
			  "Ville de résidence - nom actuel",
			  "Ville de résidence - nom 1999",
			  "Ville de résidence - nom 1801",
			  "Latitude lieu résidence",
			  "Longitude lieu résidence",
			  "Arrondissement de résidence",
			  "Département de résidence transcrit",
			  "Département de résidence",
			  "Situation maritale",
			  "Enfants",
			  "Profession extraite",
			  "Profession normalisée",
			  "Catégorie socioprofessionnelle",
			  "Rang du soldat (extrait)",
			  "Rang du soldat (normalisé)",
			  "Affectation du soldat",
			  "Numéro de matricule",
			  "Chef d'accusation extrait",
			  "Chef d'accusation normalisé",
			  "Antécédents",
			  "Condamnation (extraite)",
			  "Condamnation (normalisée)",
			  "Sursis",
			  "Vote transcrit",
			  "Vote extrait",
			  "Voix (vote majoritaire)",
			  "Peine transcrite",
			  "Type de peine",
			  "Nombre de mois",
			  "Frais du procès"]
	for idx_minute, minute in extractions.items():
		interm = []
		interm.append(f"{idx_minute}")

		if minute != {}:
			interm.append(f'{minute["metadata"]["images"][0].split("/")[-1].split(".")[0]}-{minute["metadata"]["images"][-1].split("/")[-1].split(".")[0]}')
		else:
			interm.append("UNK")
		if minute == {}:
			interm.append("True")
		else:
			interm.append("False")
		# Image


		# Plusieurs soldats
		try:
			plusieurs_soldats = "True" if minute["metadata"]["plusieurs_soldats"] is True else "False"
		except KeyError:
			plusieurs_soldats = "False"
		interm.append(plusieurs_soldats)
		# Date du procès
		try:
			date_proces = minute['informations_proces']['date_du_proces']['date_reconciliee']
		except (TypeError, KeyError):
			date_proces = "UNK"
		interm.append(date_proces)


		# Jours de guerre
		try:
			jours_de_guerre = minute['informations_proces']['date_du_proces']['jours_de_guerre']
		except (TypeError, KeyError):
			jours_de_guerre = "UNK"
		interm.append(jours_de_guerre)

		# mois de guerre
		try:
			mois_de_guerre = minute['informations_proces']['date_du_proces']['mois_de_guerre']
		except (TypeError, KeyError):
			mois_de_guerre = "UNK"
		interm.append(mois_de_guerre)

		# trimestres de guerre
		try:
			trimestres_de_guerre = minute['informations_proces']['date_du_proces']['trimestres_de_guerre']
		except (TypeError, KeyError):
			trimestres_de_guerre = "UNK"
		interm.append(trimestres_de_guerre)
		

		# annee de guerre
		try:
			annees_de_guerre = minute['informations_proces']['date_du_proces']['annees_de_guerre']
		except (TypeError, KeyError):
			annees_de_guerre = "UNK"
		interm.append(annees_de_guerre)

		# Lieu du procès
		try:
			institution = minute['informations_proces']['lieu_jugement']['institution']
		except (TypeError, KeyError):
			institution = "UNK"
		interm.append(institution)

		try:
			lieu_proces = minute['informations_proces']['lieu_jugement']['siège']
		except (TypeError, KeyError):
			lieu_proces = "UNK"
		interm.append(lieu_proces)

		# Numéro de jugement
		try:
			numero_jugement = minute['informations_proces']['numero_jugement']['extracted']
		except (TypeError, KeyError):
			numero_jugement = "UNK"
		interm.append(numero_jugement)

		# Numéro d'ordre
		try:
			numero_ordre = minute['informations_proces']['numero_ordre']['extracted']
		except  (KeyError, TypeError):
			numero_ordre = "UNK"
		interm.append(numero_ordre)

		# Président du jury (rôle non extrait)
		try:
			president = minute['informations_proces']['magistrats']['president']['extracted']['persName']
		except (KeyError, TypeError):
			president = "UNK"
		interm.append(president)
		try:
			president_normalise = minute['informations_proces']['magistrats']['president']['normalized']
		except (KeyError, TypeError):
			president_normalise = "UNK"
		interm.append(president_normalise)

		# Jurés (on n'extrait pas les rôles)
		try:
			jures = minute['informations_proces']['magistrats']['jures']
			for i in range(4):
				try:
					extracted_jure = jures[i]['extracted']['persName']
				except (TypeError, IndexError):
					extracted_jure = "UNK"
				try:
					normalized_jure = jures[i]['normalized']
				except (TypeError, IndexError):
					normalized_jure = "UNK"
				interm.append(extracted_jure)
				interm.append(normalized_jure)
		except (KeyError, TypeError):
			interm.extend(["UNK" for _ in range(4)])
			interm.extend(["UNK" for _ in range(4)])

		# Greffier (on n'extrait pas les rôles)
		try:
			greffier = minute['informations_proces']['magistrats']['greffier']['extracted']['persName']
		except  (KeyError, TypeError):
			greffier = "UNK"
		interm.append(greffier)

		# Commissaire du gouvernement (on n'extrait pas les rôles)
		try:
			commissaire = minute['informations_proces']['magistrats']['commissaire']['extracted']['persName']
		except  (KeyError, TypeError):
			commissaire = "UNK"
		interm.append(commissaire)

		# Général
		try:
			general = minute['informations_proces']['magistrats']['general']['extracted']
		except  (KeyError, TypeError):
			general = "UNK"
		interm.append(general)

		# Date du crime
		try:
			date_crime = minute['accusation']['date_du_crime_ou_delit']
		except  (KeyError, TypeError):
			date_crime = "UNK"
		if isinstance(date_crime, dict):
			date_crime = json.dumps(date_crime)
		interm.append(date_crime)

		# Nom et prénom du soldat
		try:
			prenoms_soldat = minute['soldat']['identite']['prenom']['extracted']
			nom_soldat = minute['soldat']['identite']['nom']['extracted']
		except  (KeyError, TypeError):
			nom_soldat = "UNK"
			prenoms_soldat = "UNK"
			interm.append(nom_soldat)
			interm.append(prenoms_soldat)
			extracted_data.append(interm)
			continue
		if isinstance(nom_soldat, list):
			nom_soldat = " ou ".join(nom_soldat)
		if isinstance(prenoms_soldat, list):
			prenoms_soldat = " ou ".join(prenoms_soldat)
		interm.append(nom_soldat)
		interm.append(prenoms_soldat)

		# Date de naissance et âge du soldat
		try:
			date_naissance = minute['soldat']['identite']['date_naissance']
		except (TypeError, KeyError):
			date_naissance = "UNK"
		try:
			age = minute['soldat']["identite"]["age"]
			age = age if age else "UNK"
		except  (KeyError, TypeError):
			age = "UNK"
		interm.append(date_naissance)
		interm.append(age)

		try:
			taille = minute['soldat']["description_physique"]["taille"]["extracted"]
		except  (KeyError, TypeError):
			taille = "UNK"
		try:
			cheveux = minute['soldat']["description_physique"]["cheveux"]["extracted"]
		except  (KeyError, TypeError):
			cheveux = "UNK"
		try:
			front = minute['soldat']["description_physique"]["front"]["extracted"]
		except  (KeyError, TypeError):
			front = "UNK"
		try:
			yeux = minute['soldat']["description_physique"]["yeux"]["extracted"]
		except  (KeyError, TypeError):
			yeux = "UNK"
		try:
			nez = minute['soldat']["description_physique"]["nez"]["extracted"]
		except  (KeyError, TypeError):
			nez = "UNK"
		try:
			visage = minute['soldat']["description_physique"]["visage"]["extracted"]
		except  (KeyError, TypeError):
			visage = "UNK"
		try:
			renseignements_complementaires = \
				minute['soldat']["description_physique"]["renseignements_complementaires"]["extracted"]
		except  (KeyError, TypeError):
			renseignements_complementaires = "UNK"
		try:
			marques_particulieres = minute['soldat']["description_physique"]["marques_particulières"][
				"extracted"]
		except  (KeyError, TypeError):
			marques_particulieres = "UNK"
		interm.append(taille)
		interm.append(cheveux)
		interm.append(front)
		interm.append(yeux)
		interm.append(nez)
		interm.append(visage)
		interm.append(renseignements_complementaires)
		interm.append(marques_particulieres)

		# Naissance hors métropole
		try:
			naissance_hors_metropole = minute['soldat']['identite']['lieu_naissance']['hors_metropole']
		except (KeyError, TypeError):
			naissance_hors_metropole = "UNK"
		interm.append(naissance_hors_metropole)

		# Naissance étranger
		try:
			naissance_etranger = minute['soldat']['identite']['lieu_naissance']['etranger']
		except (KeyError, TypeError):
			naissance_etranger = "UNK"
		interm.append(naissance_etranger)


		# Pays naissance
		try:
			pays_naissance = minute['soldat']['identite']['lieu_naissance']['pays']
		except (KeyError, TypeError):
			pays_naissance = "UNK"
		interm.append(pays_naissance)


		# Lieu de naissance
		if minute['soldat']['identite']['lieu_naissance']:
			try:
				ville_naissance_transcrite = minute['soldat']['identite']['lieu_naissance']['ville'][
					'extracted']
				ville_naissance_actuelle = minute['soldat']['identite']['lieu_naissance']['ville'][
					'nom_actuel']
			except (KeyError, TypeError):
				ville_naissance_actuelle = "UNK"
				ville_naissance_transcrite = "UNK"
			except  (KeyError, TypeError):
				try:
					ville_naissance_actuelle = minute['soldat']['identite']['lieu_naissance']['ville'][
						'extracted']
				except  (KeyError, TypeError):
					ville_naissance_actuelle = "UNK"
				ville_naissance_1999 = "UNK"
				ville_naissance_1801 = "UNK"
			try:
				latitude_ville_naissance = minute['soldat']['identite']['lieu_naissance']["coordonnées"][
					"lat"]
				longitude_ville_naissance = minute['soldat']['identite']['lieu_naissance']["coordonnées"][
					"lon"]
			except  (KeyError, TypeError):
				latitude_ville_naissance, longitude_ville_naissance = "UNK", "UNK"
			try:
				arrondissement_naissance = minute['soldat']['identite']['lieu_naissance'][
				'arrondissement']['extracted']
			except  (KeyError, TypeError):
				arrondissement_naissance = None
			try:
				departement_naissance = minute['soldat']['identite']['lieu_naissance'][
					'departement']['extracted']
			except  (KeyError, TypeError):
				departement_naissance = None
			try:
				departement_naissance_transcrit = minute['soldat']['identite']['lieu_naissance'][
					'departement']['corrected']
			except  (KeyError, TypeError):
				departement_naissance_transcrit = None
			try:
				ville_naissance_1999 = minute['soldat']['identite']['lieu_naissance']['ville']['nom_1999']
			except (TypeError, KeyError):
				ville_naissance_1999 = None
			try:
				ville_naissance_1801 = minute['soldat']['identite']['lieu_naissance']['ville']['nom_1801']
			except (TypeError, KeyError):
				ville_naissance_1801 = None
		else:
			ville_naissance_transcrite = None
			ville_naissance_actuelle = None
			ville_naissance_1999 = None
			ville_naissance_1801 = None
			latitude_ville_naissance = None
			longitude_ville_naissance = None
			arrondissement_naissance = None
			departement_naissance = None
			departement_naissance_transcrit = None
		interm.append(ville_naissance_transcrite)
		interm.append(ville_naissance_actuelle)
		interm.append(ville_naissance_1999)
		interm.append(ville_naissance_1801)
		interm.append(latitude_ville_naissance)
		interm.append(longitude_ville_naissance)
		interm.append(arrondissement_naissance)
		interm.append(departement_naissance)
		interm.append(departement_naissance_transcrit)

		# Lieu de résidence
		try:
			ville_residence_transcrite = minute['soldat']['identite']['lieu_residence']['ville'][
				'extracted']
		except  (KeyError, TypeError):
			ville_residence_transcrite = None
		try:
			ville_residence_actuelle = minute['soldat']['identite']['lieu_residence']['ville'][
				'nom_actuel']
			ville_residence_1999 = minute['soldat']['identite']['lieu_residence']['ville']['nom_1999']
			ville_residence_1801 = minute['soldat']['identite']['lieu_residence']['ville']['nom_1801']
		except  (KeyError, TypeError):
			ville_residence_actuelle = "UNK"
			ville_residence_1999 = "UNK"
			ville_residence_1801 = "UNK"
		except  (KeyError, TypeError):
			ville_residence_actuelle = minute['soldat']['identite']['lieu_residence']['ville'][
				'extracted']
			ville_residence_1999 = ville_residence_actuelle
			ville_residence_1801 = ville_residence_actuelle

		try:
			latitude_ville_residence = minute['soldat']['identite']['lieu_residence']["coordonnées"][
				"lat"]
			longitude_ville_residence = minute['soldat']['identite']['lieu_residence']["coordonnées"][
				"lon"]
		except (KeyError, TypeError):
			latitude_ville_residence, longitude_ville_residence = "UNK", "UNK"
		try:
			arrondissement_residence = minute['soldat']['identite']['lieu_residence'][
				'arrondissement']['extracted']
		except  (KeyError, TypeError):
			arrondissement_residence = "UNK"
		try:
			departement_residence_transcrit = minute['soldat']['identite']['lieu_residence'][
			'departement']['extracted']
		except (KeyError, TypeError):
			departement_residence_transcrit = "UNK"
		try:
			departement_residence = minute['soldat']['identite']['lieu_residence'][
			'departement']['corrected']
		except  (KeyError, TypeError):
			departement_residence = "UNK"
			

		# residence hors métropole
		try:
			hors_metropole = minute['soldat']['identite']['lieu_residence']['hors_metropole']
		except (KeyError, TypeError):
			hors_metropole = "UNK"
		interm.append(hors_metropole)
		# residence étranger
		try:
			residence_etranger = minute['soldat']['identite']['lieu_residence']['etranger']
		except (KeyError, TypeError):
			residence_etranger = "UNK"
		interm.append(residence_etranger)
		

		# Pays residence
		try:
			pays_residence = minute['soldat']['identite']['lieu_residence']['pays']
		except (KeyError, TypeError):
			pays_residence = "UNK"
		interm.append(pays_residence)
		
		interm.append(ville_residence_transcrite)
		interm.append(ville_residence_actuelle)
		interm.append(ville_residence_1999)
		interm.append(ville_residence_1801)
		interm.append(latitude_ville_residence)
		interm.append(longitude_ville_residence)
		interm.append(arrondissement_residence)
		interm.append(departement_residence_transcrit)
		interm.append(departement_residence)

		# Femme et enfants
		situation_maritale = minute['soldat']["identite"]["famille"]['situation_maritale']
		if situation_maritale:
			pass
		else:
			situation_maritale = "célibataire"
		interm.append(situation_maritale)
		try:
			enfants = minute['soldat']["identite"]["famille"]['enfants']
			if enfants is None:
				enfants = 0
			interm.append(enfants)
		except (KeyError, TypeError):
			interm.append(0)

		# Profession
		try:
			profession_extraite = minute['soldat']["identite"]['profession_extraite']
			if isinstance(profession_extraite, list):
				profession_extraite = " ou ".join(profession_extraite)
			interm.append(profession_extraite)
		except  (KeyError, TypeError):
			interm.append(None)
			
		try:
			profession_normalisee = minute['soldat']["identite"]['profession_normalisee']
			if isinstance(profession_normalisee, list):
				profession_normalisee = " ou ".join(profession_normalisee)
			interm.append(profession_normalisee)
		except  (KeyError, TypeError):
			interm.append(None)

		# catégorie sociopro
		try:
			categorie_sociopro = minute['soldat']["identite"]['categorie_socioprofessionnelle']
			interm.append(categorie_sociopro)
		except  (KeyError, TypeError):
			interm.append("UNK")

		# Rang du soldat
		try:
			rang_extrait = minute['soldat']["situation_militaire"]['rang_extrait']
		except (TypeError, KeyError):
			rang_extrait = None
		interm.append(rang_extrait)
		# Rang du soldat
		try:
			rang_normalise = minute['soldat']["situation_militaire"]['rang_normalise']
		except (TypeError, KeyError):
			rang_normalise = None
		interm.append(rang_normalise)

		# Affectation du soldat

		try:
			affectation = minute['soldat']["situation_militaire"]['affectation']
		except (TypeError, KeyError):
			affectation = None
		interm.append(affectation)

		# Numéro de matricule
		try:
			matricule = minute['soldat']["situation_militaire"]['matricule']
		except:
			matricule = None
		interm.append(matricule)

		# Chef d'accusation extrait
		try:
			chef_accusation_extrait = minute['accusation']['chef_accusation_extrait']
		except  (KeyError, TypeError):
			chef_accusation_extrait = "UNK"
		interm.append(chef_accusation_extrait)

		# Chef d'accusation normalise
		try:
			chef_accusation_normalise = minute['accusation']['chef_accusation_normalise']
		except  (KeyError, TypeError):
			chef_accusation_normalise = "UNK"
		interm.append(chef_accusation_normalise)

		# Antécédent (juste le nombre)
		try:
			antecedents = len(minute['soldat']['antecedents'])
		except  (KeyError, TypeError):
			antecedents = "UNK"
		interm.append(antecedents)

		# Décision du tribunal
		try:
			decision_normalisee = minute['decision_tribunal']["jugement"]["decision_normalisee"]
		except (KeyError, TypeError):
			decision_normalisee = "UNK"
		try:
			decision_extraite = minute['decision_tribunal']["jugement"]["decision_extraite"]
		except (KeyError, TypeError):
			decision_extraite = "UNK"

		try:
			sursis = minute['decision_tribunal']["jugement"]["sursis"]
		except  (KeyError, TypeError):
			sursis = "UNK"
		except  (KeyError, TypeError):
			sursis = False

		try:
			vote = minute['decision_tribunal']["jugement"]["vote"]
		except (KeyError, TypeError):
			vote = "UNK"

		try:
			transcription_vote = minute['decision_tribunal']["jugement"]["voix"]["predicted"]
		except (KeyError, TypeError):
			transcription_vote = "UNK"

		try:
			voix = minute['decision_tribunal']["jugement"]["voix"]["extracted"]
		except (KeyError, TypeError):
			voix = None
		try:
			peine_transcrite = minute['decision_tribunal']["jugement"]["peine"]["predicted"]["extracted"]
		except (KeyError, TypeError):
			peine_transcrite = "UNK"
		try:
			type_peine = minute['decision_tribunal']["jugement"]["peine"]["extracted"]["type"]
		except (KeyError, TypeError):
			type_peine = "UNK"
		try:
			duree_peine = minute['decision_tribunal']["jugement"]["peine"]["extracted"]["duree"]
		except (KeyError, TypeError):
			duree_peine = "UNK"

		interm.append(decision_extraite)
		interm.append(decision_normalisee)
		interm.append(sursis)
		interm.append(transcription_vote)
		interm.append(vote)
		interm.append(voix)
		interm.append(peine_transcrite)
		interm.append(type_peine)
		interm.append(duree_peine)

		# Frais
		try:
			frais = minute['decision_tribunal']['frais']
		except  (KeyError, TypeError):
			frais = "UNK"
		interm.append(frais)

		extracted_data.append(interm)

	df = pd.DataFrame(extracted_data, columns=header)
	examples_number = df.shape[1]
	counts = []
	ratios = []
	for item in df.columns.values:
		missing_value = len(df[df[item] == 'UNK'])
		counts.append(missing_value)
		ratios.append(round(missing_value / examples_number, 2))
	df.loc[-1] = counts
	df.loc[-1] = ratios
	df.to_csv(outpath, sep='$', index=False)


def random_string():
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def extract_string_from_bbox(box:YOLOZone, lines: OCRRecord) -> str:
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
	out_string = ""
	(xmin, ymin), (xmax, ymax) = box.coordinates
	polygon_soldat = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
	for line in lines:
		# Solution tirée de https://gis.stackexchange.com/a/90063
		for char, cut in zip(line.prediction, line.cuts):
			cut = Polygon([tuple(coords) for coords in cut])
			intersection = polygon_soldat.intersects(cut)
			if intersection:
				out_string += char
		return out_string


def test_number_of_zones(annotations: YOLORecord, label: str, number: int) -> bool:
	"""
	Cette fonction permet de vérifier si les annotations YOLO contiennent le nombre de zones attendues
	:param annotations: La liste des annotations (dictionnaire {label, coordinates})
	:param label: le label à vérifier
	:param number: le nombre de zones attendues
	:return:
	"""
	filtered_list = [item for item in annotations if item.label == label]
	if len(filtered_list) == number:
		return True
	else:
		return False


def similarite_ratcliff(string_a, string_b):
	string_a = nfc_normalize(string_a)
	string_b = nfc_normalize(string_b)
	return SequenceMatcher(None, string_a, string_b).ratio()


def check_neant(string: str) -> bool:
	"""
	Cette fonction vérifie si le terme `néant` se trouve dans une chaîne
	:param string:
	:return: True/False
	"""
	if similarite_ratcliff(string, "neant") > 0.5:
		return True
	else:
		return False


def clean_small_string(input_string):
	"""
	Cette fonction supprime les ponctuations et les espaces
	de début et de fin de chaîne de caractère.
	:param input_string: la chaîne à nettoyer
	:return: la chaîne nettoyée
	"""
	if input_string is None:
		return None
	regexp = re.compile("^\s?[(.:,?\-;!]?\s?|\s?[.:,?\-;!)]?\s?$")
	return re.sub(regexp, "", input_string)


def approximate_word_split(sentence: str,
						   word: str,
						   sensibility: float = 0.5,
						   return_word: bool = False) -> list | \
														 tuple[
															 list, str] | None | \
														 tuple[
															 None, None]:
	"""
	Cette fonction découpe une phrase selon un mot qui peut être approximatif. Le mot n'est pas retourné.
	:param sentence: la phrase à découper
	:param word: le mot sur lequel s'appuyer
	:param sensibility: la sensibilité a appliquer à la recherche (à trouver par l'expérience)
	:return: La chaîne splittée, ou la chaîne splittée et le mot qui matche.
	"""
	sentence = nfc_normalize(sentence)
	word = nfc_normalize(word)
	match, matching_word = check_word_in_sentence(sentence, word, sensibility)
	if match:
		if return_word is True:
			return sentence.split(matching_word), matching_word
		else:
			return sentence.split(matching_word)
	else:
		if return_word is True:
			return None, None
		else:
			return None


def approximate_sentence_split(sentence: str, substring: str, max_dist: int = 1,
							   return_match: bool = False) -> list | None:
	"""
	Cette fonction découpe une phrase selon un groupe de mots qui peut être approximatif. Le mot n'est pas retourné.
	:param sentence: la phrase à découper
	:param substring: la sous-chaîne cible
	:param max_dist: la distance de levensthein maximale
	:param return_match: faut-il retourner la chaîne de caractère trouvée qui sert de délimiteur
	:return: La chaîne splittée ou None
	"""
	sentence = nfc_normalize(sentence)
	substring = nfc_normalize(substring)
	is_match, matches = check_substring_in_sentence(sentence=sentence,
													target_substring=substring,
													return_subtring=True,
													max_distance=max_dist)
	if is_match is True:
		longest_match = [(item.matched, len(item.matched)) for item in matches]
		longest_match.sort(key=lambda x: x[1], reverse=True)
		if return_match:
			return sentence.split(longest_match[0][0]), longest_match[0][0]
		else:
			return sentence.split(longest_match[0][0])
	else:
		return None


def delete_key(k, dic):
	"""
	Fonction qui supprime une clé récursivement dans un dictionnaire.
	Copié de https://stackoverflow.com/a/64815158
	:param k: la clé
	:param dic: le dictionnaire
	:return: le dictionnaire mis à jour
	"""
	if k in dic:
		del dic[k]
	for val in dic.values():
		if isinstance(val, dict):
			delete_key(k, val)
	return dic




def check_word_in_list(word_list: list, target_word: str, sensibility=0.7) -> (bool, str | None):
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
	max_dist: int = distances.index(max(distances))
	return True, matching_words[max_dist]


def check_substring_in_sentence(sentence: str,
								target_substring: str,
								return_subtring: bool = False,
								max_distance: int = 1) -> bool:
	"""
	Recherche d'une sous-chaîne dans une chaîne, avec distance possible.
	:param sentence: la phrase
	:param target_substring: le mot à chercher
	:param return_subtring: retourner la sous-chaîne trouvée ?
	:param max_distance: distance maximale autorisée
	:return:
	"""
	search = fuzzysearch.find_near_matches(target_substring,
										   sentence,
										   max_l_dist=max_distance)
	if len(search) == 0:
		match = False
	else:
		match = True
	if return_subtring is False:
		return match
	else:
		return match, search


def check_word_in_sentence(sentence: str, target_word: str | list, sensibility=0.5, debug: bool = False) -> tuple[
	bool, str | None]:
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
			# if debug is True:
			# 	log_print(word_lower)
			# 	log_print(dist)
			if dist > sensibility:
				distances.append(dist)
				matching_word.append(word)
	if len(distances) == 0:
		return False, False
	elif len(distances) > 1:
		log_print(f"Plus d'un mot trouvé, une erreur est possiblement survenue: {matching_word}."
			  f"On prend le dernier mot identifié.")
		# Dans ce cas on considère le dernier mot identifié, étant imprimé en fin de ligne.
		return True, matching_word[1]
	else:
		return True, matching_word[0]


def recursive_search(corresponding_lines: OCRRecord, string_to_match: str, n_gram: int):
	string_to_match = nfc_normalize(string_to_match)
	lines = [line.prediction for line in corresponding_lines]
	n_gram_lines = [((n, n + n_gram), nfc_normalize(" ".join(lines[n:n + n_gram]))) for n in range(len(lines))]
	inclusion_test = [string_to_match in line_group[1] for line_group in n_gram_lines]
	if any(inclusion_test):
		current_lines = next(idx for idx, item in enumerate(inclusion_test) if item is True)
		lines_range = n_gram_lines[current_lines][0]
		return corresponding_lines[slice(*lines_range)]
	else:
		if n_gram + 1 == len(corresponding_lines):
			return None
		result = recursive_search(corresponding_lines, string_to_match, n_gram + 1)
	return result


def match_line_by_substring(corresponding_lines: OCRRecord,
							string_to_match: str | list,
							return_index: bool = False,
							exact_match: bool = False) -> \
		tuple[OCRLine, list, int] | tuple[OCRLine, list] | OCRLine:
	"""
	Cette fonction extrait la ou les lignes qui contiennent une sous-chaîne la plus proche de la chaîne cible
	:param corresponding_lines: l'ensemble des lignes dans lesquelles chercher. Objet OCRRecord
	:param string_to_match: la chaîne à trouver ou une liste de chaines alternatives à identifier
	:param exact_match: Faut-il chercher la chaîne exacte ?
	:return: la ligne qui contient la chaîne de caractères et le zip de 1) la ligne et 2) la similarité avec la requête.
	Peut également retourner l'indice de l'item identifié dans la liste.
	"""

	if exact_match is True:
		# On va travailler avec des n-grams de ligne
		# Pour l'instant ne fonctionne que sur du exact matching
		return recursive_search(corresponding_lines=corresponding_lines,
								string_to_match=string_to_match,
								n_gram=1)
	else:
		distances = []
		for idx, ligne in enumerate(corresponding_lines):
			prediction = ligne.prediction
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
		debug_zip = list(zip([item.prediction for item in corresponding_lines], distances))
		if return_index is True:
			return name_line, debug_zip, correct_line_index
		else:
			return name_line, debug_zip


def levensthein_distance(string_a, string_b):
	return distance(string_a, string_b)

def weighted_levenshtein_distance(string_a, string_b):
	"""
	Distance de levensthein pondérée adaptée à l'HTR: il doit être plus coûteux d'ajouter ou de supprimer
	un caractère que d'effectuer une substitution
	:param string_a: la chaîne a
	:param string_b: la chaîne b
	:return: la distance pondérée.
	"""
	return rapidfuzz.distance.Levenshtein.distance(s1=string_a, s2=string_b, weights=(1.5, 1.5, 1))

def rectangle_to_baseline(rectangle):
	return [[rectangle.xmin, rectangle.ymin], [rectangle.xmax, rectangle.ymax]]


def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.
	Source: https://stackoverflow.com/a/75256388

	The angle should be given in degrees.
	"""
	ox, oy = origin
	px, py = point

	converted_angle = math.radians(angle)
	qx = ox + math.cos(converted_angle) * (px - ox) - math.sin(converted_angle) * (py - oy)
	qy = oy + math.sin(converted_angle) * (px - ox) + math.cos(converted_angle) * (py - oy)
	return [qx, qy]


def get_center_of_rectangle(rectangle):
	"""
	Cette fonction retourne la position du point central d'un rectangle
	:param rectangle:
	:return:
	"""
	center = (rectangle.xmin + ((rectangle.xmax - rectangle.xmin) / 2),
			  rectangle.ymin + ((rectangle.ymax - rectangle.ymin) / 2))
	return center


def expand_placename_abreviations(abbr_name):
	"""
	Cette fonction résoud les abréviations (sous/sur/saint)
	:return: l'abréviation résolue
	"""
	saint_regexp = re.compile(r"[Ss]\^t")

	# On résoud tout à "sur"
	sous_sur_regexp = re.compile(r"[sS]/")
	expanded = re.sub(saint_regexp, "Saint", abbr_name)
	expanded = re.sub(sous_sur_regexp, "sur", expanded)

	return expanded


def get_angle(line: OCRLine):
	"""
	Cette fonction calcule l'angle moyen d'une lignes par rapport aux abcisses, en radians.
	:param lines:
	:return:
	"""
	current_bl = [line.baseline[0], line.baseline[-1]]
	((aX, aY), (bX, bY)) = current_bl
	myradians = math.atan2(bY - aY, bX - aX)
	return myradians


def get_average_angle(lines: OCRRecord):
	"""
	Cette fonction calcule l'angle moyen d'un ensemble de lignes, en degrés.
	:param lines:
	:return:
	"""
	all_baselines = [[line.baseline[0], line.baseline[-1]] for line in lines]
	all_angles = []
	for current_bl in all_baselines:
		((aX, aY), (bX, bY)) = current_bl
		myradians = math.atan2(bY - aY, bX - aX)
		mydegrees = math.degrees(myradians)
		all_angles.append(mydegrees)
	average_angle = np.average(all_angles)
	return average_angle


def retrieve_substring_span(string: str, substring: str) -> list[int, int]:
	"""
	Cette fonction récupère la position d'un sous-chaîne étant donnée une chaîne de caractères
	"""
	return [string.find(substring), string.find(substring) + len(substring)]


def check_if_line_in_box(box_coord: namedtuple, baseline: list[int], intersect_ratio=.5, record=None) -> bool:
	"""
	Cette fonction vérifie si une ligne est comprise pour au moins 50% dans une zone.
	Présuppose des lignes globalement droites (= représentables par des fonctions affines).
	:param box_coord: les coordonnées de la zone [[x1, y1], [x2, y2]]
	:param baseline: les points de la ligne [x1, y1, x2, y2]
	:param intersect_ratio: la proportion de la ligne comprise dans la zone pour retourner vrai. Diminuer pour les petites zones.
	:return: Bool
	"""

	# On identifie la fonction qui représente la droite passant par les 2 points extrêmes de la ligne
	a, b = produce_line_function(baseline)
	# On regarde la distance horizontale entre ces deux points
	number_points = 20
	# La baseline est de forme [x1, y1, x2, y2]
	x_distance = round(baseline[-2] - baseline[0])
	steps = x_distance // number_points

	# On crée 20 points le long de la droite. Si la moitié sont dans la zone, on renvoie True
	if steps == 0:
		steps = x_distance / number_points
		n_points = [(round(item), round((a * item) + b)) for item in [baseline[0] + n*steps for n in range(number_points + 1)]]
	else:
		try:
			n_points = [(item, (a * item) + b) for item in range(baseline[0], baseline[-2], steps)]
		except ValueError as e:
			# log_print(f"La ligne est verticale.")
			steps = x_distance // number_points
			n_points = [(baseline[0], baseline[1] + (steps*item)) for item in range(20)]
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


def pickle_object(obj, path) -> None:
	"""
	Pickle un objet dans le chemin choisi
	:param obj:  l'objet à pickliser
	:param path: le chemin
	"""
	with open(path, "wb") as segmentation_as_file:
		pickle.dump(obj, segmentation_as_file, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_object(path):
	"""
	Unpickle un objet enregistré dans un chemin
	:param path: le chemin en question
	:return: l'objet cherché
	"""
	with open(path, "rb") as object:
		return pickle.load(object)


def save_as_dict(dictionnary: dict, path: str):
	with open(path, 'w') as f:
		# https://stackoverflow.com/a/36142844 default permet de gérer la sérialisation des objets bizarres (dates...)
		json.dump(dictionnary, f, indent=2, default=str)


def list_depth(lst: list) -> int:
	"""
	Retourne la profondeur maximale d'une liste de listes
	:param lst: la liste à analyser
	:return: un entier
	"""
	return isinstance(lst, list) and max(map(list_depth, lst)) + 1


def extraire_frais(chaine_caractere):
	regexp = re.compile("\^f\s?|[.,]\s?")
	split = re.split(regexp, chaine_caractere)
	if isinstance(split, str):
		split = chaine_caractere.split("^f ")
		split = [item.strip() for item in split]
	reconstructed = ".".join(split)
	try:
		as_float = float(reconstructed)
	except ValueError:
		return None
	return as_float


def correct_numbers_in_string(input_string):
	all_french_numbers = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze",
						  "douze", "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante", "cinquante",
						  "soixante", "cent", "mille"]
	corrected_string = []
	split = re.split(re.compile(r"\s+|-"), input_string)
	for word in split:
		if word in all_french_numbers:
			corrected_string.append(word)
		else:
			all_distances = []
			for correct_number in all_french_numbers:
				distance = levensthein_distance(word, correct_number)
				all_distances.append(distance)
			closest_match = all_french_numbers[all_distances.index(min(all_distances))]
			corrected_string.append(closest_match)

	return " ".join(corrected_string)


def sum_to_float(input: string) -> float:
	"""
	Cette fonction prend une somme (de frais par exemple) en toutes lettres et la convertit en flottant
	:return: le flottant correspondant
	"""
	input = input.lower().strip()
	split_francs = approximate_sentence_split(sentence=input, substring="francs")
	try:
		entiers = split_francs[0].strip()
	except TypeError:
		return None
	entiers = correct_numbers_in_string(entiers)
	try:
		centimes = approximate_sentence_split(split_francs[-1], substring="centimes")[0].strip()
	except TypeError:
		return {"value": entiers,
				"certainty": "low",
				"somme": input}
	centimes = correct_numbers_in_string(centimes)
	try:
		entiers = text_to_num.text2num(entiers, "fr")
	except ValueError:
		return None
	try:
		centimes = text_to_num.text2num(centimes, "fr")
	except ValueError:
		return {"value": entiers,
				"certainty": "low"}

	concat = f"{entiers}.{centimes}"
	as_float = float(concat)
	return {
		"value": as_float,
		"certainty": "high"
	}


def load_json_to_dict(path):
	try:
		with open(path, 'r') as f:
			return json.load(f)
	except json.decoder.JSONDecodeError:
		print(f"Error with file {path}")
		return None


def serialize_dict(dictionnaire, path) -> None:
	"""
	Sérialise un dictionnaire en fichier json
	:param dictionnaire: le dictionnaire en question
	:param path: Le chemin vers le fichier json
	"""
	with open(path, 'w') as f:
		json.dump(dictionnaire, f, indent=2, default=str)


def get_name_from_path(path):
	basename = path.split('/')[-1].split('.')[0]
	dossier = "_".join(basename.split('_')[:-1])
	ident = basename.split('_')[-1]
	return dossier, int(ident)


def format_coordinates(coords):
	rounded = [round(item) for item in coords]
	return [[rounded[0], rounded[1]], [rounded[2], rounded[3]]]


def longest_common_substring(string1, string2):
	"""
	Trouver la sous-chaîne commune la plus longue entre deux chaînes
	:param string1:
	:param string2:
	:return:
	"""
	# Source - https://stackoverflow.com/q/66162740
	len1, len2 = len(string1), len(string2)
	answer = ""
	for i in range(len1):
		for j in range(len2):
			lcs_temp = 0
			match = ''
			while ((i + lcs_temp < len1) and (j + lcs_temp < len2) and string1[i + lcs_temp] == string2[j + lcs_temp]):
				match += string2[j + lcs_temp]
				lcs_temp += 1
			if (len(match) > len(answer)):
				answer = match
	return answer


def get_baseline_from_string(line: OCRRecord | OCRLine,
							 target_string: str,
							 image_path: str,
							 loaded_image: Image.Image = None,
							 show_image: bool = False) -> tuple[tuple[int, int], tuple[int, int]] | None:
	"""
	Cette fonction récupère les coordonnées du fragment de baseline qui contient une chaîne de caractère donnée.
	Elle extrait plusieurs baselines si une chaîne court sur 2 lignes
	:param line: La ligne, objet OCRLine ou liste d'OCRLine
	:return: La ligne de base qui contient le texte : [[x_1, y_1], [x_2, y_2]]
	"""

	# Dans le cas où on a une liste de lignes, on boucle sur chacune d'entre elles, on récupère la portion de string
	# qui correspond et la baseline correspondant.
	if isinstance(line, OCRRecord) or isinstance(line, list):
		baselines = []
		for item in line:
			a, b = nfc_normalize(item.prediction), nfc_normalize(target_string)
			result = longest_common_substring(string1=a, string2=b)
			if result == target_string:
				first_char_idx = a.find(result)
				last_char_idx = first_char_idx + len(result)
			else:
				continue
			# Dans le cas où la sous-chaîne correspond à la chaîne
			if len(item.prediction) == last_char_idx:
				last_char_idx = -1
			cuts = item.cuts
			baseline = item.baseline
			first_cut = cuts[first_char_idx]
			last_cut = cuts[last_char_idx]
			x_1 = min(item[0] for item in first_cut) - 40
			x_2 = max(item[0] for item in last_cut) + 40
			baseline = [baseline[0], baseline[-1]]
			formatted_baseline = baseline[0][0], baseline[0][1], baseline[1][0], baseline[1][1]
			a, b = produce_line_function(formatted_baseline)
			# On calcule y1 et y2
			y_1 = round(a * x_1 + b)
			y_2 = round(a * x_2 + b)
			target_baseline = [[x_1, y_1], [x_2, y_2]]
			baselines.append(target_baseline)
			if show_image:
				assert loaded_image is not None, "Merci d'ajouter l'image si vous voulez la montrer."
				cropped = loaded_image.crop((x_1, y_1 - 70, x_2, y_2 + 70))
				cropped.show()
		baselines = {"coords": baselines,
					 "image_path": image_path}
		return baselines
	else:
		try:
			cuts = line.cuts
		except AttributeError:
			return None
		baseline = line.baseline
		prediction = line.prediction
		prediction = prediction.lower()
		target_string = target_string.lower()
		target_string = target_string.strip()
		prediction = prediction.strip()
		target_string = nfc_normalize(target_string)
		prediction = nfc_normalize(prediction)
		if target_string not in prediction:
			log_print(f"Attention, la ligne '{prediction}' ne contient pas la chaîne recherchée: '{target_string}'.")
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
		y_1 = round(a * x_1 + b)
		y_2 = round(a * x_2 + b)
		target_baseline = {"coords": [[x_1, y_1], [x_2, y_2]],
						   "image_path": image_path}

		if show_image:
			cropped = loaded_image.crop((x_1, y_1 - 70, x_2, y_2 + 70))
			cropped.show()
		return target_baseline


def draw_lines_on_image(image, baselines: list, return_image=False, load_image=True):
	log_print("Attempting to show image")
	if load_image:
		image = Image.open(image)
	draw = PIL.ImageDraw.Draw(image)
	for line in baselines:
		draw.line(line, width=5, fill="green", joint="curve")
	if return_image is False:
		image.show()
	else:
		return image


def get_string_between_two_words(target_string: str, word_a: str, word_b: str) -> str:
	"""
	Cette fonction extrait une sous-chaîne de caractères bornée par 2 mots.
	:param target_string: La chaîne cible
	:param word_a: mot a
	:param word_b: mot b
	:return: la sous-chaîne visée
	"""
	try:
		after_first_delimiter = approximate_sentence_split(sentence=target_string,
														   substring=word_a,
														   max_dist=4,
														   return_match=False)[-1]
	except:
		return None
	try:
		before_second_delimiter = approximate_word_split(sentence=after_first_delimiter,
														 word=word_b,
														 sensibility=0.9,
														 return_word=False)[0]
	except TypeError:
		return None
	return before_second_delimiter

def filter_pages(pages: list, page_class: str):
	"""
	Cette fonction retourne la page d'une classe donnée. Suppose que la minute soit correctement classée
	:param pages: la liste des pages d'une minute
	:param page_class: la classe visée
	:return:
	"""
	try:
		filtered = next(item for item in pages if item["classe"] == page_class)
	except StopIteration:
		log_print(f"Stop iteration. Pages: {pages}")
	return filtered

def log_print(message, print_message=True, filepath="logs/log.txt"):
	if print_message:
		print(message)
	else:
		with open(filepath, "a") as output_log:
			output_log.write("\n" + message)


def extract_lines_from_named_zone(annotations,
								  target_zone: str | list,
								  show_images: bool = False,
								  loaded_image: PIL.Image.Image = None,
								  ocr_prediction: OCRRecord = None,
								  intersect_ratio: float | list[float] = 0.5,
								  select_highest_prob_zone: bool = False,
								  resize_factor=1) -> tuple[OCRRecord, list] | tuple[None, None]:
	"""
	Cette fonction extrait la ou les lignes correspondant à une zone
	:param annotations: L'ensemble des zones prédites
	:param target_zone: Le nom de la ou des zones à récupérer. Dans le cas de plusieurs zones, on récupèrera la ligne
	inclue dans l'intersection de ces zones
	:param show_images: Montrer l'image?
	:param loaded_image: L'image chargée par PIL (debug)
	:param ocr_prediction: OCRRecord: La liste de lignes transcrites (baseline, prediction, cuts)
	:param intersect_ratio: La proportion d'intersection minimale pour considérer la ligne dans la zone ciblée. Un flottant
	ou une liste de flottant si on vise plusieurs zones
	:param select_highest_prob_zone: Faut-il sélectionner la zone détecter la plus probable, en cas de zones multiples
	:return: La liste des lignes concernées par la zone et les zones.
	"""
	# On récupère la boîte correspondante
	if isinstance(intersect_ratio, float):
		intersect_ratio = [intersect_ratio]
	if isinstance(target_zone, str):
		all_zones = [target_zone]
	else:
		all_zones = target_zone
	all_filtered_zones = []
	all_lines = []
	zones_filtrees = []
	for target_zone, ratio in zip(all_zones, intersect_ratio):
		zones = filter_zones(annotations=annotations, category=target_zone)
		if len(zones) == 0:
			return None, None
		elif len(zones) > 1 and len(all_zones) == 1:
			log_print(f"Erreur: plusieurs zones détectées. Target zone: {target_zone}")
			# Si on active l'option du choix de la zone la plus probable, il n'y a qu'à ordonner la liste
			if select_highest_prob_zone:
				zones.sort(key=lambda x: x.probs, reverse=True)
			else:
				return None, None

		# On va commencer par identifier le nom prédit par kraken
		coordonnees_zones_filtrees = zones[0].coordinates
		zones_filtrees.append(coordonnees_zones_filtrees)

		rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
		zones_filtrees_as_rectangle = rectangle(coordonnees_zones_filtrees[0][0],
													 coordonnees_zones_filtrees[0][1],
													 coordonnees_zones_filtrees[1][0],
													 coordonnees_zones_filtrees[1][1])
		all_filtered_zones.append(coordonnees_zones_filtrees)

		if show_images:
			# On doit adapter les dimensions à la taille de l'image chargée qui a été redimensionnée
			cropped = loaded_image.crop((coordonnees_zones_filtrees[0][0] * resize_factor,
										 coordonnees_zones_filtrees[0][1] * resize_factor,
										 coordonnees_zones_filtrees[1][0] * resize_factor,
										 coordonnees_zones_filtrees[1][1] * resize_factor))
			cropped.show()

		# On cherche la ligne qui entre dans la zone zonée
		corresponding_lines = match_lines_in_zones(ocr_prediction=ocr_prediction,
														 zone_as_rectangle=zones_filtrees_as_rectangle,
														 intersect_ratio=ratio)
		corresponding_lines = vertical_order_lines(lines=corresponding_lines)
		all_lines.append([ocr_prediction.index(item) for item in corresponding_lines])

	if len(all_lines) == 1:
		corresponding_line_index = all_lines[0]
	else:
		# https://stackoverflow.com/a/3852806
		corresponding_line_index = set.intersection(*map(set, all_lines))
	corresponding_lines = OCRRecord()
	corresponding_lines.recreate_record([ocr_prediction[idx] for idx in corresponding_line_index])
	return corresponding_lines, zones_filtrees[0] if len(zones_filtrees) == 1 else zones_filtrees

def extract_lines_from_given_zone(target_zone: YOLOZone,
								  ocr_prediction: OCRRecord = None,
								  ratio: float | list[float] = 0.5) -> OCRRecord:
	"""
	Cette fonction extrait la ou les lignes correspondant à une zone
	:param target_zone: Le nom de la ou des zones à récupérer. Dans le cas de plusieurs zones, on récupèrera la ligne
	inclue dans l'intersection de ces zones
	:param ocr_prediction: OCRRecord: La liste de lignes transcrites (baseline, prediction, cuts)
	:param ratio: La proportion d'intersection minimale pour considérer la ligne dans la zone ciblée. Un flottant
	ou une liste de flottant si on vise plusieurs zones
	:return: La liste des lignes concernées par la zone et les zones.
	"""
	# On récupère la boîte correspondante
	# On va commencer par identifier le nom prédit par kraken
	coordonnees_zones_filtrees = target_zone.coordinates

	rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
	zones_filtrees_as_rectangle = rectangle(coordonnees_zones_filtrees[0][0],
												 coordonnees_zones_filtrees[0][1],
												 coordonnees_zones_filtrees[1][0],
												 coordonnees_zones_filtrees[1][1])


	# On cherche la ligne qui entre dans la zone zonée
	corresponding_lines = match_lines_in_zones(ocr_prediction=ocr_prediction,
													 zone_as_rectangle=zones_filtrees_as_rectangle,
													 intersect_ratio=ratio)
	corresponding_lines = vertical_order_lines(lines=corresponding_lines)
	all_lines = [ocr_prediction.index(item) for item in corresponding_lines]

	corresponding_lines = OCRRecord()
	corresponding_lines.recreate_record([ocr_prediction[idx] for idx in all_lines])
	return corresponding_lines


def filter_zones(annotations: YOLORecord,
				 category: str) -> YOLORecord:
	"""
	Fonction permettant de filtrer les zones par catégorie
	:param annotations: un objet YOLORecord
	:param category: la catégorie à filtrer
	:return: un YOLORecord contenant uniquement les zones ciblées
	"""
	return [annotation for annotation in annotations if annotation.label == category]