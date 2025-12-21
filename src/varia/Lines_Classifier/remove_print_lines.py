import os

import lines_classifier as TC
import lxml.etree as ET
from PIL import Image, ImageDraw
import random
import glob
import sys
from multiprocessing import Pool
import tqdm
import numpy as np

def classify(image):
	pil_image = Image.open(image)
	print(pil_image.size)
	exit(0)
	# 0 creates transparent background. 1 makes the mask
	maskIm = Image.new('L', (pil_image.shape[1], pil_image.shape[0]), 1)
	ImageDraw.Draw(maskIm).polygon(coordinates, outline=1, fill=1)
	mask = np.array(maskIm)
	# assemble new image (uint8: 0-255)

	newImArray = np.empty(pil_image.shape, dtype='uint8')
	# colors (three first columns, RGB)
	newImArray[:, :, :3] = pil_image[:, :, :3]
	# transparency (4th column)
	newImArray[:, :, 3] = mask * 255
	# back to Image from numpy
	newIm = Image.fromarray(newImArray, "RGBA")

	x_max = max([i[0] for i in coordinates])
	x_min = min([i[0] for i in coordinates])
	y_max = max([i[1] for i in coordinates])
	y_min = min([i[1] for i in coordinates])
	rectangle_coordinates = (x_min, y_min, x_max, y_max)
	cropped_img = newIm.crop(rectangle_coordinates)
	width, height = cropped_img.size

	# Si jamais la hauteur/largeur de la ligne est + haute que la hauteur/largeur max entraînée,
	# on passe
	if height/2 > classifier.max_size[1] or width/2 > classifier.max_size[0]:
		print("Line out of bonds.")
		print(height, width, classifier.max_size)
		return
	resized = cropped_img.resize((int(width / 2), int(height / 2)))
	prediction = classifier.predict_line(line=resized)
	if prediction is None:
		return
	as_label = classifier.vocab[prediction[0]]
	return as_label

def treat_line(index_and_coordinates):
	index, coordinates, imArray = index_and_coordinates
	coordinates = convert_xml_polygon_to_list_of_tuples(coordinates)
	# 0 creates transparent background. 1 makes the mask
	maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 1)
	ImageDraw.Draw(maskIm).polygon(coordinates, outline=1, fill=1)
	mask = np.array(maskIm)
	# assemble new image (uint8: 0-255)

	newImArray = np.empty(imArray.shape, dtype='uint8')
	# colors (three first columns, RGB)
	newImArray[:, :, :3] = imArray[:, :, :3]
	# transparency (4th column)
	newImArray[:, :, 3] = mask * 255
	# back to Image from numpy
	newIm = Image.fromarray(newImArray, "RGBA")

	x_max = max([i[0] for i in coordinates])
	x_min = min([i[0] for i in coordinates])
	y_max = max([i[1] for i in coordinates])
	y_min = min([i[1] for i in coordinates])
	rectangle_coordinates = (x_min, y_min, x_max, y_max)
	cropped_img = newIm.crop(rectangle_coordinates)
	width, height = cropped_img.size

	# Si jamais la hauteur/largeur de la ligne est + haute que la hauteur/largeur max entraînée,
	# on passe
	if height/2 > classifier.max_size[1] or width/2 > classifier.max_size[0]:
		print("Line out of bonds.")
		print(height, width, classifier.max_size)
		return
	resized = cropped_img.resize((int(width / 2), int(height / 2)))
	prediction = classifier.predict_line(line=resized)
	if prediction is None:
		return
	as_label = classifier.vocab[prediction[0]]
	return index, as_label


def convert_xml_polygon_to_list_of_tuples(string):
	as_list = string.split(" ")
	x_values = [int(item) for idx, item in enumerate(as_list) if idx % 2 == 0]
	y_values = [int(item) for idx, item in enumerate(as_list) if idx % 2 == 1]
	coordinates = list(zip(x_values, y_values))
	return coordinates


def main(input_files):
	random.shuffle(input_files)
	for file in tqdm.tqdm(input_files):
		corresponding_file_mixed = file.replace(".xml", ".mixed_line_only.xml")
		print(file)
		if ".mixed_line_only.xml" in file or os.path.isfile(corresponding_file_mixed):
			print("Already treated")
			print(file)
			continue
		corresponding_image = file.replace(".xml", ".jpg")
		im = Image.open(corresponding_image).convert("RGBA")
		print(corresponding_file_mixed)
		try:
			file_as_xml = ET.parse(file)
		except ET.XMLSyntaxError:
			continue
		all_lines = file_as_xml.xpath("//alto:TextLine/alto:Shape/alto:Polygon/@POINTS", namespaces=namespaces)
		assert len(all_lines) != [], "Problem with namespace"
		# https://stackoverflow.com/a/22650239
		im = np.asarray(im)
		index_and_coords = [(idx, coordinate, im) for idx, coordinate in enumerate(all_lines)]
		with Pool(16) as p:
			predictions = p.map(treat_line, index_and_coords)
		all_lines = file_as_xml.xpath("//alto:TextLine", namespaces=namespaces)
		for result in predictions:
			if result is None:
				continue
			index, label = result
			if label == "Print":
				target_line = all_lines[index]
				target_line.getparent().remove(target_line)

		with open(corresponding_file_mixed, "w") as f:
			f.write(ET.tostring(file_as_xml, pretty_print=True, encoding='utf8').decode("utf-8"))


if __name__ == '__main__':
	classifier = TC.LinesClassifier(build_vocab=False,
									max_size=(1735, 161),
									model_path="models/LinesClassifierRF.joblib",
									vocab_path="models/LinesClassifierVocabRF.joblib")

	input_files = glob.glob(f"{sys.argv[1]}*.xml")
	namespaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
	main(input_files)