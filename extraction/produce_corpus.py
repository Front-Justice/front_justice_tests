import json
import random
import re
import utils
from multiprocessing import Pool
import glob
import PIL.Image as Image
import numpy as np
import os
import utils.utils as utils
# https://stackoverflow.com/a/27162334
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def convert_from_ls(annotation):
	w, h = annotation['original_width'], annotation['original_height']
	return w * annotation['x'] / 100.0, \
		   h * annotation['y'] / 100.0, \
		   w * annotation['width'] / 100.0, \
		   h * annotation['height'] / 100.0

def treat_annotation(cell):
	annotation, idx, images_dir, sliding_value_x, sliding_value_y, mean_width, mean_height = cell
	image_name = annotation["image"].split("/")[-1]
	corresponding_image = glob.glob(f"{images_dir}/{image_name}")
	if corresponding_image == []:
		pass
	else:
		corresponding_image = corresponding_image[0]

	original_width, original_height = annotation["label"][0]["original_width"], annotation["label"][0]["original_height"]
	# Extract True value
	all_x = [n * sliding_value_x * original_width
			 for n in range(0, int(1 / sliding_value_x))]
	all_y = [n * sliding_value_y * original_height
			 for n in range(0, int(1 / sliding_value_y))]


	# x_min y_min x_max y_max
	for index, label in enumerate(annotation["label"]):
		x, y, width, height = convert_from_ls(label)
		loaded = utils.load_image(corresponding_image)
		# Left, Upper, Right, Lower
		good_coordinates = (x, y, x + width, y + height)
		GT_rectangle = Rectangle(good_coordinates[0],
								 good_coordinates[1],
								 good_coordinates[2],
								 good_coordinates[3])
		cropped = crop_image(loaded, good_coordinates)
		cropped.save(f"../data/name_extraction/corpus/true/label_{idx}_{index}.png")
		for idx_x, x in enumerate(all_x):
			for idx_y, y in enumerate(all_y):
				good_coordinates = (x, y, x + mean_width, y + mean_height)
				current_rectangle = Rectangle(good_coordinates[0],
											  good_coordinates[1],
											  good_coordinates[2],
											  good_coordinates[3])
				overlap = check_if_overlap(GT_rectangle, current_rectangle)

				loaded = utils.load_image(corresponding_image)
				if overlap != None and overlap > 0.4:
					cropped = utils.crop_image(loaded, good_coordinates, show_image=False)
					if not os.path.isfile(f"../data/name_extraction/corpus/true/{idx}_{index}_{idx_x}_{idx_y}.png"):
						cropped.save(f"../data/name_extraction/corpus/true/{idx}_{index}_{idx_x}_{idx_y}.png")
				else:
					random_float = random.random()
					if random_float < 0.08:
						cropped = utils.crop_image(loaded, good_coordinates, show_image=False)
						if not os.path.isfile(f"../data/name_extraction/corpus/false/{idx}_{index}_{idx_x}_{idx_y}.png"):
							cropped.save(f"../data/name_extraction/corpus/false/{idx}_{index}_{idx_x}_{idx_y}.png")
	exit(0)


def produce_corpus(json_file):
	images_dir = "/home/mgl/Téléchargements/export_images_front_justice/images"
	with open(json_file) as js_file:
		annotations = json.load(js_file)

	all_heights = []
	all_widths = []

	for annotation in annotations:
		for label in annotation["label"]:
			x, y, width, height = convert_from_ls(label)
			all_heights.append(height)
			all_widths.append(width)

	mean_height = round(np.mean(all_heights)/2)
	mean_width = round(np.mean(all_widths)/2)
	sliding_value_x = 0.08
	sliding_value_y = 0.02


	data = [(annotation,
			 idx,
			 images_dir,
			 sliding_value_x,
			 sliding_value_y,
			 mean_width,
			 mean_height
			 )
			for idx, annotation in enumerate(annotations)]
	with Pool(16) as p:
		p.map(treat_annotation, data)


def check_if_overlap(target, source):  # returns None if rectangles don't intersect
	dx = min(target.xmax, source.xmax) - max(target.xmin, source.xmin)
	dy = min(target.ymax, source.ymax) - max(target.ymin, source.ymin)
	area_source = round((source.xmax - source.xmin) * (source.ymax - source.ymin))
	if (dx>=0) and (dy>=0):
		overlap_area = round(dx*dy)
		ratio = round(overlap_area / area_source, 2)
		return ratio


def load_image(image_path):
	image = Image.open(image_path).convert('L')
	return image


def crop_image(image, coordinates, show_image=False):
	image = image.crop(coordinates)
	# dims = (image.width // resize_factor, image.height // resize_factor)
	image = image.resize((1062, 391))
	# image = image.resize(dims)
	#print(image.size)
	if show_image:
		Image.Image.show(image)
	return image



def main(annotations):
	boxes_heights, boxes_width = produce_corpus(annotations)

if __name__ == '__main__':
	main("../data/name_extraction/gold.json")