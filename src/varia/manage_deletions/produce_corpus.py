import os
import re
import uuid
import torch
import text_alignment_tool
import lxml.etree as ET
import sys
import glob
import cv2
from PIL import Image, ImageDraw
import numpy as np
import random
import PIL
import tqdm
from kraken import rpred
import kraken.containers as Containers
from kraken.lib import models
from typing import Union
# import src.varia.Lines_Classifier.lines_classifier as LC
from src.Page_Classifier.utils.utils import show_image
import multiprocessing as mp


def process_line(line):
	global model
	errors = 0
	tagrefs, gold_transcript, baseline, polygon, image_path = line
	global paths
	images = paths['images']
	# corresponding_line = paths[ident]
	# image_path = corresponding_line['path']
	image_basename = image_path.split("/")[-1].replace('.png', '')
	if images[image_path]['extracted'] == 0:
		try:
			# print(f"First line encountered. Loading image {image_path}.")
			gs_im = Image.open(image_path).convert("RGBA")
			im = np.array(Image.open(image_path))
			images[image_path]['images'] = {**images[image_path], **{'loaded_im': {'gs_im': gs_im}}}
			images[image_path]['images']['loaded_im']['im'] = im
		except FileNotFoundError:
			# print(f"File {image_path} not found.")
			return
	else:
		gs_im = images[image_path]['images']['loaded_im']['gs_im']
		im = images[image_path]['images']['loaded_im']['im']
	global image_n
	image_n += 1
	global deletion_images
	global no_deletion_images
	if "⟦" and "⟧" in gold_transcript or tagrefs == "LT8439":
		if re.match(deletion_regexp, gold_transcript):
			# if os.path.isfile(f"{output_dir}/lines/manuscript/{image_path}_{image_n}.png"):
			# 	pass
			# On va filtrer les lignes manuscrites uniquement
			converted_bl = convert_xml_polygon_to_list_of_tuples(baseline)
			converted_pl = convert_xml_polygon_to_list_of_tuples(polygon)
			# line_img = polygon_extraction(polygon=converted_pl, image=gs_im, keep_alpha=False, return_image=True)
			im_as_array = np.asarray(gs_im)
			# try:
			# 	line_img, result = treat_line(index_and_coordinates=(None, polygon, im_as_array))
			# except TypeError:
			# 	images[image_path]['extracted'] += 1
			# 	return
			# if result in ["Mixed", "Manuscript"]:
			# 	pass
			# line_img.save(f"{output_dir}/lines/manuscript/{image_path}_{image_n}.png")
			deletion_images += 1
			# if deletion_images == 10_000:
			# 	print("Deletion images reach 10.000.")
			# 	exit()
			search_regexp = re.compile(r'⟦[^⟦⟧]+⟧')
			between_brackets = re.finditer(search_regexp, gold_transcript)
			transcription_no_bracket = gold_transcript.replace("⟧", "").replace("⟦", "")
			all_spans = []
			for item in between_brackets:
				span = item.span(0)
				correct_span = [span[0] + 1, span[-1] - 1]
				all_spans.append(correct_span)
			alto_line_to_img(loaded_im=im_as_array,
							 points=polygon,
							 out_name=f"{output_dir}/lines/deleted/{image_basename}_{image_n}.png", show_image=False)
			baseline = [Containers.BaselineLine(id='test', baseline=converted_bl, boundary=converted_pl)]
			segmentation = Containers.Segmentation(type="baselines",
												   imagename=image_basename,
												   text_direction="horizontal-lr",
												   lines=baseline,
												   script_detection=False)
			pred_it = rpred.rpred(model, gs_im, segmentation)
			try:
				predicted_line = next(pred_it)
			except Exception as e:
				images[image_path]['extracted'] += 1
				return
			cuts = predicted_line.cuts
			pred = predicted_line.prediction
			try:
				dictionnary = align_strings(text_a=transcription_no_bracket, text_b=pred)
			except text_alignment_tool.text_loaders.text_loader.LoaderError:
				images[image_path]['extracted'] += 1
				return
			last_string_no_del = all_spans[-1][-1]
			first_string_no_del = all_spans[0][0]
			for item in all_spans:
				try:
					first_pos = dictionnary[item[0]]
				except KeyError:
					try:
						first_pos = dictionnary[item[0]]
					except KeyError:
						try:
							first_pos = dictionnary[item[0] + 2]
						except KeyError:
							continue
				try:
					last_pos = dictionnary[item[1]]
				except KeyError:
					try:
						last_pos = dictionnary[item[1]]
					except KeyError:
						try:
							last_pos = dictionnary[item[1] - 2]
						except KeyError:
							continue
				im_n = 0
				names = [f"{output_dir}/chars/deleted/{image_basename}_{im_n}_{str(uuid.uuid4())}.png"
						 for im_n in range(len(cuts[first_pos:last_pos]))]
				batch_alto_line_to_img_cv2(loaded_im=im,
										   points=cuts[first_pos:last_pos],
										   out_names=names,
										   coords_conversion=False,
										   horizontal_expand=30)

			names = [f"{output_dir}/chars/undeleted/{image_basename}_{im_n}_{str(uuid.uuid4())}.png"
					 for im_n in range(len(cuts[last_string_no_del:]))] + [f"{output_dir}/chars/undeleted/{image_basename}_{im_n}_2_{str(uuid.uuid4())}.png"
					 for im_n in range(len(cuts[:first_string_no_del]))]
			points = cuts[last_string_no_del:] + cuts[:first_string_no_del]
			batch_alto_line_to_img_cv2(loaded_im=im, points=points,
									   out_names=names,
									   coords_conversion=False,
									   horizontal_expand=30)

	else:
		# print(no_deletion_images / (deletion_images + 1))
		if no_deletion_images / (deletion_images + 1) < 3:
			no_deletion_images += 1
			alto_line_to_img(loaded_im=gs_im, points=polygon,
								 out_name=f"{output_dir}/lines/undeleted/{image_basename}_{image_n}.png", show_image=False)
	images[image_path]['extracted'] += 1
	if images[image_path]['extracted'] == images[image_path]['len']:
		print(f"Unloading image {image_path}.")
		del paths['images'][image_path]
	return

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
	return resized, as_label


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
		# if show_image:
		# 	cropped_img = Image.fromarray(cropped_img)
		# 	cropped_img.show()
		return None


def polygon_extraction(polygon, image:Union[Image.Image,np.array], keep_alpha:bool=True, return_image:bool=False, vertical_padding=None):
	"""
	Cette fonction extrait un polygone d'une image et la montre
	https://stackoverflow.com/a/22650239
	:return:
	"""
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


def align_strings(text_a, text_b):
	a = text_alignment_tool.text_loaders.StringTextLoader(text_a)
	b = text_alignment_tool.text_loaders.StringTextLoader(text_b)
	aligner = text_alignment_tool.TextAlignmentTool(a, b)
	first_alignment_algorithm = text_alignment_tool.ChunkAlignmentAlgorithm()
	aligner.align_text(first_alignment_algorithm)
	alignments = []
	for single_alignment in aligner.collect_all_alignments()[0][0].query_to_target_mapping.alignments:
		alignments.append((single_alignment.query_idx, single_alignment.target_idx))
	alignment_dict = {source:target for source, target in alignments}
	return alignment_dict


def batch_alto_line_to_img_cv2(loaded_im,
							   points,
							   out_names,
							   coords_conversion=True,
							   horizontal_expand=None,
							   keep_alpha=False,
							   vertical_crop=10):
	# Créer un masque vide (uint8: 0-255)
	mask = np.zeros((loaded_im.shape[0], loaded_im.shape[1]), dtype=np.uint8)

	# Remplir le polygone dans le masque
	all_sizes = []
	for indiv_coord, name in list(zip(points, out_names)):
		if coords_conversion:
			coordinates = convert_xml_polygon_to_list_of_tuples(indiv_coord)
		else:
			coordinates = indiv_coord

		cv2.fillPoly(mask, [np.array(coordinates, dtype=np.int32)], 255)
		# Calculer la bounding box
		x_coords = [p[0] for p in coordinates]
		y_coords = [p[1] for p in coordinates]
		x_min, x_max = max(0, min(x_coords) - horizontal_expand), min(loaded_im.shape[1], max(x_coords) + horizontal_expand)
		y_min, y_max = max(0, min(y_coords) + vertical_crop), min(loaded_im.shape[0], max(y_coords))

		# Recadrer l'image et le masque
		cropped_img = loaded_im[y_min:y_max, x_min:x_max]
		cropped_mask = mask[y_min:y_max, x_min:x_max]

		# Ajouter le canal alpha si nécessaire
		if keep_alpha:
			if cropped_img.shape[2] == 3:  # RGB → RGBA
				cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)
			cropped_img[:, :, 3] = cropped_mask  # Appliquer le masque comme alpha

		as_img = Image.fromarray(cropped_img)
		as_img.save(f"{name}")


def batch_alto_line_to_img(loaded_im, points, out_names, coords_conversion=True, padding=None):
	imArray = np.asarray(loaded_im)
	# 0 creates transparent background. 1 makes the mask
	maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 1)
	mask = np.array(maskIm)
	# assemble new image (uint8: 0-255)

	newImArray = np.empty(imArray.shape, dtype='uint8')
	# colors (three first columns, RGB)
	newImArray[:, :, :3] = imArray[:, :, :3]
	# transparency (4th column)
	newImArray[:, :, 3] = mask * 255
	# back to Image from numpy
	newIm = Image.fromarray(newImArray, "RGBA")
	for indiv_coord, name in list(zip(points, out_names)):
		if coords_conversion:
			coordinates = convert_xml_polygon_to_list_of_tuples(indiv_coord)
		else:
			coordinates = indiv_coord
		ImageDraw.Draw(maskIm).polygon(coordinates, outline=1, fill=1)
		x_max = max([i[0] for i in coordinates])
		x_min = min([i[0] for i in coordinates])
		if padding:
			x_min, x_max = x_min - padding, x_max + padding
		y_max = max([i[1] for i in coordinates])
		y_min = min([i[1] for i in coordinates])
		rectangle_coordinates = (x_min, y_min, x_max, y_max)
		cropped_img = newIm.crop(rectangle_coordinates)
		try:
			cropped_img.save(f"{name}")
		except SystemError:
			continue



def alto_line_to_img(loaded_im, points, out_name, coords_conversion=True, padding=None, show_image=False):
	if coords_conversion:
		coordinates = convert_xml_polygon_to_list_of_tuples(points)
	else:
		coordinates = points
	imArray = np.asarray(loaded_im)
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
	if padding:
		x_min, x_max = x_min - padding, x_max + padding
	y_max = max([i[1] for i in coordinates])
	y_min = min([i[1] for i in coordinates])
	rectangle_coordinates = (x_min, y_min, x_max, y_max)
	cropped_img = newIm.crop(rectangle_coordinates)
	cropped_img.save(f"{out_name}")
	if show_image:
		cropped_img.show()

def convert_xml_polygon_to_list_of_tuples(coords):
	as_list = coords.split(" ")
	x_values = [int(item) for idx, item in enumerate(as_list) if idx % 2 == 0]
	y_values = [int(item) for idx, item in enumerate(as_list) if idx % 2 == 1]
	coordinates = list(zip(x_values, y_values))
	return coordinates

random.seed(1234)
input_files = [glob.glob(sys.argv[n]) for n in range(1, len(sys.argv) - 1)]
file_list = []
[file_list.extend(item) for item  in input_files]
random.shuffle(input_files)
output_dir = sys.argv[-1]
namespaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
ocr_model = "src/Vision/models/htr_29500l.mlmodel"
deletion_regexp = re.compile(r'.*⟦[^⟦⟧]+⟧.*')

# classifier = LC.LinesClassifier(build_vocab=False,
# 								max_size=(1735, 161),
# 								model_path="src/varia/Lines_Classifier/models/LinesClassifierRF_3classes.joblib",
# 								vocab_path="src/varia/Lines_Classifier/models/LinesClassifierRF_3classes.voc")

os.makedirs(exist_ok=True, name=f"{output_dir}/lines/deleted")
os.makedirs(exist_ok=True, name=f"{output_dir}/lines/undeleted")
os.makedirs(exist_ok=True, name=f"{output_dir}/chars/undeleted")
os.makedirs(exist_ok=True, name=f"{output_dir}/chars/deleted")
print(output_dir)
deletion_images = len(glob.glob(f"{output_dir}/lines/deleted/*.png"))
no_deletion_images = len(glob.glob(f"{output_dir}/lines/undeleted/*.png"))
all_zips = []
global paths
paths = {}
for file in tqdm.tqdm(file_list):
	basename = "/".join(file.split("/")[:-1])
	as_tree = ET.parse(file)
	image_path = as_tree.xpath("//alto:fileName", namespaces=namespaces)[0].text
	absolute_path = f"{basename}/{image_path}"
	# lines = as_tree.xpath("//alto:TextLine", namespaces=namespaces)
	# LT8439
	tagrefs = as_tree.xpath("//alto:TextLine/@TAGREFS", namespaces=namespaces)
	baselines = as_tree.xpath("//alto:TextLine/@BASELINE", namespaces=namespaces)
	transcriptions = as_tree.xpath("//alto:TextLine/alto:String/@CONTENT", namespaces=namespaces)
	polygons = as_tree.xpath("//alto:TextLine/alto:Shape/alto:Polygon/@POINTS", namespaces=namespaces)
	all_paths = [absolute_path for i in range(len(polygons))]
	all_ids = [uuid.uuid4() for i in range(len(polygons))]
	# line_ids = as_tree.xpath("//alto:TextLine/@ID", namespaces=namespaces)
	# paths = {**paths, **{ident: {'path': absolute_path, 'loaded_im': None} for ident, path in zip(all_ids, all_paths)}}
	try:
		paths['images'][absolute_path] = {'len': len(polygons), 'extracted': 0}
	except KeyError:
		paths['images'] = {absolute_path: {'len': len(polygons), 'extracted': 0}}
	paths['images'][absolute_path]['images'] = None
	zipped_content = list(zip(tagrefs, transcriptions, baselines, polygons, all_paths))
	all_zips.extend(zipped_content)
global image_n
image_n = 0
global model
model = models.load_any(ocr_model, device="cpu")
deletion_images = 0
no_deletion_images = 0

torch.set_num_threads(1)
with mp.Pool(processes=32) as pool:
	data = [(line,) for line in all_zips]
	pool.starmap(process_line, tqdm.tqdm(data))
	print("Images Done.")

