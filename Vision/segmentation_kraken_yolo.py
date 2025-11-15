##########

## Script qui permet de faire la segmentation et l'ocr kraken ainsi que la segmentation YOLO
# TODO: réfléchir à la sortie produite
# TODO: fusionner avec le script PARTY

##########

import pickle
from ultralytics import YOLO
from kraken.lib import vgsl
from kraken import blla
import PIL.Image as Image
from kraken.lib import models
from kraken import rpred
import kraken
import utils as utils
import predict_with_party



def segment_lines_with_kraken(image):
	segmentation_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/lignes_updated.mlmodel"
	seg_model = vgsl.TorchVGSLModel.load_model(segmentation_model)
	baseline_seg = blla.segment(image, model=seg_model)
	with open("../Information_Extractor/segments.json", "wb") as segmentation_as_file:
		pickle.dump(baseline_seg, segmentation_as_file, protocol=pickle.HIGHEST_PROTOCOL)
	exit(0)
	return baseline_seg

def predict_with_kraken(segments:kraken.blla.Segmentation):
	"""
	Production de l'inférence à l'aide d'un modèle kraken et de segments.
	:param segments:
	:return:
	"""
	ocr_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/ocr_updated_150p.mlmodel"
	model = models.load_any(ocr_model)

	with open('../Information_Extractor/segments.json', 'rb') as file:
		segments = pickle.load(file)
	pred_it = rpred.rpred(model, im, segments)
	for line, record in zip(segments.lines, pred_it):
		print("---")
		print(f"Baseline: {line.baseline}")
		# print(f"Cuts: {record.cuts}")
		print(f"Prediction: {record.prediction}")
		# print(f"Confidences: {record.confidences}")
		# print(f"Type: {record.type}")


def segment_with_YOLO(image:str, page_class:int=None) -> list[dict[str, list[int]]]:
	"""
	La segmentation d'une image à l'aide de plusieurs modèles YOLO, adaptés au type de page.
	:param image: le chemin vers l'image
	:param page_class: La classe de l'image (0 -> 4), 0 étant la classe autre
	:return:
	"""
	page_class = 1
	model_dict = {1: "../segmentation_models/yolov11_page1.pt"}
	yolo_model = model_dict[page_class]
	# Load a model
	model = YOLO(yolo_model)  # pretrained YOLO11n model

	# Run batched inference on a list of images
	results = model([image])  # return a list of Results objects

	# Les résultats de l'analyse sur la page 1
	results_dict = []
	for result in results:
		print("---")
		classes_dict = model.names
		print(classes_dict)
		boxes = result.boxes  # Boxes object for bounding box outputs
		classes = [round(item) for item in boxes.cls.tolist()]
		as_labels = [classes_dict[obj] for obj in classes]
		coordinates = boxes.xyxy.tolist()
		for label, coordinate in zip(as_labels, coordinates):
			results_dict.append({"label": label,
								 "coordinates": utils.simplify_coordinate(coordinate)})
			if label == "Nom du soldat":
				correct_coord = utils.simplify_coordinate(coordinate)
				correct_coord = [[correct_coord[0], correct_coord[1]],
								 [correct_coord[2], correct_coord[3]]]
				as_image = Image.open(image)
				as_image.show()
				predictor = predict_with_party.PartyPredict()
				print(correct_coord)
				segmentation = predictor.create_baseline(correct_coord, corresponding_image=image)
				prediction = predictor.inference(segmentation=segmentation, image=as_image)
				print(prediction)
				exit(0)
		print(f"Classes: {classes}")
		print(f"As labels: {as_labels}")
		print(f"xyxy: {coordinates}")

	# Si on est page 1, on applique le modèle sur les tables
	if page_class == 1:
		# On applique le deuxième modèle, d'identification des tables de magistrat
		yolo_model_table = "../segmentation_models/yolov11_table_magistrats.pt"
		# Load a model
		model = YOLO(yolo_model_table)  # pretrained YOLO11n model

		results_magistrates = model([image])  # return a list of Results objects
		for result in results_magistrates:
			print("---")
			classes_dict = model.names
			print(classes_dict)
			boxes = result.boxes  # Boxes object for bounding box outputs
			classes = [round(item) for item in boxes.cls.tolist()]
			as_labels = [classes_dict[obj] for obj in classes]
			coordinates = boxes.xyxy.tolist()
			print(f"Classes: {classes}")
			print(f"As labels: {as_labels}")
			print(f"xyxy: {coordinates}")

if __name__ == '__main__':
	image = "data/test_data/11_J_31(1)-0011.jpg"
	im = Image.open(image)
	# segments = segment_lines_with_kraken(im)
	segments = None
	predict_with_kraken(segments)
	segment_with_YOLO(image=image)