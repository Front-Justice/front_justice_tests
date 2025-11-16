##########

## Script qui permet de faire la segmentation et l'ocr kraken ainsi que la segmentation YOLO
# TODO: réfléchir à la sortie produite
# TODO: fusionner avec le script PARTY

##########

from ultralytics import YOLO
import PIL.Image as Image
import utils.utils as utils
import Vision.PARTY as PARTY


def load(model_path):
	return YOLO(model_path)


class YOLOSegmenter():
	def __init__(self, model):
		self.model = model

	def segment(self, image:str) -> list[dict[str, list[int]]]:
		"""
		La segmentation d'une image à l'aide de plusieurs modèles YOLO, adaptés au type de page.
		:param image: le chemin vers l'image
		:return:
		"""

		# Run batched inference on a list of images
		results = self.model([image])  # return a list of Results objects

		# Les résultats de l'analyse sur la page 1
		results_dict = []
		for result in results:
			print("---")
			classes_dict = self.model.names
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
					predictor = PARTY.PartyPredict()
					print(correct_coord)
					segmentation = predictor.create_baseline(correct_coord, corresponding_image=image)
					prediction = predictor.inference(segmentation=segmentation, image=as_image)
					print(prediction)
					exit(0)
			print(f"Classes: {classes}")
			print(f"As labels: {as_labels}")
			print(f"xyxy: {coordinates}")

		page_class = 1
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
	YOLO = YOLOSegmenter()
	YOLO.segment(segments)