##########

## Script qui permet de faire la segmentation et l'ocr kraken ainsi que la segmentation YOLO
# TODO: réfléchir à la sortie produite
# TODO: fusionner avec le script PARTY

##########

from ultralytics import YOLO
import PIL.Image as Image
from src.utils.utils import YOLORecord
import src.utils.utils as utils
# import src.Vision.PARTY as PARTY


def load(model_path):
	"""
	Cette fonction charge un modèle YOLO à partir d'un chemin donné.
	Elle prend en argument le chemin vers le fichier de modèle et
	renvoie une instance de la classe YOLO, qui représente le modèle
	chargé.
	:param model_path: le chemin vers le modèle
	:return: le modèle chargé
	"""
	return YOLO(model_path)


class YOLOSegmenter():
	def __init__(self, models):
		self.model_page_1 = models["page_1"]
		self.model_table_magistrats = models["magistrats"]


	def segment_zones(self,
					  image:str,
					  target_classes:list,
					  confidence=0.5,
					  model=None,
					  show_image=False) -> tuple[YOLORecord, list[str]]:
		"""
		La segmentation d'une image à l'aide de plusieurs modèles YOLO, adaptés au type de page.
		:param image: Le chemin vers l'image
		:param target_classes: Les classes qu'il faut trouver dans l'image
		:param confidence: Le seuil de confiance minimal
		:param model: Le modèle de segmentation
		:param show_image: Option pour faire apparaître l'image
		:return: Un tuple (YOLORecord, classes manquantes)
		"""


		# Run batched inference on a list of images
		results = model([image], conf=confidence, verbose=False)[0]  # return a list of Results objects
		if show_image:
			results.show()
		check_list = []
		# Les résultats de l'analyse sur la page 1
		results_list = []
		classes_dict = model.names
		boxes = results.boxes  # Boxes object for bounding box outputs
		classes = [round(item) for item in boxes.cls.tolist()]
		probs = [round(item, 3) for item in boxes.conf.tolist()]
		as_labels = [classes_dict[obj] for obj in classes]
		coordinates = boxes.xyxy.tolist()
		for label, coordinate, prob in list(zip(as_labels, coordinates, probs)):
			results_list.append({"label": label,
								 "coordinates": utils.format_coordinates(coordinate),
								 "probs": prob})
			check_list.append(label)

		missing = utils.check_if_missing(target_classes, check_list)
		if len(missing) > 0:
			print(f"Certains éléments de la page n'ont pas été identifiés: {missing}")
		else:
			missing = []
		return YOLORecord(results_list), missing


if __name__ == '__main__':
	image = "data/test_data/11_J_31(1)-0011.jpg"
	im = Image.open(image)
	# segments = segment_lines_with_kraken(im)
	segments = None
	yol = YOLOSegmenter()
	yol.segment(segments)