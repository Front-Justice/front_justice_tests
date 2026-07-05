import PIL.Image
from kraken.lib import vgsl
from kraken.serialization import serialize as serialize
from kraken import blla
from kraken.lib import models
from kraken import rpred
import kraken.containers

import src.utils.utils as utils
import dataclasses

class KRAKEN():
	"""
	Classe permettant la segmentation en ligne et l'OCR d'une page.
	La segmentation **en zones** n'est pas gérée par cette classe mais pas une classe YOLO.
	"""
	def __init__(self, segmentation_model, ocr_model, device):
		self.segmentation_model = segmentation_model
		self.ocr_model = ocr_model
		self.device = device

	def segment_lines_with_kraken(self, image):
		seg_model = vgsl.TorchVGSLModel.load_model(self.segmentation_model)
		baseline_seg:kraken.containers.Segmentation = blla.segment(image, model=seg_model, device=self.device)
		return baseline_seg



	def serialize(self, prediction):
		"""
		Applique la sérialisation d'un fichier en ALTO.
		:param prediction:
		:return:
		"""
		serialized = serialize(results=prediction,
							   template="alto",
							   sub_line_segmentation=False)
		return serialized


	def predict_with_kraken(self, im:PIL.Image.Image,
							segments:kraken.blla.Segmentation,
							extract_polygons:bool = False,
							return_kraken_preds = False,
							image_name = None) -> utils.OCRRecord:
		"""
		Production de l'inférence à l'aide d'un modèle kraken et de segments.
		:param im: L'image chargée
		:param segments: Les segments (objet Kraken)
		:return: un objet OCRRecord.
		"""
		model = models.load_any(self.ocr_model, device=self.device)
		pred_it = rpred.rpred(model, im, segments)
		if return_kraken_preds == True:
			results = dataclasses.replace(pred_it.bounds, lines=[item for item in pred_it], imagename=image_name)
			return results
		prediction = []
		for line, record in zip(segments.lines, pred_it):
			interm_dict = {}
			interm_dict['baseline'] = line.baseline
			if extract_polygons:
				interm_dict['polygon'] = line.boundary
			else:
				interm_dict['polygon'] = None
			interm_dict['prediction'] = utils.nfc_normalize(record.prediction)
			interm_dict['cuts'] = record.cuts
			interm_dict['image_path'] = image_name
			prediction.append(interm_dict)
		my_OCR_record = utils.OCRRecord(record=prediction)
		return my_OCR_record
