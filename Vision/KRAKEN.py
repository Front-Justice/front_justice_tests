
from kraken.lib import vgsl
from kraken import blla
from kraken.lib import models
from kraken import rpred
import kraken
import pickle


class KRAKEN():
	def __init__(self, segmentation_model, ocr_model):
		self.segmentation_model = segmentation_model
		self.ocr_model = ocr_model

	def segment_lines_with_kraken(self, image):
		# segmentation_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/lignes_updated.mlmodel"
		seg_model = vgsl.TorchVGSLModel.load_model(self.segmentation_model)
		baseline_seg = blla.segment(image, model=seg_model)
		# with open("../Information_Extractor/segments.json", "wb") as segmentation_as_file:
			# pickle.dump(baseline_seg, segmentation_as_file, protocol=pickle.HIGHEST_PROTOCOL)
		return baseline_seg

	def predict_with_kraken(self, im, segments:kraken.blla.Segmentation):
		"""
		Production de l'inférence à l'aide d'un modèle kraken et de segments.
		:param segments:
		:return:
		"""
		# ocr_model = "/home/mgl/Bureau/Travail/projets/Front_Justice/inference/dataset/models/ocr_updated_150p.mlmodel"
		model = models.load_any(self.ocr_model)

		# with open('../Information_Extractor/segments.json', 'rb') as file:
			# segments = pickle.load(file)
		pred_it = rpred.rpred(model, im, segments)
		for line, record in zip(segments.lines, pred_it):
			print("---")
			print(f"Baseline: {line.baseline}")
			print(f"Prediction: {record.prediction}")