import sys

import Page_Classifier.page_classifier as PC
import glob
import PIL.Image as Image

class Pipeline():
	def __init__(self,
				 page_classifier_model,
				 page_classifier_vocab):
		self.page_classifier = PC.PageClassifier(build_vocab=False,
												 model=page_classifier_model,
												 vocab=page_classifier_vocab)
		self.current_image = None
		self.current_image_path = None

	def load_image(self, image):
		self.current_image = Image.open(image)
		self.current_image_path = image

	def classify_image(self):
		self.current_page = self.page_classifier.predict(image=self.current_image_path)

	def process(self, images):
		for image in images:
			print("---")
			print(image)
			self.load_image(image)
			self.classify_image()
			print(self.current_page)


def main(images_dir):
	images = glob.glob(f"{images_dir}/*.jpg")
	images.sort(key=lambda x:int(x.split("/")[-1].split(".jpg")[0].split("_")[-1]))
	pipeline = Pipeline(page_classifier_model="Page_Classifier/models/PageClassifier.joblib",
						page_classifier_vocab="Page_Classifier/models/vocab.joblib")
	pipeline.process(images)

if __name__ == '__main__':
	images_dir = sys.argv[1]
	main(images_dir)