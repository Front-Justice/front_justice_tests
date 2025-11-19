import glob
import pickle
import shutil
import sys
from multiprocessing import Pool

import tqdm
import PIL.Image as Image
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import skimage
from skimage import color, data, exposure
from skimage import io
import random
import cv2
import joblib
import os
import matplotlib.pyplot as plt


class LinesClassifier():
	def __init__(self,
				 build_vocab=True,
				 corpus_path=None,
				 binary_corpus_path=None,
				 max_size=None,
				 model_path=None,
				 vocab_path=None):
		self.corpus_path=corpus_path
		if build_vocab:
			self.vocab, self.reverse_vocab = self.build_classes_vocab(path=corpus_path)
		self.model_path = None
		self.binary_corpus_path = binary_corpus_path
		self.max_size = max_size

		self.model = joblib.load(model_path)
		self.vocab = joblib.load(vocab_path)


	def pad_image(self, image):
		width, height = image.size
		padding_x = round((self.max_size[0] - width) / 2) + 1
		padding_y = round((self.max_size[1] - height) / 2) + 1
		new_width = self.max_size[0]
		new_height = self.max_size[1]
		result = Image.new(image.mode, (new_width, new_height), (255,))
		result.paste(image, (padding_x, padding_y))
		return result

	def load_to_greyscale(self, image_path):
		if isinstance(image_path, str):
			image = Image.open(image_path).convert('L')
		else:
			image = image_path.convert('L')
		return image

	def extract_features(self, image, show_features=False):
		if show_features:
			fd, hog_image = skimage.feature.hog(
				image,
				orientations=8,
				pixels_per_cell=(16, 16),
				cells_per_block=(2, 2),
				visualize=True,
			)
		else:
			hog_features = skimage.feature.hog(
				image,
				orientations=9,
				pixels_per_cell=(16, 16),
				cells_per_block=(2, 2),
				visualize=show_features
			)
		if show_features:
			self.reveal_hog_features(image=image, hog_image=hog_image)
		return hog_features

	def reveal_hog_features(self, image, hog_image):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('Input image')

		# Rescale histogram for better display
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

		ax2.axis('off')
		ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		ax2.set_title('HOG visualization')
		# plt.show()
		plt.savefig('assets/foo.png', dpi=300)
		exit(0)

	def build_classes_vocab(self, path):
		vocab = {idx: dir.split("/")[-1] for idx, dir in enumerate(glob.glob(path))}
		reverse_vocab = {value: key for key, value in vocab.items()}
		return vocab, reverse_vocab

	def hog_label_from_image(self, image_path, produce_labels=False, show_image=False):
		image = self.load_to_greyscale(image_path)
		width, height = image.size
		if width < 60 or height < 30:
			return
		padded_image = self.pad_image(image)
		if show_image:
			Image.Image.show(padded_image)
		features = self.extract_features(padded_image)
		if produce_labels:
			return features, self.reverse_vocab.get(image_path.split("/")[-2])
		else:
			return features

	def retrieve_hog_and_label(self, image):
		return self.hog_label_from_image(image, produce_labels=True)

	def build_dataset(self):
		print("Treating images.")
		images = glob.glob(f'{self.corpus_path}/*.png')
		random.shuffle(images)
		with Pool(12) as p:
			corpus = p.map(self.retrieve_hog_and_label, images)
		X = [array for array, _ in corpus]
		y = [label for _, label in corpus]
		with open(self.binary_corpus_path, "wb") as corpus_file:
			pickle.dump((X, y), corpus_file)
		return X, y

	def load_corpus(self):
		with open(self.binary_corpus_path, "rb") as corpus_file:
			inputs, labels = pickle.load(corpus_file)
		return inputs, labels

	def train(self,
			  model_path=None,
			  vocab_path=None):
		print(self.vocab)
		if not os.path.isfile(self.binary_corpus_path):
			inputs, labels = self.build_dataset()
		else:
			inputs, labels = self.load_corpus()
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)
		model = svm.SVC()
		model = RandomForestClassifier(n_estimators=100, random_state=0)
		# fit the model to the training set
		print("Training model")
		model.fit(X_train, y_train)
		accuracy = model.score(X_test, y_test)
		print(accuracy)
		y_pred = model.predict_to_dir(X_test)
		print(classification_report(y_pred, y_test))
		# https://stackoverflow.com/a/20662980
		joblib.dump(model, self.model_path)
		joblib.dump(self.vocab, self.vocab_path)

	def predict_line(self,
					 line):

		image_as_hog = self.hog_label_from_image(image_path=line, show_image=False, produce_labels=False)
		if image_as_hog is not None:
			prediction = self.model.predict([image_as_hog])
			return prediction
		else:
			return None
	
	def predict_to_dir(self,
				debug_model=False,
				images=False):

		images = glob.glob(f"{sys.argv[1]}*.png")
		assert images != [], "No images found."
		model = joblib.load(self.model_path)
		vocab = joblib.load(self.vocab_path)
		random.shuffle(images)
		if debug_model:
			importances = model.feature_importances_

			# Trier les caractéristiques par importance
			indices = importances.argsort()[::-1]

			# Afficher les 20 caractéristiques les plus importantes
			plt.figure(figsize=(12, 6))
			plt.title("Top 20 caractéristiques les plus importantes")
			plt.bar(range(20), importances[indices][:20], align="center")
			plt.xticks(range(20), indices[:20])
			plt.xlabel("Index de la caractéristique (bin HOG)")
			plt.ylabel("Importance")
			plt.show()
			exit(0)

		for image in images:
			test_image = self.hog_label_from_image(image_path=image, show_image=False)
			if test_image is None:
				continue
			prediction = model.predict([test_image])
			out_dir = vocab[prediction[0]]
			try:
				os.mkdir(f"predictions/")
			except FileExistsError:
				pass
			try:
				os.mkdir(f"predictions/{out_dir}")
			except FileExistsError:
				pass
			shutil.copyfile(image, f"predictions/{out_dir}/{image.split('/')[-1]}")

	# Predict on the test set results


if __name__ == '__main__':
	classifier = LinesClassifier(build_vocab=True,
								 corpus_path="/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/page_classification/lines_classification/data/*",
								 binary_corpus_path="/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/page_classification/lines_classification/corpus.data",
								 max_size=(1735,161),
								 model_path="models/LinesClassifier.joblib",
								vocab_path="models/LinesClassifierVocab.joblib")
	# classifier.train()
	classifier.predict_to_dir()
