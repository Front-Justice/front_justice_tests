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

from src.Page_Classifier.utils.utils import show_image


class PageClassifier():
	def __init__(self,
				 build_vocab=True,
				 corpus_path=None,
				 model=None,
				 vocab=None):
		if build_vocab:
			self.vocab, self.reverse_vocab = self.build_classes_vocab(path="src/Page_Classifier/data/corpus/page_*")
			joblib.dump(self.vocab, vocab)
		self.model_path = model
		self.corpus_path = corpus_path
		self.vocab_path = vocab
		try:
			self.model = joblib.load(model)
			if build_vocab is False:
				self.vocab = joblib.load(vocab)
		except FileNotFoundError:
			print("Some file not found.")
			self.model = model
			self.vocab = vocab

	def crop_and_resize(self, image, vertical_crop_factor):
		height_resized = image.height // vertical_crop_factor
		image = image.crop((0, 0, image.width, height_resized))
		image = image.resize((1062, 391))
		# image = image.resize(dims)
		#print(image.size)
		#Image.Image.show(image)
		return image

	def load_to_greyscale(self, image_path):
		image = Image.open(image_path).convert('L')
		return image

	def extract_features(self, image, show_features=False):
		if show_features:
			fd, hog_image = skimage.feature.hog(
				image,
				orientations=9,
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
		plt.savefig('assets/foo.png', dpi=300)
		exit(0)

	def build_classes_vocab(self, path):
		vocab = {idx: dir.split("/")[-1] for idx, dir in enumerate(glob.glob(path))}
		reverse_vocab = {value: key for key, value in vocab.items()}
		print(vocab)
		print(reverse_vocab)
		return vocab, reverse_vocab

	def load_image(self, image_path, produce_labels=False, show_image=False):
		image = self.load_to_greyscale(image_path)
		if show_image:
			Image.Image.show(image)
		cropped = self.crop_and_resize(image, vertical_crop_factor=2)
		features = self.extract_features(cropped)
		if produce_labels:
			return features, self.reverse_vocab.get(image_path.split("/")[-2])
		else:
			return features

	def retrieve_hog_and_label(self, image):
		return self.load_image(image, produce_labels=True)

	def build_dataset(self):
		print("Treating images.")
		images = glob.glob('src/Page_Classifier/data/corpus/page_*/*.jpg')
		random.shuffle(images)
		with Pool(12) as p:
			corpus = p.map(self.retrieve_hog_and_label, images)
		X = [array for array, _ in corpus]
		y = [label for _, label in corpus]
		with open(self.corpus_path, "wb") as corpus_file:
			pickle.dump((X, y), corpus_file)
		return X, y

	def load_corpus(self):
		with open(self.corpus_path, "rb") as corpus_file:
			inputs, labels = pickle.load(corpus_file)
		return inputs, labels

	def test(self):
		inputs, labels = self.load_corpus()
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
		y_pred = self.model.predict(X_test)
		print(classification_report(y_pred, y_test))

	def train(self,
			  show_features=False):
		if not os.path.isfile(self.corpus_path):
			print("Creating corpus file...")
			inputs, labels = self.build_dataset()
		else:
			inputs, labels = self.load_corpus()
		assert inputs != [], "Inputs empty"
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
		self.model = RandomForestClassifier(n_estimators=100, random_state=0)
		# self.model = svm.SVC()
		# fit the model to the training set
		self.model.fit(X_train, y_train)
		accuracy = self.model.score(X_test, y_test)
		print(accuracy)
		y_pred = self.model.predict(X_test)
		print(classification_report(y_pred, y_test))
		# https://stackoverflow.com/a/20662980
		joblib.dump(self.model, self.model_path)
		joblib.dump(self.vocab, self.vocab_path)

	def test(self):
		inputs, labels = self.load_corpus()
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
		self.model.fit(X_train, y_train)
		accuracy = self.model.score(X_test, y_test)
		print(accuracy)
		y_pred = self.model.predict(X_test)
		print(classification_report(y_pred, y_test))

	def predict(self,
				debug_model=False,
				image=False):
		if debug_model:
			importances = self.model.feature_importances_
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

		test_image = self.load_image(image_path=image, show_image=False)
		prediction = self.model.predict([test_image])
		return self.vocab[prediction[0]]


	# Predict on the test set results


if __name__ == '__main__':
	# SVC_classifier = PageClassifier(build_vocab=False,
	# 							corpus_path="src/Page_Classifier/data/corpus.data",
	# 							model="src/Page_Classifier/models/PageClassifier_SVC.joblib",
	# 							vocab="src/Page_Classifier/models/vocab_SVC.joblib")
	RF_classifier = PageClassifier(build_vocab=True,
								corpus_path="src/Page_Classifier/data/corpus.data",
								model="src/Page_Classifier/models/PageClassifier_RF.joblib",
								vocab="src/Page_Classifier/models/vocab_RF.joblib")
	# print(SVC_classifier.vocab)
	if len(sys.argv) > 1:
		images = glob.glob(f"{sys.argv[1]}*.jpg")
		assert images != [], "No images found."
		out_dir = sys.argv[2]
	else:
		out_dir = "src/Page_Classifier/data/predictions/"
	print("Random Forest:")
	RF_classifier.train()
	RF_classifier.test()
	exit(0)
	print("---\nSVC test:")
	# SVC_classifier.test()
	# exit(0)
	# for image in tqdm.tqdm(images):
	# 	if len(glob.glob(f"{out_dir}/*/{image.split('/')[-1]}")) > 0:
	# 		print("Already treated.")
	# 		continue
	# 	prediction = RF_classifier.predict(image=image)
	# 	try:
	# 		os.mkdir(f"{out_dir}")
	# 	except FileExistsError:
	# 		pass
	# 	try:
	# 		os.mkdir(f"{out_dir}/{prediction}")
	# 	except FileExistsError:
	# 		pass
	# 	shutil.copyfile(image, f"{out_dir}/{prediction}/{image.split('/')[-1]}")
