import glob
import shutil

import tqdm
import PIL.Image as Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skimage
from skimage import color, data, exposure
from skimage import io
import random
import cv2
import joblib
import os
import matplotlib.pyplot as plt


class RandomForestImageClassifier():
	def __init__(self, build_vocab=True):
		if build_vocab:
			self.vocab, self.reverse_vocab = self.build_classes_vocab(path="data/page_*")
		self.model_path = None

	def crop_and_resize(self, image, vertical_crop_factor):
		height_resized = image.height // vertical_crop_factor
		image = image.crop((0, 0, image.width, height_resized))
		# dims = (image.width // resize_factor, image.height // resize_factor)
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

	def load_image(self, image_path, produce_labels=False, show_image=False):
		image = self.load_to_greyscale(image_path)
		if show_image:
			Image.Image.show(image)
		cropped = self.crop_and_resize(image, vertical_crop_factor=4)
		features = self.extract_features(cropped)
		if produce_labels:
			return features, self.reverse_vocab.get(image_path.split("/")[-2])
		else:
			return features

	def build_dataset(self):
		X, y = [], []
		print("Treating images.")
		images = glob.glob('data/page_*/*.jpg')
		random.shuffle(images)
		images = images
		for image in tqdm.tqdm(images):
			features, label = self.load_image(image, produce_labels=True)
			X.append(features)
			y.append(label)
		return X, y

	def train(self,
			  model_path=None,
			  vocab_path=None,
			  show_features=False):
		inputs, labels = self.build_dataset()
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)
		rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
		# fit the model to the training set
		rfc_100.fit(X_train, y_train)
		accuracy = rfc_100.score(X_test, y_test)
		print(accuracy)
		# https://stackoverflow.com/a/20662980
		joblib.dump(rfc_100, model_path)
		joblib.dump(self.vocab, vocab_path)

	def predict(self,
				model_path=None,
				vocab_path=None,
				debug_model=False):
		model = joblib.load(model_path)
		vocab = joblib.load(vocab_path)
		images = glob.glob('/home/mgl/Bureau/Travail/projets/Front_Justice/htr-front-justice/data/all_images/*.jpg')
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
			test_image = self.load_image(image_path=image, show_image=False)
			prediction = model.predict([test_image])
			out_dir = vocab[prediction[0]]
			try:
				os.mkdir(f"data/predictions/{out_dir}")
			except FileExistsError:
				pass
			shutil.copyfile(image, f"data/predictions/{out_dir}/{image.split('/')[-1]}")

	# Predict on the test set results


if __name__ == '__main__':
	classifier = RandomForestImageClassifier(build_vocab=False)
	# classifier.train(model_path="models/PageClassifier.joblib", vocab_path="models/vocab.joblib")
	classifier.predict(model_path="models/PageClassifier.joblib", vocab_path="models/vocab.joblib")
