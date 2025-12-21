import glob
import pickle
import shutil
import sys
from multiprocessing import Pool
import PIL
import numpy as np
import PIL.Image as Image
import skimage
from skimage import color, data, exposure
import random
from skimage.transform import resize
import joblib
import os
import matplotlib.pyplot as plt

import shap

class PageClassifier():
	def __init__(self,
				 build_vocab=True,
				 corpus_path=None,
				 model=None,
				 vocab=None):
		if build_vocab:
			self.vocab, self.reverse_vocab = self.build_classes_vocab(path="data/page_classification/corpus/page_*")
		self.model_path = None
		self.corpus_path = corpus_path
		print(model)
		self.model = joblib.load(model)
		self.vocab = joblib.load(vocab)
		self.reverse_vocab = {value:key for key, value in self.vocab.items()}

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

	def extract_features(self, image, return_viz=False):
		if return_viz:
			hog_features, hog_image = skimage.feature.hog(
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
				visualize=False
			)
		if return_viz:
			return hog_features, hog_image
		else:
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

	def load_image_extract_features(self, image_path, produce_labels=False, show_image=False):
		image = self.load_to_greyscale(image_path)
		if show_image:
			Image.Image.show(image)
		cropped = self.crop_and_resize(image, vertical_crop_factor=4)
		features = self.extract_features(cropped)
		if produce_labels:
			return features, self.reverse_vocab.get(image_path.split("/")[-2])
		else:
			return features

	def retrieve_hog_and_label(self, image):
		return self.load_image_extract_features(image, produce_labels=True)

	def build_dataset(self):
		print("Treating images.")
		images = glob.glob('data/page_classification/corpus/page_*/*.jpg')
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

	def show_feature_importance(self, image, name):
		# Supposons que tu as un modèle Random Forest entraîné
		# model.feature_importances_ contient les importances des features
		correct_page = image.split("/")[-2]
		image = self.load_to_greyscale(image)
		cropped = self.crop_and_resize(image, vertical_crop_factor=4)
		image_as_array = np.array(cropped)
		features, hog_image = self.extract_features(cropped, return_viz=True)
		print(features.shape)

		# plt.imshow(image, cmap=plt.cm.gray)
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 5))
		import cv2
		plt.subplot(2, 1, 1)
		plt.imshow(image_as_array, cmap='gray')


		# Reshape les importances pour correspondre à la grille HOG
		pixels_per_cell = (16, 16)
		cells_per_block = (2, 2)

		# Calculer le nombre de blocs
		n_cells_x = image_as_array.shape[1] // pixels_per_cell[0]
		n_cells_y = image_as_array.shape[0] // pixels_per_cell[1]
		n_blocs_x = n_cells_x - (cells_per_block[0] - 1)
		n_blocs_y = n_cells_y - (cells_per_block[1] - 1)
		n_blocs_y, n_blocs_x = 23, 65
		pred = self.model.predict(features.reshape(1, -1))
		explainer = shap.TreeExplainer(self.model)
		features_reshaped = features.reshape(1, -1)
		shap_values = explainer.shap_values(features_reshaped)
		print("Nombre de classes dans shap_values :", len(shap_values))
		print("Forme de shap_values[0] :", shap_values[0].shape)  # Doit être (1, 53820)
		target_class = self.reverse_vocab[correct_page]
		print(f"Page: {correct_page}, class: {target_class}")
		shap_values_classe_0 = shap_values[0][:, target_class]  # Toutes les features pour la classe 0
		shap_grid = shap_values_classe_0.reshape(n_blocs_y, n_blocs_x, 36)  # Pour la classe 1
		shap_grid_norm = np.linalg.norm(shap_grid, axis=2)  # Norme des contributions
		shap_upscaled = np.kron(shap_grid_norm, np.ones((32, 32)))  # Taille du bloc = 32x32
		# image_hog_features = features HOG de ton image
		target_height, target_width = image_as_array.shape
		shap_upscaled_resized = resize(
			shap_upscaled,
			(target_height, target_width),
			order=0,  # Interpolation "nearest" pour éviter le flou
			preserve_range=True
		)
		plt.title(f"Correct class: {target_class}, target: {correct_page}")
		plt.subplot(2, 1, 2)
		plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		plt.title(f"Pred: {self.vocab[pred[0].item()]}")
		plt.imshow(shap_upscaled_resized, cmap='hot', alpha=0.5)
		plt.savefig(f"/home/mgl/Bureau/Travail/Communications_et_articles/Front_Justice_papier/heatmap_image_{name}_classe_{target_class}.png")

		return
		# Vérifier la taille du vecteur HOG
		print("Taille attendue :", n_blocs_x * n_blocs_y * 36)

		# Récupérer les importances du Random Forest
		# importances = self.model.feature_importances_
		#
		# # Reshape des importances (pour une seule classe)
		# importances_grid = importances.reshape(n_blocs_y, n_blocs_x, 36)
		#
		# # Calculer la norme des importances par bloc
		# importances_grid_norm = np.linalg.norm(importances_grid, axis=2)
		#
		# # Redimensionner pour superposer sur l'image
		# block_size = (cells_per_block[0] * pixels_per_cell[0], cells_per_block[1] * pixels_per_cell[1])
		# importances_upscaled = np.kron(importances_grid_norm, np.ones((block_size[1], block_size[0])))
		# importances_upscaled_resized = resize(importances_upscaled, image_as_array.shape, order=0, preserve_range=True)
		# print(importances_upscaled_resized.shape)
		# print(image_as_array.shape)
		# # Superposer
		# plt.imshow(image_as_array, cmap='gray')
		# plt.imshow(importances_upscaled_resized, cmap='hot', alpha=0.5)
		plt.savefig(f"/home/mgl/Bureau/Travail/Communications_et_articles/Front_Justice_papier/heatmap_{name}.png")

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

		test_image = self.load_image_extract_features(image_path=image, show_image=False)
		prediction = self.model.predict([test_image])
		return self.vocab[prediction[0]]


	# Predict on the test set results


if __name__ == '__main__':
	classifier = PageClassifier(build_vocab=False,
								corpus_path="/home/mgl/Bureau/Travail/projets/Front_Justice/"
								"alternative_pipeline/page_classification/data/page_classification/corpus.data",
								model="src/Page_Classifier/models/PageClassifier_RF.joblib",
								vocab="src/Page_Classifier/models/vocab_RF.joblib")
	if len(sys.argv) > 1:
		images = glob.glob(f"{sys.argv[1]}*.jpg")
		assert images != [], "No images found."
	random.shuffle(images)
	for idx, image in enumerate(images):
		idx += 1
		classifier.show_feature_importance(image, idx)