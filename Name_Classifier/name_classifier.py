import glob
import shutil

import numpy as np
import pylab
import utils.utils as utils
import tqdm
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import skimage
from skimage import color, data, exposure, measure
from skimage import io
import random
import cv2
import joblib
import matplotlib.pyplot as plt
import pickle


# https://stackoverflow.com/a/27162334
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


class RandomForestImageClassifier():
	def __init__(self,
				 build_vocab=True,
				 corpus_path=None,
				 params_file=None):
		if build_vocab:
			self.vocab, self.reverse_vocab = self.build_classes_vocab(path="../data/name_extraction/corpus/*")
		else:
			self.vocab, self.reverse_vocab = None, None
		self.model_path = None
		self.model = None
		self.corpus_path = corpus_path
		assert self.vocab is not None, "Vocab is None, check path."
		self.params_file = utils.load_json_to_dict(params_file)



	def extract_features(self, image, show_features=False):
		# image = image.resize((200, 200))
		if show_features:
			fd, hog_image = skimage.feature.hog(
				image,
				orientations=8,
				pixels_per_cell=(8, 8),
				cells_per_block=(1, 1),
				visualize=True,
			)
			print(fd.shape)
		else:
			try:
				hog_features = skimage.feature.hog(
					image,
					orientations=8,
					pixels_per_cell=(8, 8),
					cells_per_block=(1, 1),
					visualize=show_features
				)
			except OSError:
				return None
		if show_features:
			self.reveal_hog_features(image=image, hog_image=hog_image)
			exit(0)
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
		plt.show()
		plt.savefig('../assets/foo.png', dpi=300)
		exit(0)

	def build_classes_vocab(self, path):
		vocab = {idx: dir.split("/")[-1] for idx, dir in enumerate(glob.glob(path))}
		reverse_vocab = {value: key for key, value in vocab.items()}
		return vocab, reverse_vocab

	def process_image(self, image_path, produce_labels=False, show_image=False, load_image=True, resize=False):
		if load_image is True:
			image = utils.load_image(image_path)
		else:
			image = image_path
		if show_image:
			Image.Image.show(image)
		if resize:
			try:
				image = utils.resize(image, self.params_file['dims'][0], self.params_file['dims'][1])
			except OSError:
				return None
		features = self.extract_features(image)
		if features is None:
			return None
		if produce_labels:
			return features, self.reverse_vocab.get(image_path.split("/")[-2])
		else:
			return features

	def build_dataset(self):
		X, y = [], []
		print("Treating images.")
		images = glob.glob('../data/name_extraction/corpus/*/*.png')
		random.shuffle(images)
		coefficient = 3
		for index, image in enumerate(tqdm.tqdm(images)):
			result = self.process_image(image, produce_labels=True, resize=True)
			count_false = y.count(1)
			count_true = y.count(0)
			if result is not None and index > 0:
				features, label = result
				X.append(features)
				y.append(label)
			elif result is not None and count_true < count_false * coefficient:
				features, label = result
				X.append(features)
				y.append(label)
		print(f"Corpus size: {len(X)}")
		print(f"True labels: {count_true}.\n"
			  f"False labels: {count_false}.")
		return X, y



	def train(self,
			  model_path=None,
			  vocab_path=None):
		inputs, labels = self.load_corpus()
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
		print("Training...")
		param_dist = {
			'n_estimators': range(50, 300, 40),
			'max_depth': [None, 10, 20, 30],
			'min_samples_split': range(2, 11),
			'min_samples_leaf': range(1, 5),
			'max_features': ['sqrt', 'log2']
		}
		my_model = RandomForestClassifier(random_state=42)
		random_search = RandomizedSearchCV(my_model,
										   param_distributions=param_dist,
										   n_iter=10,
										   cv=5,
										   n_jobs=11,
										   verbose=2)
		# my_model = svm.SVC()
		# fit the model to the training set
		random_search.fit(X_train, y_train)
		best_rf = random_search.best_estimator_
		y_pred = best_rf.predict(X_test)
		print(classification_report(y_pred, y_test))
		# https://stackoverflow.com/a/20662980
		joblib.dump(best_rf, model_path)
		joblib.dump(self.vocab, vocab_path)


	def slide_over_image(self, image):
		# Pour les rectangles
		image_name = image.split("/")[-1]
		image = utils.load_image(image, greyscale=True)
		print(image_name)

		params = utils.load_json_to_dict("../data/name_extraction/params.json")
		mean_width,  mean_height = params["dims"][0], params["dims"][1]
		TINT_COLOR = (0, 0, 0)  # Black
		TRANSPARENCY = .25  # Degree of transparency, 0-100%
		OPACITY = int(255 * TRANSPARENCY)


		sliding_value_x = 0.03
		sliding_value_y = 0.05

		# sliding_value_x = 0.07
		# sliding_value_y = 0.07

		loaded_as_rgb = Image.merge("RGB", (image, image, image))
		draw = ImageDraw.Draw(loaded_as_rgb, "RGBA")
		original_width, original_height = image.size

		# overlay = Image.new('RGBA', loaded.size, TINT_COLOR + (0,))
		# draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
		# draw.rectangle(((0, 0), (949, 355)), fill=TINT_COLOR + (OPACITY,))


		# img = Image.alpha_composite(loaded, overlay)
		# img = img.convert("RGB")  # Remove alpha for saving in jpg format.
		# Extract True value
		all_x = [n * sliding_value_x * original_width
				 for n in range(0, int(1 / sliding_value_x))]
		# all_x = [item for item in all_x if item < original_width]
		all_y = [original_height*.5 + (n * sliding_value_y * original_height)
				 for n in range(0, int(1 / sliding_value_y))]
		all_y = [item for item in all_y if item < original_height]
		contrast = 10
		mask = np.zeros(image.size).transpose()
		for idx_x, x in enumerate(tqdm.tqdm(all_x)):
			for idx_y, y in enumerate(all_y):
				# print(f"Coords: {x}, {y}")
				good_coordinates = (round(x), round(y), round(x + mean_width), round(y + mean_height))
				current_rectangle = Rectangle(good_coordinates[0],
											  good_coordinates[1],
											  good_coordinates[2],
											  good_coordinates[3])
				cropped = utils.crop_image(image, good_coordinates, show_image=False, resize=True, dimensions=(mean_width, mean_height))
				hog = self.process_image(cropped, produce_labels=False, load_image=False)
				prediction = self.model.predict([hog])[0]
				if prediction == 0:
					draw.rectangle(((good_coordinates[0], good_coordinates[1]), (good_coordinates[2], good_coordinates[3])),
								   fill=(255, 52, 0, 30))
					mask[good_coordinates[1]:good_coordinates[3],
						good_coordinates[0]:good_coordinates[2]] = mask[good_coordinates[1]:good_coordinates[3],
																		good_coordinates[0]:good_coordinates[2]] + contrast
					draw.rectangle(((good_coordinates[0], good_coordinates[1]), (good_coordinates[2], good_coordinates[3])),
							   fill=(0, 0, 0, 0), width=2, outline="red")
		# max_value = np.max(mask)
		# normalized_mask = np.divide(mask, max_value) * 255
		normalized_mask = mask < (contrast*2) - 1
		contours = measure.find_contours(normalized_mask, 0.8)

		for contour in contours:
			print("New contour")
			y_bottom = np.max(contour[:, 0])
			y_top = np.min(contour[:, 0])
			x_right = np.max(contour[:, 1])
			x_left = np.min(contour[:, 1])
			print(x_left, y_top, x_right, y_bottom)
			draw.rectangle(((x_left, y_top), (x_right, y_bottom)),
						   fill=(0, 0, 0, 0), width=5, outline="blue")
			padding = 200
			draw.rectangle(((x_left - padding, y_top - padding), (x_right + padding, y_bottom + padding)),
						   fill=(0, 0, 0, 0), width=5, outline="black")

			# On recommence sur le contour
			sliding_value_x = 0.1
			sliding_value_y = 0.1
			rectangle_width = (x_right + padding) - (x_left - padding)
			rectangle_height = (y_bottom + padding) - (y_top - padding)
			all_y = [(y_top - padding) + (n * sliding_value_y * rectangle_height)
					 for n in range(0, int(1 / sliding_value_y))]
			all_y = [int(item) for item in all_y if item < (y_bottom + padding - mean_height)]
			all_x = [(x_left - padding) + n * sliding_value_x * rectangle_width
					 for n in range(0, int(1 / sliding_value_x))]
			all_x = [int(item) for item in all_x if item < (x_right + padding - mean_width)]

			contrast = 10
			# mask = np.zeros(image.size).transpose()
			for idx_x, x in enumerate(tqdm.tqdm(all_x)):
				for idx_y, y in enumerate(all_y):
					print(f"Coords: {x}, {y}")
					good_coordinates = (round(x), round(y), round(x + mean_width), round(y + mean_height))
					current_rectangle = Rectangle(good_coordinates[0],
												  good_coordinates[1],
												  good_coordinates[2],
												  good_coordinates[3])
					cropped = utils.crop_image(image, good_coordinates, show_image=False, resize=True,
											   dimensions=(mean_width, mean_height))
					hog = self.process_image(cropped, produce_labels=False, load_image=False)
					prediction = self.model.predict([hog])[0]
					if prediction == 0:
						print("Positive result")
						draw.rectangle(
							((good_coordinates[0], good_coordinates[1]), (good_coordinates[2], good_coordinates[3])),
							fill=(220, 52, 0, 30))
						mask[good_coordinates[1]:good_coordinates[3],
						good_coordinates[0]:good_coordinates[2]] = mask[good_coordinates[1]:good_coordinates[3],
																   good_coordinates[0]:good_coordinates[2]] + contrast
						draw.rectangle(
							((good_coordinates[0], good_coordinates[1]), (good_coordinates[2], good_coordinates[3])),
							fill=(0, 0, 0, 0), width=2, outline="yellow")

		# utils.show_image(loaded_as_rgb)
		loaded_as_rgb.save(f"../data/name_extraction/corpus/predicted/{image_name.replace('.jpg', '.png')}")

	def predict(self,
				model_path=None,
				vocab_path=None,
				debug_model=False):
		self.model = joblib.load(model_path)
		self.vocab = joblib.load(vocab_path)
		images = glob.glob('/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/page_classification/data/page_classification/predictions/page_1/*.jpg')
		for image in images:
			# image = utils.crop_image(image, coordinates=(0, y/2, x*(2/3), y*(5/6)), resize=False)
			self.slide_over_image(image)

	def process_corpus(self):
		inputs, labels = self.build_dataset()
		with open(self.corpus_path, "wb") as corpus_file:
			pickle.dump((inputs, labels), corpus_file)

	def load_corpus(self):
		with open(self.corpus_path, "rb") as corpus_file:
			inputs, labels = pickle.load(corpus_file)
		return inputs, labels



if __name__ == '__main__':
	random.seed(1234)
	classifier = RandomForestImageClassifier(build_vocab=True,
											 corpus_path="../data/name_extraction/corpus/corpus.data",
											 params_file="../data/name_extraction/params.json",)
	# classifier.process_corpus()
	# classifier.train(model_path="../models/name/NameClassifier.joblib", vocab_path="../models/name/vocab.joblib")
	classifier.predict(model_path="../models/name/NameClassifier.joblib", vocab_path="../models/name/vocab.joblib")
