import glob
import shutil
import utils.utils as utils
import tqdm
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import skimage
from skimage import color, data, exposure
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
				 corpus_path=None):
		if build_vocab:
			self.vocab, self.reverse_vocab = self.build_classes_vocab(path="../data/name_extraction/corpus/*")
		else:
			self.vocab, self.reverse_vocab = None, None
		self.model_path = None
		self.model = None
		self.corpus_path = corpus_path
		assert self.vocab is not None, "Vocab is None, check path."

	def crop_and_resize(self, image, vertical_crop_factor):
		height_resized = image.height // vertical_crop_factor
		image = image.crop((0, 0, image.width, height_resized))
		# dims = (image.width // resize_factor, image.height // resize_factor)
		image = image.resize((1062, 391))
		# image = image.resize(dims)
		#print(image.size)
		#Image.Image.show(image)
		return image


	def extract_features(self, image, show_features=False):
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
			x, y = 1062, 391
			try:
				image = utils.resize(image, x, y)
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
		coefficient = 4
		for index, image in enumerate(tqdm.tqdm(images)):
			result = self.process_image(image, produce_labels=True, resize=False)
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
		return X, y



	def train(self,
			  model_path=None,
			  vocab_path=None):
		inputs, labels = self.load_corpus()
		X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
		print("Training...")
		my_model = svm.SVC()
		my_model = RandomForestClassifier(n_estimators=200,
										 random_state=0)
		# fit the model to the training set
		my_model.fit(X_train, y_train)
		accuracy = my_model.score(X_test, y_test)
		y_pred = my_model.predict(X_test)
		print(classification_report(y_pred, y_test))
		# https://stackoverflow.com/a/20662980
		joblib.dump(my_model, model_path)
		joblib.dump(self.vocab, vocab_path)


	def slide_over_image(self, image_name):
		# Pour les rectangles
		if isinstance(image_name, str):
			loaded = utils.load_image(image_name, greyscale=True)
		else:
			loaded = image_name
		TINT_COLOR = (0, 0, 0)  # Black
		TRANSPARENCY = .25  # Degree of transparency, 0-100%
		OPACITY = int(255 * TRANSPARENCY)


		mean_width = 1062
		mean_height = 391
		sliding_value_x = 0.05
		sliding_value_y = 0.04

		loaded_as_rgb = Image.merge("RGB", (loaded, loaded, loaded))
		draw = ImageDraw.Draw(loaded_as_rgb, "RGBA")
		original_width, original_height = loaded.size

		# overlay = Image.new('RGBA', loaded.size, TINT_COLOR + (0,))
		# draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
		# draw.rectangle(((0, 0), (949, 355)), fill=TINT_COLOR + (OPACITY,))


		# img = Image.alpha_composite(loaded, overlay)
		# img = img.convert("RGB")  # Remove alpha for saving in jpg format.
		# Extract True value
		all_x = [n * sliding_value_x * original_width
				 for n in range(0, int(1 / sliding_value_x))]
		all_y = [n * sliding_value_y * original_height
				 for n in range(0, int(1 / sliding_value_y))]

		for idx_x, x in enumerate(all_x):
			for idx_y, y in enumerate(all_y):
				print("---")
				good_coordinates = (x, y, x + mean_width, y + mean_height)
				current_rectangle = Rectangle(good_coordinates[0],
											  good_coordinates[1],
											  good_coordinates[2],
											  good_coordinates[3])
				cropped = utils.crop_image(loaded, good_coordinates, show_image=False)
				hog = self.process_image(cropped, produce_labels=False, load_image=False)
				print(hog.shape)
				print(current_rectangle)
				prediction = self.model.predict([hog])[0]
				if prediction == 0:
					draw.rectangle(((good_coordinates[0], good_coordinates[1]), (good_coordinates[2], good_coordinates[3])),
								   fill=(255, 52, 0, 20))

		utils.show_image(loaded_as_rgb)
		exit(0)


	def predict(self,
				model_path=None,
				vocab_path=None,
				debug_model=False):
		self.model = joblib.load(model_path)
		self.vocab = joblib.load(vocab_path)
		images = glob.glob('/home/mgl/Bureau/Travail/projets/Front_Justice/alternative_pipeline/page_classification/data'
						   '/page_classification/predictions/page_1/11_J_76_0009.jpg')
		image = utils.load_image(images[0], greyscale=True)
		x, y = image.size
		image = utils.crop_image(image, coordinates=(0, y/2, x, y), resize=False)
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
											 corpus_path="../data/name_extraction/corpus/corpus.data")
	classifier.process_corpus()
	classifier.train(model_path="../models/name/NameClassifier.joblib", vocab_path="../models/name/vocab.joblib")
	classifier.predict(model_path="../models/name/NameClassifier.joblib", vocab_path="../models/name/vocab.joblib")
