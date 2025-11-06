import json

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

def load_image(image, greyscale=True):
	if greyscale:
		image = Image.open(image).convert('L')
	else:
		image = Image.open(image)
	return image

def resize(image, x, y):
	image = image.resize((x, y))
	return image


def load_json_to_dict(path):
	with open(path, "r") as input_json:
		return json.load(input_json)

def save_image(image, path):
	image.save(path)

def crop_image(image, coordinates, show_image=False, resize=False, dimensions=None):
		image = image.crop(coordinates)
		if resize:
			image = image.resize(dimensions)
		if show_image:
			Image.Image.show(image)
		return image


def show_image(image):
	Image.Image.show(image)