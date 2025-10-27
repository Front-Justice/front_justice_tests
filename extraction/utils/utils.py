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


def crop_and_resize(image, vertical_crop_factor):
		height_resized = image.height // vertical_crop_factor
		image = image.crop((0, 0,  image.width, height_resized))
		# dims = (image.width // resize_factor, image.height // resize_factor)
		image = image.resize((1062, 391))
		# image = image.resize(dims)
		#print(image.size)
		#Image.Image.show(image)
		return image


def crop_image(image, coordinates, show_image=False, resize=False):
		image = image.crop(coordinates)
		if resize:
			image = image.resize((1062, 391))
		if show_image:
			Image.Image.show(image)
		return image


def show_image(image):
	Image.Image.show(image)