import lxml.etree as ET
import sys
import glob
from PIL import Image, ImageDraw
import numpy as np
import random
import cv2 as cv

random.seed(1234)
input_files = glob.glob(f"{sys.argv[1]}*.xml")
namespaces = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

def convert_xml_polygon_to_list_of_tuples(string):
	as_list = string.split(" ")
	x_values = [int(item) for idx, item in enumerate(as_list) if idx % 2 == 0]
	y_values = [int(item) for idx, item in enumerate(as_list) if idx % 2 == 1]
	coordinates = list(zip(x_values, y_values))
	return coordinates


current_line_number = 0
random.shuffle(input_files)
all_widths = []
all_heights = []
for file in input_files[201:300]:
	corresponding_image = file.replace(".xml", ".jpg")
	file_as_xml = ET.parse(file)
	all_lines = file_as_xml.xpath("//alto:TextLine/alto:Shape/alto:Polygon", namespaces=namespaces)
	assert len(all_lines) != [], "Problem with namespace"
	# https://stackoverflow.com/a/22650239
	for idx, line in enumerate(all_lines):
		current_line_number += 1000000
		coordinates = line.xpath("@POINTS")[0]
		coordinates = convert_xml_polygon_to_list_of_tuples(coordinates)
		im = Image.open(corresponding_image).convert("RGBA")
		imArray = np.asarray(im)
		# 0 creates transparent background. 1 makes the mask
		maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 1)
		ImageDraw.Draw(maskIm).polygon(coordinates, outline=1, fill=1)
		mask = np.array(maskIm)
		# assemble new image (uint8: 0-255)

		newImArray = np.empty(imArray.shape, dtype='uint8')
		# colors (three first columns, RGB)
		newImArray[:, :, :3] = imArray[:, :, :3]
		# transparency (4th column)
		newImArray[:, :, 3] = mask * 255
		# back to Image from numpy
		newIm = Image.fromarray(newImArray, "RGBA")

		x_max = max([i[0] for i in coordinates])
		x_min = min([i[0] for i in coordinates])
		y_max = max([i[1] for i in coordinates])
		y_min = min([i[1] for i in coordinates])
		rectangle_coordinates = (x_min, y_min, x_max, y_max)
		cropped_img = newIm.crop(rectangle_coordinates)
		width, height = cropped_img.size
		resized = cropped_img.resize((int(width/2), int(height/2)))
		all_widths.append(width/2)
		all_heights.append(height/2)
		resized.save(f"lines/v_3{current_line_number}.png")

max_width = max(all_widths)
max_height = max(all_heights)

print(f"Maximum image size: {max_width},{max_height}")