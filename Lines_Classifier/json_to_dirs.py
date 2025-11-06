import json
import shutil

with open("corpus_v1.json", "r") as file:
	annotations = json.load(file)

for image in annotations:
	label = image['annotations'][0]['result'][0]['value']['choices'][0]
	image_path = image['data']['image'].split("/")[-1].split("-")[-1]
	shutil.move(f"lines/{image_path}", f"lines/{label}/{image_path}")