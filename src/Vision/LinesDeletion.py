import argparse
import itertools
import os
import shutil
import sys
import torch
import torch.nn as nn
import src.utils.utils as utils
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import tqdm
import evaluate
import numpy as np

from src.utils.utils import OCRLine


class LinesDeletionsIdentifier():
	def __init__(self, model_lines, model_chars):
		self.dictionary =  {0: "deleted", 1: "undeleted"}
		self.model_lines = model_lines
		self.model_chars = model_chars
		self.resize_chars = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize((65, 65))  # Normaliser à [-1, 1]
		])
		self.normalize_chars = transforms.Compose([
			transforms.Normalize(mean=[0.5], std=[0.5]),  # Normaliser à [-1, 1]
		])

		self.transform_lines = transforms.Compose([
			transforms.Resize((65, 1500)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5], std=[0.5]),  # Normaliser à [-1, 1]
		])


		self.model_chars = SimpleCNN(num_classes=2).to("cpu")
		state_dict = torch.load(model_chars, weights_only=True,
								map_location=torch.device("cpu"))
		self.model_chars.load_state_dict(state_dict)

		self.model_lines = SimpleCNN(num_classes=2).to("cpu")
		state_dict = torch.load(model_lines, weights_only=True,
								map_location=torch.device("cpu"))
		self.model_lines.load_state_dict(state_dict)


	def predict(self, image, model):
		with torch.no_grad():
			image = image.to("cpu")
			outputs = model(image)
			maxes = np.argmax(outputs, axis=1)
			return maxes

	def identify_deletions_in_line(self, line:OCRLine, image):
		cropped = utils.polygon_extraction(line.polygon, image, keep_alpha=False, return_image=True)
		cropped = cropped.convert("L")
		normalized = self.transform_lines(cropped)
		# Shape [1, height, width]
		normalized = normalized.unsqueeze(1)
		# Shape [1, 1, height, width]
		preds = self.predict(model=self.model_lines, image=normalized)
		preds = self.dictionary[preds.tolist()[0]]
		if preds == "deleted":
			imgs = []
			image = np.asarray(image)
			for char_poly, char in zip(line.cuts, line.prediction):
				current_char = utils.polygon_extraction(char_poly, image, keep_alpha=False, return_image=True, vertical_padding=12)
				current_char = current_char.convert("L")
				resized = self.resize_chars(current_char)
				imgs.append(resized)
			imgs = torch.stack(imgs)
			normalized = self.normalize_chars(imgs)
			preds = self.predict(model=self.model_chars, image=normalized).tolist()
			preds = [self.dictionary[pred] for pred in preds]
			zipped = list(zip(line.prediction, preds))
			recreate_words = {}
			word_n = 0
			for char, pred in zipped:
				if char == " ":
					word_n += 1
					continue
				try:
					recreate_words[word_n].append((char, pred))
				except KeyError:
					recreate_words[word_n] = [(char, pred)]
			sentence = ""
			for word in recreate_words.values():
				# Cas simple 1: tous les caractères sont gardés
				if all([item[1] == "undeleted" for item in word]):
					sentence += "".join([char[0] for char in word])
					sentence += " "
				# Cas simple 2: tous les caractères sont supprimé
				elif all([item[1] == "deleted" for item in word]):
					sentence += "⟦"
					sentence += "".join([char[0] for char in word])
					sentence += "⟧ "
				# Cas autres
				else:
					# Si un seul caractère est concerné, on considère que c'est une erreur de prédiction
					if [char[1] for char in word].count("deleted") == 1 and len(word) != 1:
						sentence += "".join([char[0] for char in word])
						sentence += " "
					elif word[0][-1] == word[-1][-1] == "undeleted" and all([char[-1] == "deleted" for char in word[1:-1]]) and len(word) != 1:
						sentence += "⟦"
						sentence += "".join([char[0] for char in word])
						sentence += "⟧ "
					# Si un seul caractère est concerné, on considère que c'est une erreur de prédiction
					elif [char[1] for char in word].count("undeleted") == 1 and len(word) != 1:
						sentence += "⟦"
						sentence += "".join([char[0] for char in word])
						sentence += "⟧ "
					else:
						preds = [char[-1] for char in word]
						chars = [char[0] for char in word]
						res = [list(y) for x, y in itertools.groupby(preds)]
						correct = []
						for idx, group in enumerate(res):
							# Si on a une lettre isolée qui comprend une déletion
							if len(group) == 1 and group[0] == 'deleted':
								correct.append("undeleted")
							else:
								correct.extend([item for item in group])
						correct_correct = [list(y) for x, y in itertools.groupby(correct)]
						n = 0
						outs = ""
						for group in correct_correct:
							reconstructed = []
							for char in group:
								reconstructed.append(chars[n])
								n += 1
							reconstructed = "".join(reconstructed)
							if group[0] == "deleted":
								outs += f"⟦{reconstructed}⟧"
							else:
								outs += f"{reconstructed}"
						sentence += f"{outs} "
			sentence = sentence.replace("⟦⟦", "⟦").replace("⟧⟧", "⟧")
			print(f"Reconstructed sentence: {sentence}")
			return sentence
		else:
			return line.prediction



class SimpleCNN(nn.Module):
	def __init__(self, num_classes):
		super(SimpleCNN, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # Accepte n'importe quelle taille
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 4 * 4, 512),
			nn.ReLU(),
			nn.Linear(512, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = self.classifier(x)
		return x
