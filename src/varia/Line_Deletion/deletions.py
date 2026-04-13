import argparse
import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import tqdm
import evaluate
import numpy as np


def transform(image_size):
	transform = transforms.Compose([
		transforms.Resize(image_size),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5], std=[0.5]),  # Normaliser à [-1, 1]
	])
	return transform


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



def predict(model_path, image):
	dict = {0: "deleted", 1: "undeleted"}
	model = SimpleCNN(num_classes=2)
	model.load_state_dict(torch.load(model_path, weights_only=True))
	with torch.no_grad():
		image = image.to("cpu")
		outputs = model(image)
		return dict[np.argmax(outputs).item()]