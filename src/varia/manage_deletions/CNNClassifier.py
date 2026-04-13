import argparse
import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import tqdm
import evaluate
import numpy as np

def compute_metrics(predictions, labels):
    predictions = torch.stack(predictions)
    labels = torch.stack(labels)
    # load the metrics we want to evaluate
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("precision")
    metric4 = evaluate.load("f1")

    # get the label predictions
    predictions, labels = predictions.cpu(), labels.cpu()
    predictions = np.argmax(predictions, axis=2)

    # get the right format
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # automatically, value of -100 are produce ; we haven't understood why but we change them to 0. If not, it will give poor results
    ###
    labels = [0 if x == -100 else x for x in labels]
    ###
    # print(predictions)
    # print(labels)

    acc = metric1.compute(predictions=predictions, references=labels)
    recall = metric2.compute(predictions=predictions, references=labels, average=None)
    precision = metric3.compute(predictions=predictions, references=labels, average=None)
    f1 = metric4.compute(predictions=predictions, references=labels, average=None)
    recall = recall['recall'].tolist()
    precision = precision["precision"].tolist()
    f1 = f1["f1"].tolist()
    result = {"accuracy": acc, "recall": recall, "precision": precision, "f1": f1}
    print(result)
    return result

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("L")  # Charger en niveaux de gris
        if self.transform:
            image = self.transform(image)
        return image, label


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


# --- 7. Boucle d'entraînement ---
def train_model(train_loader, val_loader, num_epochs):
    model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

    model = models.resnet18(pretrained=False).to(DEVICE)

    # Adapter pour 1 canal (niveaux de gris)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(DEVICE)

    # Adapter pour N classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    all_accuracies = []
    all_f1 = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm.tqdm(train_loader, unit="step") as tstep:
            for images, labels in tstep:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                tstep.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        current_epoch_f1 = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                # On ignore le dernier batch s'il ne fait pas la taille du batch size
                if len(images) != BATCH_SIZE:
                    continue
                all_labels.append(labels)
                outputs = model(images)
                all_predictions.append(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        results = compute_metrics(predictions=all_predictions, labels=all_labels)
        f1_deleted = results['f1'][0]
        all_f1.append(f1_deleted)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        all_accuracies.append(val_acc)
        os.makedirs(exist_ok=True, name="models")
        torch.save(model.state_dict(), f"models/simple_cnn_grayscale_{epoch}.pth")

    best_epoch = all_f1.index(max(all_f1))
    print(f"Best F1: {max(all_f1)} for epoch {best_epoch}")
    print(f"Copying models/simple_cnn_grayscale_{best_epoch}.pth to models/simple_cnn_grayscale_best.pth")
    shutil.copy(f"models/simple_cnn_grayscale_{best_epoch}.pth", "models/simple_cnn_grayscale_best.pth")
# --- 8. Lancer l'entraînement ---


arguments = argparse.ArgumentParser()
arguments.add_argument("-d", "--device", default="cuda:0")
arguments.add_argument("-b", "--batch_size", default=32)
arguments.add_argument("-e", "--epochs", default=10)
arguments.add_argument("-w", "--workers", default=4)
arguments.add_argument("-lr", "--learning_rate", default=0.001)
arguments.add_argument("-i", "--input_dir", default="")
arguments = arguments.parse_args()
DEVICE = arguments.device
workers = int(arguments.workers)
BATCH_SIZE = int(arguments.batch_size)
NUM_EPOCHS = int(arguments.epochs)
LEARNING_RATE = float(arguments.learning_rate)
DATA_DIR = arguments.input_dir
# --- 1. Définir les paramètres ---
# DATA_DIR = "/media/mgl/stock/Front_Justice/data/HTR_data/data/main_text/extracted/data/lines_splits/"
IMAGE_SIZE = (1500, 65)  # Taille cible (après padding/redimensionnement)
NUM_CLASSES = 2  # À adapter selon votre nombre de classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normaliser à [-1, 1]
])

train_dataset = CustomDataset(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = CustomDataset(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)

train_model(train_loader, val_loader, NUM_EPOCHS)

# --- 9. Sauvegarder le modèle ---