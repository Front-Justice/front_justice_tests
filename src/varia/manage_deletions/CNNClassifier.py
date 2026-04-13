import argparse
import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import tqdm



# --- 3. Créer un Dataset personnalisé ---
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

# --- 4. Charger les données ---
train_dataset = CustomDataset(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = CustomDataset(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Définir le modèle CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 canal (niveaux de gris)
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

model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

# --- 6. Définir la perte et l'optimiseur ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 7. Boucle d'entraînement ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    all_accuracies = []
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
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        all_accuracies.append(val_acc)
        os.makedirs(exist_ok=True, name="models")
        torch.save(model.state_dict(), f"models/simple_cnn_grayscale_{epoch}.pth")

    best_epoch = all_accuracies.index(max(all_accuracies))
    print(f"Best accuracy: {max(all_accuracies)} for epoch {best_epoch}")
    print(f"Copying models/simple_cnn_grayscale_{best_epoch}.pth to models/simple_cnn_grayscale_best.pth")
    shutil.copy(f"models/simple_cnn_grayscale_{best_epoch}.pth", "models/simple_cnn_grayscale_best.pth")
# --- 8. Lancer l'entraînement ---


arguments = argparse.ArgumentParser()
arguments.add_argument("-d", "--device", default="cuda:0")
arguments.add_argument("-b", "--batch_size", default=32)
arguments.add_argument("-e", "--epochs", default=10)
arguments.add_argument("-lr", "--learning_rate", default=0.001)
arguments.add_argument("-i", "--input_dir", default="")
arguments = arguments.parse_args()
DEVICE = arguments.device
BATCH_SIZE = arguments.batch_size
NUM_EPOCHS = arguments.epochs
LEARNING_RATE = arguments.learning_rate
DATA_DIR = arguments.input_dir
# --- 1. Définir les paramètres ---
# DATA_DIR = "/media/mgl/stock/Front_Justice/data/HTR_data/data/main_text/extracted/data/lines_splits/"
IMAGE_SIZE = (1500, 65)  # Taille cible (après padding/redimensionnement)
NUM_CLASSES = 2  # À adapter selon votre nombre de classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Définir les transformations ---
# Transformation pour les images en niveaux de gris
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.Resize(IMAGE_SIZE),  # Redimensionner (conserve le ratio)
    transforms.CenterCrop(IMAGE_SIZE),  # Recadrer au centre (optionnel)
    transforms.ToTensor(),  # Convertir en tenseur [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normaliser à [-1, 1]
])
train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

# --- 9. Sauvegarder le modèle ---