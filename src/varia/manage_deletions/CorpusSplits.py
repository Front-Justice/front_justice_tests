import os
import shutil
import uuid

from sklearn.model_selection import train_test_split

# Dossier source (ex: all_images/classe_A/, all_images/classe_B/)
# SOURCE_DIR = "/media/mgl/stock/Front_Justice/data/HTR_data/data/main_text/extracted/data/lines/"
# TARGET_DIR = "/media/mgl/stock/Front_Justice/data/HTR_data/data/main_text/extracted/data/lines_splits"

SOURCE_DIR = ["/media/mgl/stock/Front_Justice/data/Deletions_data/print_lines/chars/",
			  "/media/mgl/stock/Front_Justice/data/Deletions_data/manuscript_lines/chars/"]
TARGET_DIR = "/media/mgl/stock/Front_Justice/data/Deletions_data/chars_splits"


# Créer les dossiers train/val/test
os.makedirs(os.path.join(TARGET_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "test"), exist_ok=True)

# Pour chaque classe
for dir in SOURCE_DIR:
	for cls in os.listdir(dir):
		cls_dir = os.path.join(dir, cls)
		if not os.path.isdir(cls_dir):
			continue

		# Lister toutes les images de la classe
		images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

		train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
		val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

		# Créer les dossiers de destination
		for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
			target_cls_dir = os.path.join(TARGET_DIR, split, cls)
			os.makedirs(target_cls_dir, exist_ok=True)
			for img in imgs:
				src = os.path.join(cls_dir, img)
				dst = os.path.join(target_cls_dir, img)
				if os.path.isfile(dst):
					if os.path.getsize(dst) != os.path.getsize(src):
						print("---")
						print(os.path.getsize(dst))
						print(os.path.getsize(src))
						print("Two different files, same name")
						print("---")
						dst = dst.replace(".png", "") + str(uuid.uuid4()) + ".png"
					else:
						continue
				shutil.copy(src, dst)

print("Organisation des données terminée !")