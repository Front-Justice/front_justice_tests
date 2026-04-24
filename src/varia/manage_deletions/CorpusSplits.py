import glob
import os
import shutil
import sys
import uuid

from sklearn.model_selection import train_test_split

SOURCE_DIR = sys.argv[1]
TARGET_DIR = sys.argv[2]

# Créer les dossiers train/val/test
os.makedirs(os.path.join(TARGET_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "test"), exist_ok=True)
all_deleted = glob.glob(f"{SOURCE_DIR}/deleted/*")
all_undeleted = glob.glob(f"{SOURCE_DIR}/undeleted/*")
target_ratio = sys.argv[3]
n_undeleted = len(all_undeleted)
n_deleted = round(float(target_ratio) * n_undeleted)
print(n_undeleted)
print(n_deleted)

for cls in os.listdir(SOURCE_DIR):
	print(cls)
	cls_dir = os.path.join(SOURCE_DIR, cls)
	if not os.path.isdir(cls_dir):
		continue

	images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
	if cls_dir == "deleted":
		images = images[:n_deleted + 1]
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