# split_dataset.py

import os
import random
from pathlib import Path
import shutil

# Set paths
SOURCE_IMAGES = Path("dataset/images/train")  # Where all labeled images are
SOURCE_LABELS = Path("dataset/labels/train")
DEST_DIR = Path("dataset")

# Target split folders
IMAGE_DIR = DEST_DIR / "images"
LABEL_DIR = DEST_DIR / "labels"

# Backup folder for unlabeled images
UNLABELED_BACKUP_DIR = DEST_DIR / "unlabeled_backup"
UNLABELED_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Create folders
for split in ["train", "val", "test"]:
    (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

# Gather all images in source
all_image_paths = list(SOURCE_IMAGES.glob("*.jpg")) + list(SOURCE_IMAGES.glob("*.png"))

# Separate labeled from unlabeled images
labeled_image_paths = [p for p in all_image_paths if (SOURCE_LABELS / f"{p.stem}.txt").exists()]
unlabeled_image_paths = [p for p in all_image_paths if p not in labeled_image_paths]

# Backup unlabeled images
for img_path in unlabeled_image_paths:
    shutil.copy(img_path, UNLABELED_BACKUP_DIR / img_path.name)
print(f"[INFO] Backed up {len(unlabeled_image_paths)} unlabeled images to {UNLABELED_BACKUP_DIR}")

# Shuffle labeled images and calculate split sizes
random.shuffle(labeled_image_paths)
total = len(labeled_image_paths)
train_split = int(total * 0.7)
val_split = int(total * 0.85)

train_images = labeled_image_paths[:train_split]
val_images = labeled_image_paths[train_split:val_split]
test_images = labeled_image_paths[val_split:]

def move_split(images, split):
    for img_path in images:
        label_path = SOURCE_LABELS / f"{img_path.stem}.txt"
        dst_img_path = IMAGE_DIR / split / img_path.name
        dst_label_path = LABEL_DIR / split / label_path.name

        if img_path.resolve() != dst_img_path.resolve():
            shutil.move(img_path, dst_img_path)
            shutil.move(label_path, dst_label_path)

# Move data
move_split(train_images, "train")
move_split(val_images, "val")
move_split(test_images, "test")

print("[DONE] Split complete:")
print(f"  - {len(train_images)} train")
print(f"  - {len(val_images)} val")
print(f"  - {len(test_images)} test")
