# cleanup_for_retrain.py

import os
import argparse
import sys
from pathlib import Path
from collections import Counter

# Fix encoding issue for Windows consoles
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path("dataset")
IMAGE_ROOT = BASE_DIR / "images"
LABEL_ROOT = BASE_DIR / "labels"
IMAGE_EXTS = [".jpg", ".png"]

missing_label = []
missing_image = []
empty_labels = []
label_counts = Counter()

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Also clear images/train")
args = parser.parse_args()

# Ensure folder structure exists
print("\n--- Ensuring folder structure ---")
for split in ["train", "val", "test"]:
    (IMAGE_ROOT / split).mkdir(parents=True, exist_ok=True)
    (LABEL_ROOT / split).mkdir(parents=True, exist_ok=True)
print("Folder structure ensured.")

# Clean non-train images and all labels if desired
if args.full:
    print("\n--- Full clean: removing contents of all splits including images/train ---")
    for split in ["train", "val", "test"]:
        for file in (IMAGE_ROOT / split).glob("*"):
            if file.is_file():
                file.unlink()
        for file in (LABEL_ROOT / split).glob("*"):
            if file.is_file():
                file.unlink()
else:
    print("\n--- Cleaning all splits EXCEPT images/train ---")
    for split in ["val", "test"]:
        for file in (IMAGE_ROOT / split).glob("*"):
            if file.is_file():
                file.unlink()
    for split in ["train", "val", "test"]:
        for file in (LABEL_ROOT / split).glob("*"):
            if file.is_file():
                file.unlink()

# Check for mislocated .txt labels in any image split and auto-move them
print("\n--- Checking for misplaced label files in all image splits ---")
for split in ["train", "val", "test"]:
    wrong_label_dir = IMAGE_ROOT / split
    correct_label_dir = LABEL_ROOT / split

    misplaced_txt_files = list(wrong_label_dir.glob("*.txt"))
    if misplaced_txt_files:
        for txt_file in misplaced_txt_files:
            target_path = correct_label_dir / txt_file.name
            txt_file.rename(target_path)
            print(f"[MOVE] Misplaced label: {txt_file.name} -> labels/{split}/")
    else:
        print(f"[OK] No misplaced .txt files found in images/{split}/")

# Now proceed with standard validation
for split in ["train", "val", "test"]:
    print(f"\n--- Checking {split} split ---")
    img_dir = IMAGE_ROOT / split
    lbl_dir = LABEL_ROOT / split

    image_files = [f for f in img_dir.glob("*") if f.suffix.lower() in IMAGE_EXTS]
    label_files = list(lbl_dir.glob("*.txt"))

    image_stems = set(f.stem for f in image_files)
    label_stems = set(f.stem for f in label_files)

    for img in image_files:
        if (lbl_dir / f"{img.stem}.txt").exists() is False:
            missing_label.append(img)

    for lbl in label_files:
        if not any((img_dir / f"{lbl.stem}{ext}").exists() for ext in IMAGE_EXTS):
            missing_image.append(lbl)

    for lbl in label_files:
        if lbl.stat().st_size == 0:
            empty_labels.append(lbl)
        else:
            with lbl.open() as f:
                for line in f:
                    cls_id = line.strip().split()[0]
                    label_counts[cls_id] += 1

    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    print(f"[MISSING] Labels: {len(missing_label)}")
    print(f"[MISSING] Images: {len(missing_image)}")
    print(f"[WARN] Empty label files: {len(empty_labels)}")

print("\n--- Label Class Counts ---")
for cls, count in label_counts.items():
    print(f"Class {cls}: {count} instances")

if missing_label:
    print("\nMissing label files for:")
    for p in missing_label:
        print(" -", p)

if missing_image:
    print("\nMissing image files for:")
    for p in missing_image:
        print(" -", p)

if empty_labels:
    print("\nEmpty label files:")
    for p in empty_labels:
        print(" -", p)

print("\nDataset check complete.")
