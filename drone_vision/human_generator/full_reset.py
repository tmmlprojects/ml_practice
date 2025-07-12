import os
import shutil
from pathlib import Path

# Define paths
PREDICTIONS_DIR = Path("predictions")
DATASET_DIR = Path("dataset")
RUNS_DIR = Path("runs/detect")
WANDB_DIR = Path("wandb")

print("⚠️ This will delete:")
print(" - Entire 'predictions' folder")
print(" - All label files in 'dataset/labels/{train,val,test}'")
print(" - YOLOv8 training outputs in 'runs/detect'")
print(" - Any '.cache' files under 'dataset/'")
confirm = input("❓ Proceed with reset? Type 'yes' to continue: ").strip().lower()

if confirm != 'yes':
    print("❌ Reset cancelled.")
    exit()

# 1. Delete prediction visualization images
if PREDICTIONS_DIR.exists():
    shutil.rmtree(PREDICTIONS_DIR)
    print("🧼 Deleted 'predictions' folder")

# 2. Delete only train/val/test label folders (not images)
for split in ["train", "val", "test"]:
    label_folder = DATASET_DIR / "labels" / split
    if label_folder.exists():
        shutil.rmtree(label_folder)
        print(f"🧼 Deleted labels: {label_folder}")

# 3. Delete YOLO training outputs
if RUNS_DIR.exists():
    shutil.rmtree(RUNS_DIR)
    print("🧼 Deleted training runs in 'runs/detect'")

# 4. Delete .cache files
for cache_file in DATASET_DIR.rglob("*.cache"):
    cache_file.unlink()
    print(f"🧼 Deleted cache file: {cache_file}")

# OPTIONAL: Delete .pt model weights
delete_weights = input("❓ Delete all YOLO '.pt' model weights in 'runs/detect'? Type 'yes' to confirm: ").strip().lower()
if delete_weights == 'yes':
    for pt_file in Path("runs").rglob("*.pt"):
        pt_file.unlink()
        print(f"🧼 Deleted model weight: {pt_file}")

# OPTIONAL: Delete local wandb logs
delete_wandb = input("❓ Delete local wandb logs? Type 'yes' to confirm: ").strip().lower()
if delete_wandb == 'yes' and WANDB_DIR.exists():
    shutil.rmtree(WANDB_DIR)
    print("🧼 Deleted local 'wandb/' logs")

print("✅ Full reset complete.")