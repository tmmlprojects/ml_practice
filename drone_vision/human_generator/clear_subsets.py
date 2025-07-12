import shutil
from pathlib import Path

folders = [
    Path("dataset/yolo_train_subset"),
    Path("dataset/lora_dataset/images"),
    Path("dataset/lora_dataset/captions")
]

for folder in folders:
    print(f"🧹 Clearing: {folder}")
    for file in folder.glob("*"):
        file.unlink()