import cv2
import os
from pathlib import Path
import shutil
from tqdm import tqdm

# ---------- SETTINGS ----------
INPUT_DIR = Path("dataset/images/train")
YOLO_OUT_DIR = Path("dataset/yolo_train_subset")
LORA_IMG_DIR = Path("dataset/lora_dataset/images")
LORA_CAP_DIR = Path("dataset/lora_dataset/captions")
TOP_YOLO = 1000
TOP_LORA = 200

# ---------- SETUP ----------
YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)
LORA_IMG_DIR.mkdir(parents=True, exist_ok=True)
LORA_CAP_DIR.mkdir(parents=True, exist_ok=True)

# ---------- SHARPNESS SCORING ----------
def calculate_sharpness(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

# ---------- LOAD AND SCORE ----------
all_images = sorted(INPUT_DIR.glob("*.jpg"))
scored_images = []

print("📊 Scoring image sharpness...")
for img_path in tqdm(all_images):
    score = calculate_sharpness(img_path)
    scored_images.append((img_path, score))

# ---------- SORT BY SHARPNESS ----------
scored_images.sort(key=lambda x: x[1], reverse=True)
top_yolo = scored_images[:TOP_YOLO]
top_lora = scored_images[:TOP_LORA]

# ---------- COPY FOR YOLO ----------
print(f"📦 Copying top {TOP_YOLO} images to YOLO training subset...")
for img_path, _ in top_yolo:
    shutil.copy(img_path, YOLO_OUT_DIR / img_path.name)
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        shutil.copy(txt_path, YOLO_OUT_DIR / txt_path.name)

# ---------- COPY FOR LORA ----------
print(f"🎨 Copying top {TOP_LORA} images to LoRA dataset...")
for i, (img_path, _) in enumerate(top_lora, start=1):
    new_name = f"img_{i:03}.jpg"
    caption_name = f"img_{i:03}.txt"
    shutil.copy(img_path, LORA_IMG_DIR / new_name)
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        shutil.copy(txt_path, LORA_CAP_DIR / caption_name)

print("✅ Selection complete.")