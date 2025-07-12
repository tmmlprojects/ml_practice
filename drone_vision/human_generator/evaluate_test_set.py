from ultralytics import YOLO
from pathlib import Path

# Path to best trained model
MODEL_PATH = "runs/detect/yolov8-topdown-v1/weights/best.pt"
TEST_IMAGES_DIR = Path("dataset/images/test")
PREDICTIONS_DIR = Path("predictions/test")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Load the trained model
model = YOLO(MODEL_PATH)

# Evaluate on test set (metrics only)
metrics = model.val(data="dataset/data.yaml", split="test")
print("\n📊 Evaluation Metrics on Test Set:")
print(metrics)

# Run predictions and save visualized images
print("\n🖼️ Saving prediction images...")
results = model.predict(source=str(TEST_IMAGES_DIR), save=True, save_dir=str(PREDICTIONS_DIR), conf=0.1)
print(f"✅ Predictions saved to: {PREDICTIONS_DIR}")