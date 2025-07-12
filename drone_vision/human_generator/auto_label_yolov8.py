# auto_label_yolov8.py

from ultralytics import YOLO
from pathlib import Path
import shutil

# Configuration
IMAGE_DIR = Path("dataset/images/train")
LABEL_DIR = Path("dataset/labels/train")
PREDICT_DIR = Path("predictions")
CONFIDENCE_THRESHOLD = 0.1
MODEL_NAME = "yolov8l.pt"

# Ensure output folders exist
LABEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLOv8 model
model = YOLO(MODEL_NAME)

# Run detection
results = model.predict(source=str(IMAGE_DIR), save=False, conf=CONFIDENCE_THRESHOLD)

# Loop through results
for result in results:
    boxes = result.boxes.xywhn
    classes = result.boxes.cls
    name = Path(result.path).name
    stem = Path(name).stem
    label_path = LABEL_DIR / f"{stem}.txt"
    pred_image_path = PREDICT_DIR / f"{stem}.jpg"

    lines = []
    for box, cls in zip(boxes, classes):
        if int(cls) != 0:
            continue  # Only keep class 0 (person)
        x, y, w, h = box.tolist()
        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if lines:
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        print(f"[OK] Labels saved for {name}")

        # Save visualized image with boxes
        result.save(filename=pred_image_path)
        print(f"[OK] Preview saved: {pred_image_path}")
    else:
        print(f"[SKIP] No person detected in {name}")
