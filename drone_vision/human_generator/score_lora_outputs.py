from datetime import datetime
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# === Configuration ===
RUN_NAME = f"scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SCORED_DIR = Path("scored_images") / RUN_NAME
SCORED_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR = Path("dataset/images/train")
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Change if needed

# === Load YOLO model ===
model = YOLO(MODEL_PATH)
print(f"🔍 Running detection on {IMG_DIR}...")

# === Run inference and record scores ===
results = model.predict(source=str(IMG_DIR), save=False, conf=0.001, verbose=False)
data = []

for result in results:
    image_path = Path(result.path)
    predictions = result.boxes
    high_conf_preds = [float(box.conf[0]) for box in predictions if float(box.conf[0]) >= CONFIDENCE_THRESHOLD]

    data.append({
        "filename": image_path.name,
        "confidence_sum": sum(high_conf_preds),
        "num_detections": len(high_conf_preds),
        "high_conf_ratio": len(high_conf_preds) / max(1, len(predictions))
    })

    # Save high-confidence visuals
    if sum(high_conf_preds) >= 1.0:
        result.save(filename=str(SCORED_DIR / image_path.name))

# === Save summary ===
df = pd.DataFrame(data)
df = df.sort_values("confidence_sum", ascending=False)
df.to_csv(SCORED_DIR / "detection_scores.csv", index=False)
print(f"✅ Scored detections saved to {SCORED_DIR}")

# === Optionally copy best images to LoRA training folder ===
RETRAIN_DIR = Path("lora_refined/images")
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
for img_name in df[df["confidence_sum"] >= 1.0]["filename"]:
    src = IMG_DIR / img_name
    dst = RETRAIN_DIR / img_name
    if src.exists():
        dst.write_bytes(src.read_bytes())

print(f"📦 {len(df[df['confidence_sum'] >= 1.0])} images prepared for next LoRA training.")
