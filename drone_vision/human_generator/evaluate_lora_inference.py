"""
evaluate_lora_inference.py

Evaluates LoRA-generated images using detection-aware metrics via YOLOv8.
Logs results to Weights & Biases.
"""

import os
import wandb
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from PIL import Image
import shutil

# ========== CONFIG ==========
RUN_NAME = "lora-eval-" + datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_NAME = "drone-human-detection"
WANDB_ENTITY = "tsmatthx2-oklahoma-state-university"
IMAGE_DIR = "inference_outputs"
MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
RESULTS_DIR = f"eval_results/{{RUN_NAME}}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========== INIT WANDB ==========
wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY, name=RUN_NAME)

# ========== LOAD YOLOv8 MODEL ==========
model = YOLO(MODEL_PATH)

# ========== RUN DETECTION ==========
results = model.predict(
    source=IMAGE_DIR,
    save=True,
    save_txt=True,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    project=RESULTS_DIR,
    name="preds",
    verbose=True
)

# ========== LOG RESULTS ==========
pred_dir = Path(RESULTS_DIR) / "preds"
wandb.log({{"eval/detection_output": [wandb.Image(str(img)) for img in pred_dir.glob("*.jpg")]}})
wandb.alert(title="LoRA Evaluation Completed", text=f"Run saved to {{RESULTS_DIR}}")
wandb.finish()

print(f"✅ Evaluation complete. Results saved to {{RESULTS_DIR}}")
