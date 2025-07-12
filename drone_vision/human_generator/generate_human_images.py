import os
import sys
import torch
import random
import csv
from diffusers import StableDiffusionPipeline
from datetime import datetime
from PIL import Image
import wandb
from pathlib import Path

# === Force UTF-8 globally (Windows safe) ===
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# === Config ===
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
SAVE_IMAGE_DIR = Path("dataset/images/train")
SAVE_LABEL_DIR = Path("dataset/labels/train")
PROMPT_LOG_DIR = Path("logs/prompts")
METADATA_CSV = Path("logs/metadata.csv")

for d in [SAVE_IMAGE_DIR, SAVE_LABEL_DIR, PROMPT_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

POSITIVE_PROMPTS = [
    "4K high quality top-down drone view of a person walking alone on a dirt trail in the countryside, photorealistic, full body visible, clear shadow, early afternoon lighting",
    "overhead 4K high resolution drone view of a single person standing still in a wide concrete parking lot, top-down angle, ultra-realistic ground texture, sharp contrast, natural lighting",
    "aerial 4K drone perspective of a man walking on a paved urban street, facing away from the camera, photorealistic, midday shadows, clear textures",
    "overhead 4K drone shot of a person sitting in the middle of a grassy field, high quality, full body clearly framed, realistic color tones",
    "top-down 4K drone photo of a woman walking diagonally across a pedestrian crosswalk, clean lines, defined limbs, sharp high quality detail",
    "high quality overhead 4K drone image of a group of 5 people walking together in a park, aerial top-down view, natural spacing, realistic environment",
    "realistic 4K drone shot of a small crowd scattered across an open plaza, top-down angle, photorealistic shadows and spacing",
    "4K top-down drone perspective of pedestrians crossing a busy intersection, realistic occlusion, sharp contrast, natural lighting",
    "overhead 4K drone view of people gathered in a parking lot, partially obscured by vehicles or structures, photorealistic layout, clear geometry",
    "high-resolution 4K aerial view of several people sitting on benches and walking in a public square, top-down view, lifelike textures and perspective",
    "high-altitude 4K drone view of a person walking on a sidewalk, small bounding box, sharp clarity, photo-real",
    "low-altitude 4K drone image of a person waving at the camera in an empty parking lot, overhead view, ultra sharp and realistic",
    "medium-altitude 4K drone view of a jogger running through a trail, overhead angle, motion blur, natural environment, photorealistic detail",
    "top-down 4K drone perspective of a child and adult walking side by side on a suburban road, scale difference clear, realistic scenery",
    "overhead 4K drone shot of a person walking near a building, partially visible under an awning, realistic light and shadow",
    "aerial 4K drone view of a man walking directly toward the camera, full limbs visible, sharp light contrast, natural materials",
    "top-down 4K drone shot of a woman lying in a grassy area, full body visible, ultra sharp resolution, photoreal",
    "realistic 4K drone view of a person walking parallel to a fence, profile view from above, straight clean path, detailed ground",
    "high-res 4K aerial image of a person walking under tree cover, natural light and shadows, partial occlusion, photo-real",
    "overhead 4K drone shot of a person mid-stride in a crosswalk, arms swinging, captured in motion, detailed ground and clothing"
]

NEGATIVE_PROMPT = (
    "cartoon, anime, lowres, blurry, noisy, distorted, poor quality, bad anatomy, unrealistic shadows, overexposed, "
    "overprocessed, glitch, jpeg artifacts, text, watermark, out of frame, cropped, tiling, frame, border, grainy, low detail, 3d render"
)

NUM_IMAGES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = "fp16" if DEVICE == "cuda" else "fp32"

def generate_images(prompts, num_images=NUM_IMAGES, negative_prompt=None):
    wandb.init(project="drone-human-detection", name="generate-human-images")

    print(f"[INFO] Loading model with {PRECISION} on {DEVICE.upper()}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if PRECISION == "fp16" else torch.float32
    ).to(DEVICE)

    # Prepare metadata log file
    if not METADATA_CSV.exists():
        with open(METADATA_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "prompt", "timestamp", "width", "height"])

    for i in range(num_images):
        prompt = random.choice(prompts)
        print(f"[INFO] Generating image {i+1}/{num_images} with prompt: {prompt}")
        if negative_prompt:
            image = pipeline(prompt, negative_prompt=negative_prompt).images[0]
        else:
            image = pipeline(prompt).images[0]

        # Generate base filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_filename = f"gen_{timestamp}"
        image_path = SAVE_IMAGE_DIR / f"{base_filename}.jpg"
        label_path = SAVE_LABEL_DIR / f"{base_filename}.txt"
        prompt_path = PROMPT_LOG_DIR / f"{base_filename}.txt"

        # Save image
        image.save(image_path)

        # Save empty YOLO label
        label_path.write_text("")  # ready for manual annotation

        # Save prompt
        prompt_path.write_text(prompt, encoding="utf-8")

        # Save metadata
        with open(METADATA_CSV, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([base_filename + ".jpg", prompt, timestamp, image.width, image.height])

        # Log to W&B
        wandb.log({"generated_image": wandb.Image(image, caption=prompt)})

    print("[DONE] All images generated.")

if __name__ == "__main__":
    generate_images(POSITIVE_PROMPTS, negative_prompt=NEGATIVE_PROMPT)

