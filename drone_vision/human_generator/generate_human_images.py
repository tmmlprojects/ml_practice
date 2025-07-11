
import os
import random
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from pathlib import Path

# Settings
OUT_DIR = "generated_humans"
NUM_IMAGES = 20  # Change this number to generate more images
HEIGHT_OPTIONS = ['10ft', '30ft', '100ft']  # Simulate drone altitudes
ANGLES = ['top-down', 'oblique', 'slightly tilted', 'high angle']
RESOLUTION = (512, 512)

# Prompt templates
BASE_PROMPT = "a photo of a person or group of people from {} at {}, drone perspective, realistic, high detail"

print("🔍 PyTorch GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🎯 GPU Name:", torch.cuda.get_device_name(0))


def generate_prompts(num):
    prompts = []
    for _ in range(num):
        height = random.choice(HEIGHT_OPTIONS)
        angle = random.choice(ANGLES)
        prompt = BASE_PROMPT.format(height, angle)
        prompts.append(prompt)
    return prompts

def create_output_dir():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def generate_images(prompts, model="runwayml/stable-diffusion-v1-5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.float16 if device == "cuda" else torch.float32

    print(f"🖥️ Loading model with {precision} on {device.upper()}...")

    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=precision)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    for i, prompt in enumerate(prompts):
        image = pipe(prompt, height=RESOLUTION[1], width=RESOLUTION[0]).images[0]
        image.save(os.path.join(OUT_DIR, f"drone_view_{i+1:03}.png"))
        print(f"Saved: drone_view_{i+1:03}.png")

if __name__ == "__main__":
    create_output_dir()
    prompts = generate_prompts(NUM_IMAGES)
    generate_images(prompts)
