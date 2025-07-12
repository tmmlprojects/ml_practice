import os
import requests
from pathlib import Path

# Local path to save the model
TARGET_DIR = Path("models/runwayml-stable-diffusion-v1-5")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Files to download (URL and target filename)
files = {
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer_config.json": "tokenizer_config.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vocab.json": "vocab.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/merges.txt": "merges.txt",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json": "model_index.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/config.json": "config.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin": "unet/diffusion_pytorch_model.bin",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json": "unet/config.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin": "vae/diffusion_pytorch_model.bin",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/config.json": "vae/config.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin": "text_encoder/pytorch_model.bin",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json": "text_encoder/config.json",
}

# Download each file
for url, rel_path in files.items():
    out_path = TARGET_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Downloading {url} → {out_path}")
    response = requests.get(url)
    if response.status_code == 200:
        out_path.write_bytes(response.content)
        print(f"✅ Saved to {out_path}")
    else:
        print(f"❌ Failed to download {url}: {response.status_code}")

print("\n✅ All downloads attempted. You can now run your training script.")
