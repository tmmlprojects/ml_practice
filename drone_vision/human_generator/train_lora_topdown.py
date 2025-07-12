# train_lora_topdown.py

import torch
import os
from torch.utils.data import DataLoader
from dataset_loader import DroneTopDownDataset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig, TaskType  # if you're using PEFT
import wandb

# --- W&B Setup ---
wandb.init(project="drone-human-lora", name="lora-train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load SD Pipeline ---
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
pipe.enable_attention_slicing()

unet = pipe.unet
vae = pipe.vae
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# --- Apply LoRA (example using PEFT) ---
# You can also use `lora_diffusion` or your own patching method
# Example below assumes PEFT for huggingface transformer-compatible training

# config = LoraConfig(r=8, lora_alpha=32, target_modules=["to_q", "to_k", "to_v"], task_type=TaskType.FEATURE_EXTRACTION)
# unet = get_peft_model(unet, config)

# --- Dataset ---
dataset = DroneTopDownDataset("dataset/images/train", "dataset/labels/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: {
    "pixel_values": torch.stack([i["pixel_values"] for i in x]),
    "labels": [i["labels"] for i in x]
})

optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

# --- Training Loop ---
for epoch in range(3):
    for step, batch in enumerate(loader):
        images = batch["pixel_values"].to(device, dtype=torch.float16)  # [B, 3, H, W]

        # 1. Encode images to latents
        latents = vae.encode(images).latent_dist.sample() * 0.18215

        # 2. Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 3. Prepare text embeddings from a fixed prompt
        prompt = ["aerial view of a person walking"] * latents.shape[0]
        input_ids = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        # 4. Forward UNet
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 5. Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

        # 6. Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})
        print(f"[Epoch {epoch}] Step {step} - Loss: {loss.item():.4f}")

# --- Save LoRA model weights
os.makedirs("lora_output", exist_ok=True)
torch.save(unet.state_dict(), "lora_output/lora_unet_weights.pt")

wandb.finish()
