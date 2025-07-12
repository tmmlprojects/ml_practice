import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import wandb
from datetime import datetime
from tqdm import tqdm

# === Configs ===
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = f"outputs/lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Enable W&B
wandb.init(
    project="drone-human-lora",
    name=f"drone-train-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

# === Load components ===
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

vae: AutoencoderKL = pipe.vae
unet = pipe.unet

# === Dataset Placeholder (replace with your own) ===
class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # Simulated drone-top image: (3, 512, 512)
        return {
            "pixel_values": torch.randn(3, 512, 512),
            "prompt": "aerial view of a person walking on a road",
        }

    def __len__(self):
        return 100

dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# === Training ===
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=100)

unet.train()
unet.enable_gradient_checkpointing()
unet = get_peft_model(unet, LoraConfig(r=4, lora_alpha=16, target_modules=["to_k", "to_q", "to_v", "to_out.0"]))

for epoch in range(1):
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        pixel_values = batch["pixel_values"].to(device="cuda", dtype=torch.float16)
        pixel_values = pixel_values.unsqueeze(0) if pixel_values.ndim == 3 else pixel_values

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        # Simulate timestep and encoder_hidden_states
        timestep = torch.randint(0, 1000, (latents.shape[0],), device="cuda").long()
        encoder_hidden_states = torch.randn((latents.shape[0], 77, 768), device="cuda", dtype=torch.float16)

        noise = torch.randn_like(latents)
        noisy_latents = latents + noise  # Simulate diffusion step

        pred = unet(noisy_latents, timestep, encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        wandb.log({"loss": loss.item(), "step": step})

# === Save adapter ===
adapter_save_path = os.path.join(output_dir, "lora_adapter.bin")
torch.save(unet.state_dict(), adapter_save_path)
artifact = wandb.Artifact("lora_adapter", type="model")
artifact.add_file(adapter_save_path)
wandb.log_artifact(artifact)

print("✅ Training complete. Adapter saved to:", adapter_save_path)
