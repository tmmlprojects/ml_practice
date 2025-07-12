import torch
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig

model_path = "runwayml/stable-diffusion-v1-5"
lora_weights = "lora_output/lora_unet_weights.pt"
prompt = (
    "top-down drone photo of people walking on a paved road from 20ft, "
    "long shadows, heads not visible"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Apply LoRA to the UNet
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.load_state_dict(torch.load(lora_weights), strict=False)
pipe.unet.eval()

# Generate image
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("inference_output.png")
print("Image saved as inference_output.png")
