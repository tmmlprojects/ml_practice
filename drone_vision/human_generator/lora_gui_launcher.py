import os
import sys
import time
import json
import shutil
import datetime
import subprocess
import torch
import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
from glob import glob
from collections import Counter
from diffusers import StableDiffusionPipeline
from generate_human_images import POSITIVE_PROMPTS as PROMPTS, NEGATIVE_PROMPT

sys.stdout.reconfigure(encoding='utf-8')

query_params = st.query_params
tab_selection = query_params["tab"][0] if "tab" in query_params else "Image Generation"

st.set_page_config(layout="wide")
st.title("🚀 Drone Human Training Launcher")

CONFIDENCE_THRESHOLD = st.sidebar.slider("Min Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
SHOW_ONLY_PERSON = st.sidebar.checkbox("Only show images with 'person'", value=False)
ENABLE_SAFETY = st.sidebar.checkbox("Enable Safety Checker", value=False)

if st.sidebar.button("🧼 Cleanup Dataset"):
    shutil.rmtree("dataset/images/train", ignore_errors=True)
    shutil.rmtree("dataset/labels/train", ignore_errors=True)
    st.sidebar.success("Dataset cleaned.")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None if not ENABLE_SAFETY else StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").safety_checker
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe

sd_pipeline = load_pipeline()

def generate_image(prompt: str, output_path: str):
    image = sd_pipeline(prompt, num_inference_steps=25).images[0]
    image.save(output_path)

COCO_CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus"]
COLOR_MAP = {cls: (255, 0, 0) for cls in COCO_CLASS_NAMES}

def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])

def count_labels(label_dir):
    counts = Counter()
    for file in glob(f"{label_dir}/*.txt"):
        with open(file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    class_id = int(parts[0])
                    conf = 1.0
                else:
                    class_id = int(parts[0])
                    conf = float(parts[-1])
                if class_id < len(COCO_CLASS_NAMES) and conf >= CONFIDENCE_THRESHOLD:
                    counts[COCO_CLASS_NAMES[class_id]] += 1
    return counts

def draw_bounding_boxes(image_path, label_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    with open(label_path) as f:
        for line in f:
            cls_id, x, y, bw, bh, *rest = map(float, line.strip().split())
            conf = rest[0] if rest else 1.0
            if conf < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(cls_id)
            cls = COCO_CLASS_NAMES[cls_id]
            if SHOW_ONLY_PERSON and cls != "person":
                continue
            left = (x - bw / 2) * w
            top = (y - bh / 2) * h
            right = (x + bw / 2) * w
            bottom = (y + bh / 2) * h
            draw.rectangle([left, top, right, bottom], outline=COLOR_MAP[cls], width=2)
            draw.text((left, top), f"{cls} {conf:.2f}", fill=COLOR_MAP[cls])
    return image

def show_thumbnails(folder="dataset/images/train"):
    images = list_images(folder)
    label_folder = "dataset/labels/train"
    if not images:
        st.write("No images to display.")
        return
    progress_bar = st.progress(0)
    cols = st.columns(4)
    debug_output = []

    for i, img_file in enumerate(images):
        image_path = os.path.join(folder, img_file)
        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(label_path):
            debug_output.append(f"❌ Skipped (no label): {img_file}")
            continue

        image = draw_bounding_boxes(image_path, label_path)
        with cols[i % 4]:
            st.image(image, use_container_width=True)
        debug_output.append(f"✅ Displayed: {img_file}")
        progress_bar.progress((i + 1) / len(images))

    with st.expander("🧪 Thumbnail Debug Log"):
        st.code("\n".join(debug_output))

def generate_images_and_display(output_folder="dataset/images/train"):
    prompts = st.session_state.get("prompts", [])
    num_images = st.session_state.get("num_images", 10)
    if not prompts:
        st.warning("No prompts found.")
        return

    os.makedirs(output_folder, exist_ok=True)
    image_container = st.container()
    progress = st.progress(0.0)

    for i in range(num_images):
        prompt = prompts[i % len(prompts)]
        filename = os.path.join(output_folder, f"generated_{i:03d}.png")

        try:
            generate_image(prompt, filename)
        except Exception as e:
            st.error(f"Image {i+1} failed: {e}")
            continue

        with image_container:
            st.markdown(f"**Image {i+1}:** `{prompt}`")
            st.image(filename, width=256)

        progress.progress((i + 1) / num_images)

    st.success("✅ Image generation complete.")

def run_step(name, script_path, log_path):
    progress = st.empty()
    log_area = st.empty()
    progress_bar = st.progress(0)
    progress.progress(0.1)

    try:
        with st.spinner(f"{name}..."):
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True
            )

            output = (result.stdout or "") + "\n" + (result.stderr or "")
            log_path.write_text(output, encoding="utf-8")

            progress.progress(1.0)
            st.success(f"{name} ✅")
            with log_area.expander(f"📄 {name} Log"):
                st.code(result.stdout or "[No output]")
    except subprocess.CalledProcessError as e:
        output = (e.stdout or "") + "\n" + (e.stderr or "")
        log_path.write_text(output, encoding="utf-8")

        progress.progress(1.0)
        st.error(f"{name} ❌ failed")
        with log_area.expander(f"📄 {name} Error Log"):
            st.code(e.stderr or "[No error output]")

tabs = {
    "Image Generation": "🧒 Image Generation",
    "Run Control": "🧪 Run Control",
    "Training Pipeline": "📈 Training Pipeline"
}
tab_names = list(tabs.values())
selected_tab = tab_names.index(tabs.get(tab_selection, "🧒 Image Generation"))
image_tab, run_tab, train_tab = st.tabs(tab_names)

with image_tab:
    st.header("Image Generation")
    default_prompt = "\n".join(PROMPTS)
    prompts_text = st.text_area("Prompt List (one per line)", value=default_prompt)
    num_images = st.slider("Number of Images", min_value=1, max_value=100, value=10)
    if st.button("Generate Images"):
        st.session_state["prompts"] = prompts_text.strip().splitlines()
        st.session_state["num_images"] = num_images
        generate_images_and_display()
    if st.button("➡️ Continue to Run Control"):
        st.query_params = {"tab": "Run Control"}

with run_tab:
    st.header("🧪 Full Training + Inference Pipeline")

    if "run_all" not in st.session_state:
        st.session_state.run_all = True
    if "manual_override" not in st.session_state:
        st.session_state.manual_override = False

    def set_manual_override():
        st.session_state.run_all = False
        st.session_state.manual_override = True

    def enable_run_all():
        st.session_state.run_all = True
        for step_key in [
            "do_cleanup", "do_autolabel", "do_split",
            "do_yolo_train", "do_lora_train", "do_inference"]:
            st.session_state[step_key] = True

    st.checkbox("✅ Run entire pipeline (all 6 steps)", value=st.session_state.run_all, key="run_all", on_change=enable_run_all)
    st.markdown("Or select individual steps:")

    col1, col2 = st.columns(2)
    with col1:
        do_cleanup = st.checkbox("🧼 Cleanup", key="do_cleanup", value=True, disabled=st.session_state.run_all, on_change=set_manual_override)
        do_autolabel = st.checkbox("🤖 Auto-label YOLOv8", key="do_autolabel", value=True, disabled=st.session_state.run_all, on_change=set_manual_override)
        do_split = st.checkbox("🧏️‍⚖️ Split Dataset", key="do_split", value=True, disabled=st.session_state.run_all, on_change=set_manual_override)
    with col2:
        do_yolo_train = st.checkbox("🚀 Train YOLOv8", key="do_yolo_train", value=True, disabled=st.session_state.run_all, on_change=set_manual_override)
        do_lora_train = st.checkbox("🎯 Train LoRA", key="do_lora_train", value=True, disabled=st.session_state.run_all, on_change=set_manual_override)
        do_inference = st.checkbox("🔍 Run Inference", key="do_inference", value=True, disabled=st.session_state.run_all, on_change=set_manual_override)

    if st.button("▶️ Run Selected Steps"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = Path(f"logs/run_{timestamp}")
        logdir.mkdir(parents=True, exist_ok=True)
        st.info(f"📂 Logs saved to `{logdir}`")

        if st.session_state.run_all or st.session_state.do_cleanup:
            run_step("Cleanup", "cleanup_for_retrain.py", logdir / "0_cleanup.log")
        if st.session_state.run_all or st.session_state.do_autolabel:
            run_step("Auto-labeling YOLOv8", "auto_label_yolov8.py", logdir / "1_autolabel.log")
        if st.session_state.run_all or st.session_state.do_split:
            run_step("Dataset Split", "split_dataset.py", logdir / "2_split.log")
        if st.session_state.run_all or st.session_state.do_yolo_train:
            run_step("YOLOv8 Training", "train_yolov8.py", logdir / "3_train_yolov8.log")
        if st.session_state.run_all or st.session_state.do_lora_train:
            run_step("LoRA Training", "train_drone_lora.py", logdir / "4_train_lora.log")
        if st.session_state.run_all or st.session_state.do_inference:
            run_step("LoRA Inference", "inference_lora_topdown.py", logdir / "5_inference.log")

        st.success("✅ Pipeline completed.")
    if st.button("➡️ Continue to Training Pipeline"):
        st.query_params = {"tab": "Training Pipeline"}

with train_tab:
    st.header("📈 Training Pipeline")
    show_thumbnails()
    with st.expander("Legend"):
        label_counts = count_labels("dataset/labels/train")
        for cls, color in COLOR_MAP.items():
            if label_counts[cls]:
                st.markdown(f"<span style='color:rgb{color}; font-weight:bold'>{cls}</span>: {label_counts[cls]}", unsafe_allow_html=True)