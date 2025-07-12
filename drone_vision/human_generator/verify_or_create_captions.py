import os
from pathlib import Path

image_dir = Path("dataset/images/train")
placeholder_prompt = "Top-down drone photo of people walking outdoors"
missing, empty = [], []

for img_file in image_dir.glob("*.jpg"):
    txt_file = img_file.with_suffix(".txt")
    if not txt_file.exists():
        txt_file.write_text(placeholder_prompt)
        missing.append(txt_file.name)
    elif txt_file.read_text().strip() == "":
        txt_file.write_text(placeholder_prompt)
        empty.append(txt_file.name)

print(f"✅ Scan complete in {image_dir}")
print(f"📝 Created {len(missing)} missing captions.")
print(f"✏️ Filled {len(empty)} empty captions.")
