import os
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

# === CONFIGURATION ===
PREDICTIONS_DIR = Path("predictions")
VIS_DIR = Path("visualizations")
HTML_PATH = VIS_DIR / "index.html"
ZIP_PATH = VIS_DIR / "predictions_export.zip"
CONFIDENCE_THRESHOLD = 0.3

# === SETUP ===
VIS_DIR.mkdir(exist_ok=True)
image_data = []

# === LOAD PREDICTIONS ===
for pred_file in PREDICTIONS_DIR.glob("*.json"):
    with open(pred_file) as f:
        preds = json.load(f)
    image_path = preds.get("image_path")
    detections = preds.get("predictions", [])
    for det in detections:
        confidence = det.get("confidence")
        if confidence is None or confidence < CONFIDENCE_THRESHOLD:
            continue
        image_data.append({
            "file": Path(image_path).name,
            "confidence": confidence,
            "class": det.get("class", "unknown"),
            "box": det.get("box", []),
            "source": "Generated" if "drone_view" in image_path else "Real",
        })

# === CHECK ===
if not image_data:
    print("⚠️ No valid detections found with confidence scores.")
    exit(1)

df = pd.DataFrame(image_data)
df.sort_values(by="confidence", ascending=False, inplace=True)

# === SAVE SUMMARY PLOT ===
summary_plot = VIS_DIR / "confidence_summary.png"
plt.figure(figsize=(10, 4))
plt.hist(df["confidence"], bins=20, color="skyblue", edgecolor="black")
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(summary_plot)
plt.close()

# === GENERATE HTML VIEWER ===
html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Prediction Visualizer</title>
    <style>
        body { font-family: sans-serif; background: #f0f0f0; margin: 0; padding: 2em; }
        img { max-width: 640px; border: 4px solid #222; margin: 1em 0; }
        .info { margin-bottom: 2em; }
        .highlight { border-color: red; }
        .nav { position: fixed; top: 10px; right: 10px; background: white; padding: 1em; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="nav">
        <button onclick="prev()">⬅ Prev</button>
        <button onclick="next()">Next ➡</button>
        <label><input type="checkbox" id="hideLowConf" onchange="toggleLowConf()"> Hide &lt; {{ threshold }}</label>
    </div>

    <h1>Prediction Visualizer</h1>
    <img src="confidence_summary.png" alt="Confidence Histogram" />

    {% for row in rows %}
        <div class="info" data-conf="{{ row.confidence }}">
            <strong>{{ row.file }}</strong><br>
            Class: {{ row.class }} | Confidence: {{ "%.2f"|format(row.confidence) }} | Source: {{ row.source }}<br>
            <img src="../dataset/images/train/{{ row.file }}" />
        </div>
    {% endfor %}

<script>
let cur = 0;
const all = document.querySelectorAll('.info');
function show(i) {
    all.forEach((el, idx) => el.style.display = (idx === i ? 'block' : 'none'));
}
function next() { cur = (cur + 1) % all.length; show(cur); }
function prev() { cur = (cur - 1 + all.length) % all.length; show(cur); }
function toggleLowConf() {
    const threshold = {{ threshold }};
    const hide = document.getElementById("hideLowConf").checked;
    all.forEach(el => {
        const conf = parseFloat(el.dataset.conf);
        el.style.display = (hide && conf < threshold) ? 'none' : 'block';
    });
}
document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") next();
    if (e.key === "ArrowLeft") prev();
});
show(cur);
</script>
</body>
</html>
""")

with open(HTML_PATH, "w") as f:
    f.write(html_template.render(rows=df.to_dict("records"), threshold=CONFIDENCE_THRESHOLD))

# === EXPORT AS ZIP ===
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in VIS_DIR.glob("*.*"):
        zipf.write(file, arcname=file.name)

# === OPTIONAL: OPEN IN BROWSER ===
try:
    import webbrowser
    webbrowser.open(f"file://{HTML_PATH.resolve()}")
except:
    pass

print(f"✅ Visualization complete: {HTML_PATH}")
print(f"📦 Exported to: {ZIP_PATH}")
