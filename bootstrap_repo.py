#!/usr/bin/env python3
from pathlib import Path

# Utility to create files with content
def touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

# Base path: directory containing this script
root = Path(__file__).parent

# Directories to create
dirs = [
    "data/raw",
    "data/processed",
    "notebooks",
    "scripts",
    "src/ml_practice",
]
for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

# Notebook placeholders
touch(root / "notebooks/01_iris.ipynb")
touch(root / "notebooks/02_california_housing.ipynb")

# Script placeholders
touch(root / "scripts/train_iris.py", '''"""
Train a RandomForest on the Iris dataset.
"""
from sklearn.datasets import load_iris
# add your training code here
''')
touch(root / "scripts/train_california_housing.py", '''"""
Train a model on the California Housing dataset.
"""
from sklearn.datasets import fetch_california_housing
# add your training code here
''')
touch(root / "scripts/utils.py", '''"""
Helper functions for data loading and preprocessing.
"""
# add utility functions here
''')

# Package skeleton
touch(root / "src/ml_practice/__init__.py", '''"""
ml_practice package initialization.
"""
''')
touch(root / "src/ml_practice/data_loader.py", '''"""
Functions to load various datasets.
"""
''')
touch(root / "src/ml_practice/models.py", '''"""
Model definitions and training logic.
"""
''')
touch(root / "src/ml_practice/evaluation.py", '''"""
Evaluation utilities: metrics and plotting.
"""
''')

# Top-level files
touch(root / "requirements.txt")
touch(root / ".gitignore", '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]

# Virtual environment
.venv/

# Jupyter checkpoints
.ipynb_checkpoints/

# Data folders
data/raw/
''')
touch(root / "README.md", '''# ML Practice

Project initialized with Python script. Update this README with details.
''')

print("Python bootstrap complete: directories and files created.")