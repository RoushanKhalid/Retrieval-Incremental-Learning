"""
Utility script to set up the src/tcontext package structure.
"""
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_TCONTEXT = REPO_ROOT / "src" / "tcontext"
SRC_TCONTEXT.mkdir(parents=True, exist_ok=True)

# Copy python packages
subdirs = ["models", "retrieval", "evaluation", "visualization", "experiments"]
for s in subdirs:
    dest = SRC_TCONTEXT / s
    src = REPO_ROOT / "tcontext" / s
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"Copied {src} -> {dest}")

# Data package: copy dataset_loader.py and create __init__.py
(SRC_TCONTEXT / "data").mkdir(parents=True, exist_ok=True)
shutil.copy2(REPO_ROOT / "tcontext" / "data" / "dataset_loader.py", SRC_TCONTEXT / "data" / "dataset_loader.py")
with open(SRC_TCONTEXT / "data" / "__init__.py", "w", encoding="utf-8") as f:
    f.write("from .dataset_loader import CatsDogsDataset, get_dataloaders, get_default_transforms, prepare_stratified_split\n")

# Ensure tcontext/data/__init__.py also exists
with open(REPO_ROOT / "tcontext" / "data" / "__init__.py", "w", encoding="utf-8") as f:
    f.write("from .dataset_loader import CatsDogsDataset, get_dataloaders, get_default_transforms, prepare_stratified_split\n")

# Copy root python files
py_files = ["comparative_eda.py", "dataset_utils.py", "query_demo.py", "quick500_experiment.py", "web_app.py"]
for p in py_files:
    shutil.copy2(REPO_ROOT / "tcontext" / p, SRC_TCONTEXT / p)
    print(f"Copied {p} -> src/tcontext")

# Create __init__.py files
init_content = '''"""
tcontext: Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning.
"""

__version__ = "0.1.0"
__all__ = ["models", "retrieval", "evaluation", "data", "visualization"]
'''

with open(SRC_TCONTEXT / "__init__.py", "w", encoding="utf-8") as f:
    f.write(init_content)

with open(REPO_ROOT / "tcontext" / "__init__.py", "w", encoding="utf-8") as f:
    f.write(init_content)

print("Package structure prepared successfully.")

