import os
import runpy
import sys
from pathlib import Path

# Set up paths
ROOT_DIR = Path(__file__).resolve().parent
TCONTEXT_DIR = ROOT_DIR / "tcontext"

if str(TCONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(TCONTEXT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the modern multi-model research dashboard
web_app_file = TCONTEXT_DIR / "web_app.py"
runpy.run_path(str(web_app_file), run_name="__main__")
