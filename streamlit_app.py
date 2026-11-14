import os
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

with open(web_app_file, "r", encoding="utf-8") as f:
    code = f.read()

# Execute in the context of web_app.py
exec(compile(code, str(web_app_file), "exec"), {
    "__file__": str(web_app_file),
    "__name__": "__main__"
})
