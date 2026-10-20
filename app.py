"""
app.py
~~~~~~
Streamlit entry point for NEXUS Intelligence.
Loads and executes src.nexus.ui.streamlit_app while preserving backwards compatibility.
"""

import sys
from pathlib import Path

# Ensure src/ is on Python path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Execute the modern modular Streamlit interface
ui_entry = src_dir / "nexus" / "ui" / "streamlit_app.py"
with open(ui_entry, encoding="utf-8") as f:
    code = compile(f.read(), str(ui_entry), "exec")
    exec(code, globals())