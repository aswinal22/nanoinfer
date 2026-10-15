"""Make `src/` importable without an editable install.

Keeps the repo runnable with a bare `python experiments/m1_roofline.py`.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))