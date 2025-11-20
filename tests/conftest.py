import sys
from pathlib import Path

# Project root = parent of the "tests" folder
ROOT = Path(__file__).resolve().parents[1]

# Put project root at the beginning of sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))