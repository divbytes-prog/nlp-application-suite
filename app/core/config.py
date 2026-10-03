import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_TEXT = 5000
MAX_UPLOAD = 1_000_000
MAX_REQUEST = 1_100_000
MAX_ROWS = 100
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
if not 0 <= THRESHOLD <= 1:
    raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
ENABLE_TRANSFORMERS = os.getenv("ENABLE_TRANSFORMERS", "0") == "1"
