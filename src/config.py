from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "ravdess"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3.0
SAMPLES = int(SAMPLE_RATE * SAMPLE_RATE)