from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chest_xray"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True); (OUT / "gallery").mkdir(exist_ok=True)
SEED, IMG = 42, 224
CLASSES = ["NORMAL", "PNEUMONIA"]
TARGET_RECALL = 0.98   # clinical operating point: catch >=98% of sick patients
