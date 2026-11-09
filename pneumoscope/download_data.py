import kagglehub, shutil
from pathlib import Path
src = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
dst = Path(__file__).parent / "data" / "chest_xray"
root = next(p for p in src.rglob("chest_xray") if (p / "train").exists())
shutil.copytree(root, dst, dirs_exist_ok=True); print("Data ready at", dst)
