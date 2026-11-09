import numpy as np, torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from .config import *

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXT = {".jpeg", ".jpg", ".png"}

def list_split(d):
    return [(p, i) for i, c in enumerate(CLASSES) for p in sorted((d / c).glob("*")) if p.suffix.lower() in EXT]

def patient_id(p):  # avoid patient leakage: several X-rays per patient exist
    s = p.stem
    return s.split("_")[0] if s.startswith("person") else "-".join(s.split("-")[:-1])

def splits(val_frac=0.15):
    """Official val set has only 16 images -> re-split train+val by PATIENT."""
    pool = list_split(DATA / "train") + list_split(DATA / "val")
    groups = [patient_id(p) for p, _ in pool]
    tr, va = next(GroupShuffleSplit(1, test_size=val_frac, random_state=SEED).split(pool, groups=groups))
    return [pool[i] for i in tr], [pool[i] for i in va], list_split(DATA / "test")

def tfm(train):
    if train:
        return T.Compose([T.Resize(256), T.RandomResizedCrop(IMG, scale=(0.8, 1)), T.RandomRotation(10),
                          T.ColorJitter(0.25, 0.25), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

class XR(Dataset):
    def __init__(s, items, train=False): s.items, s.t = items, tfm(train)
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i]
        return s.t(Image.open(p).convert("RGB")), torch.tensor(y, dtype=torch.float32)

def loader(items, train=False, bs=32, workers=4):
    if train:
        y = np.array([l for _, l in items]); w = 1.0 / np.bincount(y)[y]
        smp = WeightedRandomSampler(w, len(y), replacement=True)  # class-balanced batches
        return DataLoader(XR(items, True), bs, sampler=smp, num_workers=workers, pin_memory=True)
    return DataLoader(XR(items), bs, shuffle=False, num_workers=workers, pin_memory=True)
