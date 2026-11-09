import argparse, json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .config import *; from .data import splits, loader; from .model import build, set_backbone_grad

def run(a):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, _ = splits(); dl_tr, dl_va = loader(tr, True, a.bs), loader(va, False, a.bs)
    model, head, _ = build(a.model); model.to(dev)
    crit, scaler = nn.BCEWithLogitsLoss(), torch.amp.GradScaler(enabled=dev == "cuda")
    hist, best = [], 0
    for ep in range(a.epochs):
        warm = ep < a.warmup                       # phase 1: head only, phase 2: full fine-tune
        set_backbone_grad(model, head, not warm)
        if ep == 0 or ep == a.warmup:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr * (10 if warm else 1), weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs - ep))
        model.train(); tl = 0
        for x, y in tqdm(dl_tr, desc=f"ep{ep+1}/{a.epochs}{' [warmup]' if warm else ''}", leave=False):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x).squeeze(1), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item()
        sched.step(); model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in dl_va: P += torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu().tolist(); Y += y.tolist()
        P, Y = np.array(P), np.array(Y); auc = roc_auc_score(Y, P)
        rec = ((P >= .5) & (Y == 1)).sum() / (Y == 1).sum()
        hist.append(dict(epoch=ep + 1, loss=tl / len(dl_tr), val_auc=auc, val_recall_at_0_5=float(rec)))
        print(hist[-1])
        if auc > best:
            best = auc; torch.save(dict(name=a.model, state=model.state_dict(), auc=auc), OUT / "best.pt")
        json.dump(hist, open(OUT / "history.json", "w"), indent=1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="densenet121", choices=["densenet121", "efficientnet_b0", "resnet50"])
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4)
    run(p.parse_args())
