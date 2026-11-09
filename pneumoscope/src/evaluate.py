import json, numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from .config import *; from .data import splits, loader, tfm; from .model import load; from .gradcam import GradCAM, overlay

def predict(model, dl, dev):
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl: P += torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu().tolist(); Y += y.tolist()
    return np.array(P), np.array(Y)

def stats(P, Y, t):
    pr = P >= t; tp = (pr & (Y == 1)).sum(); fn = (~pr & (Y == 1)).sum(); fp = (pr & (Y == 0)).sum(); tn = (~pr & (Y == 0)).sum()
    rec, spec, prec = tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1)
    return dict(threshold=float(t), recall=float(rec), specificity=float(spec), precision=float(prec),
                f1=float(2 * prec * rec / max(prec + rec, 1e-9)), accuracy=float((tp + tn) / len(Y)),
                tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))

def pick_threshold(P, Y, target=TARGET_RECALL):
    """Highest threshold on VALIDATION data that still reaches the target recall (=max specificity)."""
    ok = [t for t in np.linspace(0.01, 0.99, 197) if stats(P, Y, t)["recall"] >= target]
    return max(ok) if ok else 0.01

def boot(P, Y, t, n=1000):
    r, a, rng = [], [], np.random.default_rng(SEED)
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y))
        if Y[i].min() == Y[i].max(): continue
        r.append(stats(P[i], Y[i], t)["recall"]); a.append(roc_auc_score(Y[i], P[i]))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def gallery(model, layer, items, P, Y, t, dev, k=4):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cam = GradCAM(model, layer); pred = P >= t
    groups = {"TP": pred & (Y == 1), "FN": ~pred & (Y == 1), "FP": pred & (Y == 0), "TN": ~pred & (Y == 0)}
    for g, m in groups.items():
        idx = np.where(m)[0]; idx = idx[np.argsort(-P[idx])][:k]   # most confident examples per group
        for j, i in enumerate(idx):
            img = Image.open(items[i][0]).convert("RGB").resize((IMG, IMG)); x = tfm(False)(img)[None].to(dev)
            p, c = cam(x); ov = overlay(np.array(img), c[0])
            fig, ax = plt.subplots(1, 2, figsize=(6, 3.1), facecolor="#0b1020")
            for a_, im, ti in zip(ax, [np.array(img), ov], ["X-ray", "Grad-CAM"]): a_.imshow(im); a_.axis("off"); a_.set_title(ti, color="w", fontsize=9)
            fig.suptitle(f"{g}  |  P(pneumonia)={p[0]:.2f}", color="w", fontsize=10)
            fig.savefig(OUT / "gallery" / f"{g}_{j}.png", dpi=130, bbox_inches="tight", facecolor="#0b1020"); plt.close(fig)

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model, layer, ck = load(device=dev)
    tr, va, te = splits(); Pv, Yv = predict(model, loader(va), dev); Pt, Yt = predict(model, loader(te), dev)
    t = pick_threshold(Pv, Yv); rci, aci = boot(Pt, Yt, t)
    fpr, tpr, _ = roc_curve(Yt, Pt); pp, rr, _ = precision_recall_curve(Yt, Pt)
    m = dict(model=ck["name"], threshold=float(t), target_recall=TARGET_RECALL, test=stats(Pt, Yt, t), at_0_5=stats(Pt, Yt, .5),
             auc=float(roc_auc_score(Yt, Pt)), auc_ci=aci, recall_ci=rci, ap=float(average_precision_score(Yt, Pt)),
             n_test=len(Yt), roc=dict(fpr=fpr.tolist(), tpr=tpr.tolist()), pr=dict(p=pp.tolist(), r=rr.tolist()))
    json.dump(m, open(OUT / "metrics.json", "w")); pd.DataFrame(dict(path=[str(p) for p, _ in te], y=Yt, prob=Pt)).to_csv(OUT / "test_predictions.csv", index=False)
    gallery(model, layer, te, Pt, Yt, t, dev); print(json.dumps(m["test"], indent=1), "AUC", m["auc"])
