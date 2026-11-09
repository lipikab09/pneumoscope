import sys
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# 1. Resolve project root path so 'src' imports cleanly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import OUT, IMG
from src.data import tfm
from src.model import load
from src.gradcam import GradCAM, overlay

# 2. Streamlit Page Setup & Custom Dark Styling
st.set_page_config(page_title="PneumoScope", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #16264d 0, #0b1020 60%);
}
.hero {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255, 255, 255, .04);
    border: 1px solid rgba(255, 255, 255, .09);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}
.card b {
    font-size: 2rem;
    color: #22d3ee;
    display: block;
}
.card span {
    display: block;
    opacity: .7;
    font-size: .85rem;
    margin-top: 4px;
}
.pos {
    border-color: #f87171 !important;
}
.pos b {
    color: #f87171 !important;
}
.neg {
    border-color: #34d399 !important;
}
.neg b {
    color: #34d399 !important;
}
</style>
""", unsafe_allow_html=True)

# 3. Header section
st.markdown('<div class="hero">PneumoScope</div>', unsafe_allow_html=True)
st.caption("Recall-optimised pneumonia detection from chest X-rays · transfer learning + Grad-CAM explainability · research demo, not a medical device")

# 4. Helper Functions & Model Caching
@st.cache_resource
def get_model():
    m, layer, ck = load()
    m.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(device)
    return m, GradCAM(m, layer), ck

def card(v, l, cls=""):
    return f'<div class="card {cls}"><b>{v}</b><span>{l}</span></div>'

def plot(fig, **k):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        **k
    )
    return fig

# Load saved test evaluation metrics
metrics_path = OUT / "metrics.json"
M = json.load(open(metrics_path)) if metrics_path.exists() else None

# 5. Dashboard Tabs
t1, t2, t3, t4, t5 = st.tabs(["🔬 Diagnose", "📊 Performance", "🖼️ Grad-CAM Gallery", "📈 Training", "ℹ️ About"])

# --- TAB 1: DIAGNOSE ---
with t1:
    best_pt_path = OUT / "best.pt"
    if not best_pt_path.exists():
        st.warning("Train first: `python -m src.train` (see README).")
    else:
        model, cam, ck = get_model()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            up = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
            default_thr = float(M["threshold"]) if M else 0.50
            thr = st.slider("Decision threshold", 0.01, 0.99, default_thr, 0.01, help="Lower = higher recall, more false alarms")
            alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.50, 0.05)
            cmap = st.selectbox("Colormap", ["jet", "inferno", "turbo", "magma"])
            
        if up:
            img = Image.open(up).convert("RGB").resize((IMG, IMG))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensor = tfm(False)(img)[None].to(device)
            
            # Execute Grad-CAM (gradients MUST be enabled)
            prob_out, cam_map = cam(tensor)
                
            # Handle probability calculation
            if isinstance(prob_out, torch.Tensor):
                raw_val = float(prob_out.detach().cpu().numpy()[0])
            else:
                raw_val = float(prob_out[0])
                
            p = float(1.0 / (1.0 + np.exp(-raw_val))) if (raw_val < 0.0 or raw_val > 1.0) else raw_val
            pos = p >= thr
            
            with c2:
                a, b = st.columns(2)
                a.markdown(card("PNEUMONIA" if pos else "NORMAL", "model decision", "pos" if pos else "neg"), unsafe_allow_html=True)
                b.markdown(card(f"{p:.1%}", "P(pneumonia)"), unsafe_allow_html=True)
                
                g = go.Figure(go.Indicator(
                    mode="gauge",
                    value=p * 100,
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="#f87171" if pos else "#34d399"),
                        threshold=dict(line=dict(color="white", width=4), value=thr * 100)
                    )
                ))
                st.plotly_chart(plot(g, height=170), use_container_width=True)
                
            i1, i2 = st.columns(2)
            i1.image(np.array(img), caption="Input", use_container_width=True)
            overlay_arr = overlay(np.array(img), cam_map[0], alpha, cmap)
            i2.image(overlay_arr, caption="Grad-CAM: regions driving the prediction", use_container_width=True)
            st.info("Warm regions pushed the model toward 'pneumonia'. Heatmaps on bones, text markers or borders signal shortcut learning.")
        else:
            c2.info("⬅️ Upload an X-ray (e.g. from data/chest_xray/test) to get a prediction with explanation.")

# --- TAB 2: PERFORMANCE ---
with t2:
    preds_csv_path = OUT / "test_predictions.csv"
    if not M or not preds_csv_path.exists():
        st.warning("Run `python -m src.evaluate` first.")
    else:
        df = pd.read_csv(preds_csv_path)
        thr2 = st.slider("Threshold (recomputes everything live)", 0.01, 0.99, float(M["threshold"]), 0.01, key="t2")
        
        pr = df.prob >= thr2
        tp = int((pr & (df.y == 1)).sum())
        fn = int((~pr & (df.y == 1)).sum())
        fp = int((pr & (df.y == 0)).sum())
        tn = int((~pr & (df.y == 0)).sum())
        
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        
        cols = st.columns(5)
        metrics_display = [
            (f"{rec:.1%}", "Recall (sensitivity)", "pos"),
            (f"{spec:.1%}", "Specificity", ""),
            (f"{prec:.1%}", "Precision", ""),
            (f"{M['auc']:.3f}", f"ROC-AUC · 95% CI {M['auc_ci'][0]:.3f}-{M['auc_ci'][1]:.3f}", ""),
            (f"{fn}", "Missed patients (FN)", "pos")
        ]
        
        for col, (v, l, cl) in zip(cols, metrics_display):
            col.markdown(card(v, l, cl), unsafe_allow_html=True)
            
        a, b, c = st.columns(3)
        
        # Confusion Matrix
        cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Pred NORMAL", "Pred PNEUMONIA"],
            y=["True NORMAL", "True PNEUMONIA"],
            text=[[tn, fp], [fn, tp]],
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False
        ))
        a.plotly_chart(plot(cm, title="Confusion matrix", yaxis_autorange="reversed"), use_container_width=True)
        
        # ROC Curve
        roc = go.Figure(go.Scatter(x=M["roc"]["fpr"], y=M["roc"]["tpr"], fill="tozeroy", line_color="#22d3ee", name="ROC"))
        roc.add_trace(go.Scatter(x=[1 - spec], y=[rec], mode="markers", marker=dict(size=14, color="#f472b6"), name="operating point"))
        roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
        b.plotly_chart(plot(roc, title="ROC curve", xaxis_title="FPR", yaxis_title="TPR"), use_container_width=True)
        
        # Score Distribution
        h = go.Figure([
            go.Histogram(x=df.prob[df.y == 0], name="NORMAL", marker_color="#34d399", opacity=.7),
            go.Histogram(x=df.prob[df.y == 1], name="PNEUMONIA", marker_color="#f87171", opacity=.7)
        ])
        h.add_vline(x=thr2, line_color="white", line_dash="dash")
        c.plotly_chart(plot(h, title="Score distribution", barmode="overlay"), use_container_width=True)
        
        # Recall vs Specificity Trade-off
        ts = np.linspace(.01, .99, 99)
        sw = pd.DataFrame([
            dict(
                t=t,
                Recall=((df.prob >= t) & (df.y == 1)).sum() / max((df.y == 1).sum(), 1),
                Specificity=((df.prob < t) & (df.y == 0)).sum() / max((df.y == 0).sum(), 1)
            ) for t in ts
        ])
        st.plotly_chart(plot(go.Figure([go.Scatter(x=sw.t, y=sw[c_], name=c_) for c_ in ["Recall", "Specificity"]]), title="Recall vs Specificity trade-off", xaxis_title="threshold"), use_container_width=True)

# --- TAB 3: GRAD-CAM GALLERY ---
with t3:
    gallery_dir = OUT / "gallery"
    imgs = sorted(list(gallery_dir.glob("*.png"))) if gallery_dir.exists() else []
    if not imgs:
        st.warning("Run `python -m src.evaluate` to generate the gallery.")
    else:
        sel = st.multiselect("Filter", ["TP", "FN", "FP", "TN"], ["TP", "FN"])
        st.caption("TP=caught pneumonia · FN=missed (the costly error) · FP=false alarm · TN=correct normal")
        cols = st.columns(2)
        filtered_imgs = [p for p in imgs if p.stem.split("_")[0] in sel]
        for i, p in enumerate(filtered_imgs):
            cols[i % 2].image(str(p), use_container_width=True)

# --- TAB 4: TRAINING ---
with t4:
    history_path = OUT / "history.json"
    if history_path.exists():
        H = pd.DataFrame(json.load(open(history_path)))
        available_cols = [c_ for c_ in ["loss", "val_auc", "val_recall_at_0_5"] if c_ in H.columns]
        f = go.Figure([go.Scatter(x=H.index, y=H[c_], name=c_, mode="lines+markers") for c_ in available_cols])
        st.plotly_chart(plot(f, title="Training curves"), use_container_width=True)
    else:
        st.info("No training history yet.")

# --- TAB 5: ABOUT ---
with t5:
    st.markdown("""**Pipeline**: Kaggle Chest X-Ray (5,863 images) → patient-level re-split → class-balanced sampling → DenseNet-121 (ImageNet) two-phase fine-tuning (head warm-up → full) with AMP + cosine LR →
**recall-constrained threshold** chosen on validation (≥98% sensitivity, max specificity) → bootstrap 95% CIs → Grad-CAM explanations.

**Why recall?** A missed pneumonia (FN) can be fatal; a false alarm costs a follow-up. So the threshold is chosen for sensitivity, not 0.5.

**Caveats**: single-source paediatric dataset, small test set, no external validation. Research/portfolio demo only.""")