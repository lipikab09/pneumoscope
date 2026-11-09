<div align="center">

# 🫁 PneumoScope
**Recall-optimised pneumonia detection from chest X-rays, with Grad-CAM explainability and an interactive dashboard**

`PyTorch` · `DenseNet-121` · `Transfer Learning` · `Grad-CAM` · `Streamlit` · `Plotly`

</div>

> 📸 **Add screenshots here** after you run the dashboard: `docs/diagnose.png`, `docs/performance.png`, `docs/gradcam.png`, e.g. `![Diagnose](docs/diagnose.png)`

## Why this project
In screening, **a missed pneumonia (false negative) is far worse than a false alarm**. So instead of maximising accuracy, PneumoScope picks its operating point to hit a target **recall (≥98%)** and then makes the model's reasoning visible with **Grad-CAM**.

## Highlights
- **Transfer learning**: ImageNet DenseNet-121 (CheXNet backbone), also supports EfficientNet-B0 / ResNet-50
- **Two-phase fine-tuning**: head warm-up → full unfreeze, AdamW, cosine LR, mixed precision
- **Leakage-safe splits**: Kaggle's val set is only 16 images, so train+val is re-split **by patient** (GroupShuffleSplit)
- **Imbalance handling**: class-balanced sampling (~3:1 pneumonia:normal)
- **Recall-constrained threshold**: highest threshold on *validation* reaching ≥98% recall (max specificity), never tuned on test
- **Rigorous reporting**: bootstrap 95% CIs for recall & AUC, confusion matrix, ROC/PR
- **Explainability**: from-scratch Grad-CAM (no extra libs) + TP/FN/FP/TN gallery
- **Interactive dashboard**: live threshold slider recomputing confusion matrix/metrics, heatmap opacity/colormap controls, upload-and-diagnose

## Quickstart
```bash
pip install -r requirements.txt
# Kaggle API credentials needed (~/.kaggle/kaggle.json)
python download_data.py
python -m src.train --model densenet121 --epochs 12   # GPU recommended (or free Colab T4)
python -m src.evaluate                                 # metrics, CIs, Grad-CAM gallery
streamlit run app/dashboard.py
```
Or just `./run_all.sh`. Docker: `docker build -t pneumoscope . && docker run -p 8501:8501 pneumoscope` (mount `outputs/` with trained weights).

## Structure
```
src/ config.py data.py model.py train.py evaluate.py gradcam.py
app/ dashboard.py        # 5-tab Streamlit app
outputs/                 # best.pt, metrics.json, gallery/ (generated)
```

## Results
*Fill in after training (from `outputs/metrics.json`):*

| Metric (test, threshold from val) | Value |
|---|---|
| Recall | _ |
| Specificity | _ |
| ROC-AUC (95% CI) | _ |

## Limitations (be upfront in interviews)
Single-source paediatric dataset; the test set is small and distribution-shifted vs train; no external validation; Grad-CAM is a sanity check, not proof of causal reasoning. **Not a medical device.**

## Resume line
*Fine-tuned a DenseNet-121 on 5.8k chest X-rays with patient-level splits, optimising a recall-constrained decision threshold, and shipped Grad-CAM explainability in an interactive Streamlit dashboard.*

## Push to GitHub
```bash
git init && git add . && git commit -m "PneumoScope: pneumonia detection with Grad-CAM"
git branch -M main && git remote add origin https://github.com/<you>/pneumoscope.git && git push -u origin main
```
