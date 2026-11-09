
<div align="center">
# 🫁 **PneumoScope**

> **Recall-Optimised Pneumonia Detection from Chest X-Rays with Grad-CAM Explainability**  
> *A research demo and portfolio project exploring deep learning interpretability and clinical decision thresholding.*

---

## 📌 Overview

PneumoScope is an end-to-end computer vision and explainable AI (XAI) dashboard built to classify chest X-ray images into **PNEUMONIA** or **NORMAL**. 

Rather than relying on default classification thresholds (0.50), PneumoScope uses a **recall-constrained operating point** (≥98% sensitivity) on validation data to minimize costly False Negatives (missed diagnoses). Every inference is paired with interactive **Grad-CAM visual heatmaps** to highlight the exact regions driving the neural network's decisions.

---

## ✨ Features

- **🔬 Real-Time Diagnostic Dashboard**: Upload custom chest X-rays, adjust decision thresholds dynamically, and visualize Grad-CAM activation heatmaps.
- **📊 Interactive Performance Analytics**: Live confusion matrix, ROC curves, score distributions, and recall vs. specificity trade-off sliders powered by Plotly.
- **🖼️ Explainability Gallery**: Pre-calculated Grad-CAM visual evaluations organized by confusion matrix outcomes (TP, FN, FP, TN).
- **📈 Training Tracking**: Full visualization of training loss, validation AUC, and recall trajectories over epochs.
- **⚡ PyTorch & DenseNet-121 Architecture**: Transfer learning pipeline optimized with Automatic Mixed Precision (AMP) and cosine annealing learning rate schedules.

---
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


## 🛠️ Project Structure

```text
pneumoscope/
├── app/
│   └── dashboard.py          # Interactive Streamlit Web Application
├── src/
│   ├── config.py             # Hyperparameters, path configurations, and constants
│   ├── data.py               # Dataset processing, augmentation, and data loaders
│   ├── model.py              # PyTorch DenseNet-121 model loading and architecture
│   ├── train.py              # Transfer learning training pipeline with AMP
│   ├── evaluate.py           # Evaluation script (metrics, ROC, bootstrap CIs)
│   └── gradcam.py            # Grad-CAM heatmap extraction and overlay engine
├── outputs/
│   ├── best.pt               # Trained PyTorch model weights
│   ├── metrics.json          # Test set evaluation summary & metrics
│   ├── history.json          # Training history logs
│   └── test_predictions.csv  # Model outputs on test dataset
├── .gitignore                # Excludes virtual environments and datasets
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
# 🫁 PneumoScope
**Recall-optimised pneumonia detection from chest X-rays, with Grad-CAM explainability and an interactive dashboard**

`PyTorch` · `DenseNet-121` · `Transfer Learning` · `Grad-CAM` · `Streamlit` · `Plotly`


Fine-tuned a DenseNet-121 on 5.8k chest X-rays with patient-level splits, optimising a recall-constrained decision threshold, and shipped Grad-CAM explainability in an interactive Streamlit dashboard.
</div>




