# Getting Started & Developer Guide

This guide walks you through setting up the **Iris-Image** development environment, running training pipelines, evaluating classical baselines, launching the interactive web UI, and inspecting model performance.

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements
* **OS:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows (WSL2)
* **Python:** Version 3.10 or higher
* **Hardware:** Modern dual-core/quad-core CPU, $\ge 4\text{ GB RAM}$. (GPU is **not** required; the qutrit state vector is simulated natively on CPU).

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/North-Abyss/Iris-Image.git
cd Iris-Image

# Create a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Full Pipeline

### Method A: Single Command Script
Run the automated runner that initializes the environment, downloads MNIST, trains baselines, trains the Qutrit VQC, and evaluates metrics:

```bash
chmod +x run.sh
./run.sh
```

### Method B: Python Direct Execution
You can run the end-to-end pipeline directly via the Python entry point:

```bash
python3 src/main.py
```

### What Happens During Execution?
1. **Data Loading:** Automatically fetches authentic MNIST samples for classes $0$ and $1$ ($N_{\text{train}} = 320$, $N_{\text{test}} = 80$).
2. **Preprocessing:** Downsamples to $4 \times 3$ (12 pixels) and median-thresholds into 12-bit sequences.
3. **Radix Conversion:** Compresses 12 bits into 8 trits ($2^3 \to 3^2$ algorithm).
4. **Classical Baselines:** Trains and benchmarks SVM (RBF), Random Forest, XGBoost, and a classical 2-layer MLP.
5. **Quantum Optimization:** Simulates the 8-wire Qutrit VQC over 15 epochs using PennyLane's `default.qutrit` engine.
6. **Artifact Export:** Saves model checkpoints to `models/` and training curves to `showcase_graphs/`.

---

## 🌐 Interactive Web Dashboard

Iris-Image includes a real-time Flask web application allowing users to draw digits, inspect intermediate downsampled grids, view ternary compression representations, and run quantum inference live.

### Launching the Dashboard

```bash
chmod +x run-app.sh
./run-app.sh
```

Or run via Python:
```bash
python3 src/api/app.py
```

Open your browser to:
```
http://127.0.0.1:5000
```

### Dashboard Features:
* **Interactive Canvas:** Draw handwritten digits with your mouse/touchpad.
* **Live Preprocessing Visualizer:** Watch the drawing downsample to $4 \times 3$ and binarize in real time.
* **Ternary Compression Inspector:** View the exact 8-trit sequence mapped from your input.
* **Quantum Inference Engine:** Evaluates the pre-trained 72-parameter Qutrit VQC and displays prediction probabilities alongside classical model predictions.

---

## 🧪 Project Module Overview

```
src/
├── main.py              # Unified pipeline runner
├── quantum/
│   ├── qutrit_model.py  # PennyLane Qutrit VQC ansatz (72 parameters)
│   └── circuit_utils.py # CSUM entangling gates & Gell-Mann observables
├── data/
│   ├── loader.py        # MNIST binary subset data loader
│   ├── preprocessor.py  # 2³ → 3² radix compression & binarization
│   └── feature_selector.py
├── classical/
│   └── baselines.py     # SVM, Random Forest, XGBoost, MLP implementations
├── evaluation/
│   ├── metrics.py       # Accuracy, Precision, Recall, F1, ROC-AUC
│   └── visualizer.py    # Matplotlib chart generators
├── explainability/
│   └── xai_engine.py    # Kernel SHAP quantum attribution wrapper
└── api/
    ├── app.py           # Flask REST API endpoints
    ├── static/          # JavaScript canvas engine & CSS styles
    └── templates/       # HTML5 interactive interface
```

---

## 🔍 Exporting High-Resolution Paper Figures

To re-generate all publication-quality vector figures ($1200\text{ DPI}$ PNG & PDF) matching Springer Nature formatting standards:

```bash
python3 test/export_paper_figures.py
```

Generated figures will appear in `full-paper/figures/`:
* `fig1_encoding_pipeline.png` — Authentic MNIST 4-stage pipeline
* `fig2_wire_loss.png` — Quantum wire allocation & BCE loss convergence
* `fig3_accuracy_efficiency.png` — Benchmark comparison & parameter efficiency frontier
* `fig4_confusion_roc.png` — Confusion matrix & multi-model ROC curves
* `fig5_shap_importance.png` — 2-panel Quantum SHAP spatial grid & feature ranking
