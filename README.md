# Hybrid CNN–Qutrit VQC for Image Classification

> **IRIS 27** — International Research Conference on Computer Vision | Sub-Topic: Image Processing

## Overview

This project extends the [Q-Ternary VQC](https://github.com/North-Abyss/Q-Ternary-VQC) — a qutrit-based (d=3) variational quantum classifier — from tabular data classification to **image classification**. By coupling a pretrained CNN feature extractor with our proven qutrit quantum decision layer, we demonstrate that ternary quantum circuits can serve as ultra-compact, parameter-efficient classifiers for visual data.

### Core Innovation 

Our pipeline achieves image classification with as few as **72 trainable parameters** by:

1. **Direct Pixel Ternary Encoding:** Images are resized to a minimal grid (e.g., 4x6 = 24 pixels) and thresholded to binary intensities (0=black, 1=white).
2. **Lossless Ternary Compression:** Our custom 2³→3² algorithm maps these binarized pixels directly into ternary quantum states — **33% fewer quantum wires, zero information loss**.
3. **Qutrit VQC Classification:** A PennyLane `default.qutrit` variational circuit with SU(3) rotations and CSUM ring entanglement performs the final classification.
4. **SHAP Explainability:** Permutation-based feature attribution explains which visual features drive quantum decisions.

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐    ┌────────┐
│  Image   │ →  │ Resize (4x6) │ →  │  Binarize    │ →  │ 2³→3²      │ →  │ Qutrit │ → Class
│ (MNIST)  │    │  & Flatten   │    │ (Threshold)  │    │ Encoding   │    │ VQC    │
└──────────┘    └──────────────┘    └──────────────┘    └────────────┘    └────────┘
                  Dimension          Pixel               Ternary         Quantum
                  Reduction          Thresholding        Compression     Classification
```

## Repository Structure

```

├── LICENSE
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── conference/
│   │   └── ABSTRACT.md          # IRIS 27 submission abstract
│   └── research/
│       └── LOG.md               # Research timeline and development log
│
├── src/                         # Python source (from Q-Ternary VQC + extensions)
│   ├── main.py                  # Entry point — training pipeline
│   ├── core_engine.py           # Core orchestration engine
│   ├── generate_graphs.py       # Result visualization generator
│   ├── quantum/                 # Qutrit VQC circuit definition
│   │   ├── qutrit_model.py      #   PennyLane qutrit variational circuit
│   │   └── circuit_utils.py     #   Circuit helper utilities
│   ├── data/                    # Data loading and preprocessing
│   │   ├── loader.py            #   Dataset loader
│   │   ├── preprocessor.py      #   Binarization + 2³→3² ternary encoding
│   │   └── feature_selector.py  #   Mutual Information feature selection
│   ├── classical/               # Classical ML baselines
│   │   └── baselines.py         #   XGBoost, RF, SVM, MLP comparisons
│   ├── evaluation/              # Metrics and visualization
│   │   ├── metrics.py           #   Accuracy, F1, confusion matrix
│   │   └── visualizer.py        #   Training curves, comparison plots
│   ├── explainability/          # SHAP-based interpretability
│   │   └── xai_engine.py        #   Permutation SHAP for quantum model
│   ├── models/                  # Saved model weights and history
│   │   └── history.json
│   └── api/                     # Flask web API + dashboard
│       ├── app.py
│       ├── static/
│       └── templates/
│
├── data/                        # Dataset storage (gitignored)
├── outputs/                     # Training outputs and graphs
├── showcase_graphs/             # Presentation-ready visualizations
│
├── run.sh                       # Quick-start runner
├── run-app.sh                   # Full stack launcher
├── run-backend.sh               # Backend-only launcher
└── run-frontend.sh              # Frontend-only launcher
```

## Built Upon

This project directly extends [Q-Ternary VQC](https://github.com/North-Abyss/Q-Ternary-VQC), which demonstrated:
- **87.72% accuracy** with only **72 parameters** on the Wisconsin Breast Cancer dataset
- Competitive performance against XGBoost (94.74%), Random Forest (94.74%), and SVM (91.23%)
- Full CPU-only simulation on a Lenovo ThinkPad T580 (no GPU required)

## Tech Stack

- **Python 3.10+**
- **PennyLane** — Qutrit quantum simulation (`default.qutrit`)
- **PyTorch** — Hybrid quantum-classical training
- **torchvision** — Pretrained CNN feature extraction
- **SHAP** — Explainable AI
- **scikit-learn** — Classical baselines and preprocessing

## License

CC BY-NC-SA 4.0

## Author

**Yuvanesh KS** — B.Tech IT, AMCET (Anna University)
- GitHub: [North-Abyss](https://github.com/North-Abyss)
- Portfolio: [yuvaneshkumar.github.io](https://yuvaneshkumar.github.io)
