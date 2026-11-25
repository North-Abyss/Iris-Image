# Documentation Hub: Direct Pixel Ternary Encoding & Qutrit VQC

Welcome to the technical documentation for **Iris-Image**, an open-source Quantum Machine Learning (QML) framework implementing **Direct Pixel Ternary Encoding** and a **Qutrit Variational Quantum Classifier (VQC)** for parameter-efficient image recognition.

---

## 📚 Table of Contents

| Guide | Description |
| :--- | :--- |
| **[Architecture & Quantum Theory](ARCHITECTURE.md)** | Mathematical formulation of lossless $2^3 \to 3^2$ radix conversion, SU(3) subgroup rotations, CSUM ring entanglement, and Gell-Mann observables. |
| **[Getting Started & Developer Guide](GETTING_STARTED.md)** | Environment setup, training pipeline execution, Flask web dashboard launch, and interactive inference instructions. |
| **[Explainability & Benchmarks](EXPLAINABILITY_AND_BENCHMARKS.md)** | Empirical comparisons vs. classical ML models (SVM, RF, XGBoost, MLP), parameter efficiency frontier, and Quantum SHAP stroke attribution analysis. |

---

## 🔬 Core Innovations at a Glance

1. **Direct Pixel Preprocessing (No Classical CNN Backbone):**
   - High-resolution visual inputs ($28 \times 28$) are downsampled to a compact structural grid ($4 \times 3 = 12$ pixels) and binarized via adaptive median thresholding.
   - Eliminates resource-intensive classical convolutional neural network pre-training.

2. **Lossless $2^3 \to 3^2$ Radix Conversion:**
   - Compresses 12 binary bits into an 8-trit sequence ($d=3$).
   - Achieves an exact **33.33% reduction in quantum data carriers** (from 12 wires to 8 wires) with zero information loss.

3. **Compact Qutrit VQC (72 Trainable Weights):**
   - Built on PennyLane's `default.qutrit` simulation engine.
   - Employs 3 variational layers with 3-parameter SU(3) subspace rotations ($R_x, R_z$ across $\{|0\rangle, |1\rangle\}$ and $\{|1\rangle, |2\rangle\}$) and circular Controlled-SUM (CSUM) ring entanglement.
   - Achieves **98.75% test accuracy** and **98.82% F1 score** on MNIST digit classification while reducing parameter footprints by over **97%**.

4. **Transparent Quantum Explainability (Quantum SHAP):**
   - Integrated KernelExplainer wrapper attributing quantum measurement shifts directly back to the spatial pixel grid, proving the model attends to authentic digit stroke contours.

---

## 🧬 Lineage: Evolution from Q-Ternary VQC

This project builds directly upon the foundational discoveries of [Q-Ternary VQC](https://github.com/North-Abyss/Q-Ternary-VQC):

* **Foundational Work (Q-Ternary VQC):** Pioneered radix-conversion ternary compression on tabular clinical datasets (Wisconsin Breast Cancer benchmark), demonstrating 87.72% accuracy with 72 parameters.
* **This Project (Iris-Image):** Scales the radix-conversion paradigm from 1D tabular records into 2D spatial visual data, establishing the first direct pixel-to-qutrit image classification pipeline without classical convolutional backbones.

---

## 📜 Citation

If you use this software or algorithm in your research, please cite:

```bibtex
@software{yuvanesh2026irisimage,
  title={Iris-Image: Direct Pixel Ternary Encoding for Image Classification Using a Qutrit Variational Quantum Classifier},
  author={Yuvanesh, KS},
  url={https://github.com/North-Abyss/Iris-Image},
  year={2026}
}
```
