# Architecture & Quantum Theory

This document provides the theoretical and mathematical foundations of the **Direct Pixel Ternary Encoding** and **Qutrit Variational Quantum Classifier (VQC)** framework.

---

## 1. High-Level Pipeline Architecture

The end-to-end framework maps raw grayscale imagery directly into multi-level quantum states and performs classification through four sequential stages:

```
┌─────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Raw Input Image │ ───> │ Spatial Downsampling    │ ───> │ Adaptive Binarization   │
│  (28 × 28 px)   │      │ (4 × 3 Grid, 12 Pixels) │      │ (12-bit Vector X_bin)   │
└─────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                       │
                                                                       ▼
┌─────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Classification  │ <─── │ Qutrit Variational      │ <─── │ Lossless 2³ → 3²        │
│ ŷ ∈ (0, 1)      │      │ Quantum Classifier (VQC)│      │ Radix Compression       │
└─────────────────┘      │ (8 Wires, 72 Parameters)│      │ (8-Trit Register |ψ⟩)   │
                         └─────────────────────────┘      └─────────────────────────┘
```

---

## 2. Direct Pixel Preprocessing & Adaptive Binarization

To assess the raw encoding capability of the quantum circuit without classical inductive bias from deep networks:

1. **Bilinear Downsampling:**
   A raw input image $X_{\text{raw}} \in \mathbb{R}^{28 \times 28}$ is pooled to a compact structural grid:
   $$X_{\text{grid}} \in \mathbb{R}^{4 \times 3}$$

2. **Adaptive Median Binarization:**
   Let $\tau = \text{median}(X_{\text{grid}})$. Each pixel is binarized according to:
   $$b_i = \begin{cases} 1 & \text{if } X_{\text{grid}, i} \ge \tau \\ 0 & \text{otherwise} \end{cases} \quad \text{for } i \in \{0, 1, \dots, 11\}$$
   This yields a 12-bit binary feature vector:
   $$X_{\text{bin}} = [b_0, b_1, \dots, b_{11}] \in \{0, 1\}^{12}$$

---

## 3. Lossless $2^3 \to 3^2$ Radix Conversion Algorithm

### Mathematical Proof of Lossless Mapping
Conventional quantum image processing requires one qubit per binary pixel ($N_{\text{qubits}} = 12$ wires), producing an exponential Hilbert space dimensionality of $2^{12} = 4,096$.

By transitioning to three-level quantum systems (qutrits, $d = 3$), the state capacity scales as $3^{N_{\text{qutrits}}}$. We leverage the fundamental radix inequality:
$$2^3 = 8 < 3^2 = 9$$

Because the numerical capacity of 2 ternary trits ($3^2 = 9$) strictly exceeds that of 3 binary bits ($2^3 = 8$), any 3-bit binary sequence can be bijectively embedded into 2 trits with **zero information loss**.

### Radix Transformation Formulation
The 12-bit vector $X_{\text{bin}}$ is partitioned into $M = 4$ independent 3-bit blocks:
$$B_k = [b_{3k}, b_{3k+1}, b_{3k+2}], \quad k \in \{0, 1, 2, 3\}$$

Each 3-bit block is converted to its integer magnitude:
$$I_k = \sum_{j=0}^{2} b_{3k+j} \cdot 2^{2-j} \in \{0, 1, \dots, 7\}$$

The integer $I_k$ is decomposed into a base-3 pair of trits $(t_{2k}, t_{2k+1})$:
$$t_{2k} = \lfloor I_k / 3 \rfloor \in \{0, 1, 2\}$$
$$t_{2k+1} = I_k \bmod 3 \in \{0, 1, 2\}$$

Because $\max(I_k) = 7 < 8$, the combination $(t_{2k}, t_{2k+1}) = (2, 2)$ (which equals 8 in base-3) is never addressed, guaranteeing that no state overflow occurs.

### Wire Reduction Metric
* **Standard Qubit Requirement:** 12 wires
* **Ternary Qutrit Requirement:** 8 wires
$$\Delta_{\text{wires}} = \frac{12 - 8}{12} = 33.33\% \text{ reduction}$$

---

## 4. Qutrit Variational Quantum Classifier (VQC)

The 8-trit sequence $T = [t_0, t_1, \dots, t_7]$ is initialized onto an 8-wire quantum register:
$$|\psi_0\rangle = \bigotimes_{w=0}^{7} |t_w\rangle \in (\mathbb{C}^3)^{\otimes 8}$$

### Variational Layer Structure
The ansatz comprises $L = 3$ identical variational layers. Each layer consists of:

#### 1. Data Re-uploading Embeddings
To avoid barren plateaus and increase expressivity, input trits are re-embedded at each layer via parameterized rotations:
$$U_{\text{enc}}^{(l)}(t_w) = \exp\left(-i \frac{\pi t_w}{2} \hat{\lambda}_3\right)$$

#### 2. Trainable SU(3) Subgroup Rotations
A single qutrit state is spanned by the 8 Gell-Mann generators $\hat{\lambda}_1, \dots, \hat{\lambda}_8$. We parameterize each wire using three Givens rotations across the $\{|0\rangle, |1\rangle\}$ and $\{|1\rangle, |2\rangle\}$ subspaces:
$$U_{\text{var}}(\boldsymbol{\theta}_{l, w}) = R_{x}^{[0,1]}(\theta_{l,w,0}) \cdot R_{z}^{[0,1]}(\theta_{l,w,1}) \cdot R_{x}^{[1,2]}(\theta_{l,w,2})$$
* **Parameter Footprint:** $L \times W \times 3 = 3 \times 8 \times 3 = \mathbf{72 \text{ trainable parameters}}$.

#### 3. Controlled-SUM (CSUM) Ring Entanglement
To generate multi-particle entanglement without qubit overhead, two-qutrit CSUM gates are applied circularly:
$$\text{CSUM} |a, b\rangle = |a, (a + b) \bmod 3\rangle, \quad a, b \in \{0, 1, 2\}$$
Entanglement is wired in a periodic ring topology:
$$U_{\text{ent}} = \prod_{w=0}^{7} \text{CSUM}_{(w, (w+1) \bmod 8)}$$

---

## 5. Observable Measurement & Prediction

Measurement is performed on all 8 wires using the third Gell-Mann matrix ($\hat{\lambda}_3$):
$$\hat{\lambda}_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

The expectation value for wire $w$ is:
$$\langle \hat{\lambda}_3^{(w)} \rangle = \langle \psi_{\text{final}} | (\mathbb{I} \otimes \dots \otimes \hat{\lambda}_3^{(w)} \otimes \dots \otimes \mathbb{I}) | \psi_{\text{final}} \rangle \in [-1, 1]$$

The classification output $\hat{y} \in (0, 1)$ is evaluated via a collective sigmoid activation:
$$\hat{y} = \sigma\left(\sum_{w=0}^{7} \langle \hat{\lambda}_3^{(w)} \rangle\right) = \frac{1}{1 + \exp\left(-\sum_{w=0}^{7} \langle \hat{\lambda}_3^{(w)} \rangle\right)}$$

Optimization is driven by Binary Cross-Entropy (BCE) loss using the Adam optimizer with initial learning rate $\eta = 0.01$.
