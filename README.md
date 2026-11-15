# A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1+cu121](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![VectorDB ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Dashboard Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Target: Springer JIIS](https://img.shields.io/badge/Target%20Journal-Springer%20JIIS-brightgreen.svg)](https://www.springer.com/journal/10844)

> **Official Research Repository & Implementation**  
> **Authors:** Sk. Roushan Khalid$^1$ and Md. Shohel Arman$^1$  
> $^1$ *Department of Software Engineering, Daffodil International University, Dhaka, Bangladesh*  
> **Manuscript Target:** *Journal of Intelligent Information Systems (Springer JIIS)*

---

## 📑 Table of Contents
1. [Research Overview](#-1-research-overview)
2. [Problem Statement & Motivation](#-2-problem-statement--motivation)
3. [Research Objectives & Hypotheses](#-3-research-objectives--hypotheses)
4. [Proposed Framework Architecture](#-4-proposed-framework-architecture)
5. [Step-by-Step Methodological Workflow](#-5-step-by-step-methodological-workflow)
6. [Dataset & Leak-Free Partitioning Protocol](#-6-dataset--leak-free-partitioning-protocol)
7. [Evaluated Visual Paradigms](#-7-evaluated-visual-paradigms)
8. [Experimental Results & Empirical Findings](#-8-experimental-results--empirical-findings)
9. [Hardware Resource Profiling (GPU vs. CPU)](#-9-hardware-resource-profiling-gpu-vs-cpu)
10. [Noise Robustness Stress Test (Hypothesis H1)](#-10-noise-robustness-stress-test-hypothesis-h1)
11. [Scientific Contributions & Novelty](#-11-scientific-contributions--novelty)
12. [Threats to Validity & Limitations](#-12-threats-to-validity--limitations)
13. [Repository Organization](#-13-repository-organization)
14. [Installation & Reproduction Guide](#-14-installation--reproduction-guide)
15. [Citation & References](#-15-citation--references)

---

## 🔬 1. Research Overview

Continuous real-world deployment of visual intelligence requires systems to assimilate incoming visual knowledge on the fly. Traditional deep learning frameworks rely on iterative gradient backpropagation and parameter fine-tuning to incorporate new data. While effective in static environments, gradient adaptation causes **catastrophic forgetting**, demands heavy **GPU energy and wall-clock retraining**, and remains dangerously **vulnerable to label corruption**.

This research introduces a non-parametric alternative: **A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning**. The core paradigm decouples visual representation learning from incremental knowledge retention:

- **Frozen Visual Backbone:** A foundation vision-language encoder (CLIP ViT-B/32) is kept strictly invariant ($\nabla_\theta = \mathbf{0}$), mathematically eliminating catastrophic forgetting.
- **Metric Retrieval Adapter:** A lightweight residual projection head ($g_\phi$, 526k parameters) optimized via Supervised Contrastive Loss aligns representations to the target domain manifold in $<1$ second.
- **Intelligent Memory Orchestrator (IMO):** An automated multi-stage geometric firewall evaluating candidate embeddings against running class prototypes, relative classification margins, and neighborhood outlier density before committing instances into memory.
- **Non-Destructive Quarantine Buffer ($\mathcal{Q}$):** Suspicious, ambiguous, or corrupted samples are diverted away from active memory into an isolated audit buffer without data loss.
- **Dynamic Evidence Fusion:** A calibrated Bayesian consensus combining base model predictions with non-parametric retrieval evidence.

```
+---------------------------------------------------------------------------------------------------------+
|                                    TRADITIONAL vs. PROPOSED PARADIGM                                    |
+---------------------------------------------------------------------------------------------------------+
| Traditional Fine-Tuning:   New Data  -->  Backpropagation (116s - 480s)  -->  Weights Altered (Forgetting)|
| Proposed IMO Framework:    New Data  -->  IMO Gating Check (<5 ms)       -->  Indexed into Vector Memory |
|                                                                               (Backbone ∇_θ = 0 Invariant) |
+---------------------------------------------------------------------------------------------------------+
```

---

## ⚠️ 2. Problem Statement & Motivation

When intelligent systems deployed on mobile, edge, or robotics platforms encounter new visual data streams, gradient-based retraining exposes three critical bottlenecks:

```mermaid
flowchart LR
    A["Iterative Gradient Retraining\nθ ← θ - η ∇_θ L"] --> B["1. Catastrophic Forgetting\nHistorical decision boundaries drift;\nPrior classes degrade"]
    A --> C["2. Computational Inefficiency\nHundreds of seconds of GPU load;\nHigh VRAM, thermal throttling"]
    A --> D["3. Vulnerability to Noise\nBackpropagation absorbs mislabeled inputs\ninto global weights irreversibly"]
```

### Comparative Analysis: Parametric Fine-Tuning vs. Proposed Non-Parametric Memory

| Scientific Dimension | Conventional Parametric Fine-Tuning | Proposed Retrieval Framework (IMO) |
| :--- | :--- | :--- |
| **Knowledge Representation** | Distributed across synaptic weights $\theta$ | Additive external vector memory $\mathcal{M}$ (ChromaDB HNSW) |
| **Backbone Invariance** | $\nabla_\theta \ne \mathbf{0}$ (Continuous parameter drift) | $\nabla_\theta = \mathbf{0}$ (**Strict bit-for-bit invariance**) |
| **Catastrophic Forgetting** | High risk; requires replay coresets | **Mathematically eliminated by design** |
| **Knowledge Update Latency** | Slow: 116.0 s (MobileNet) to 480.0 s (ViT-B/16) | **Fast: $<5.0$ ms per image** (Ingestion + Indexing: 1.27 s total) |
| **Hardware Dependency** | Heavy GPU VRAM requirement ($>1,000$ MB) | **0.00 MB dedicated GPU VRAM**; runs in host L3 cache |
| **Noise Resilience** | Poor: Corrupted samples degrade global weights | **Robust: IMO intercepts 100% of corrupted samples** into quarantine |
| **Explainability** | Black-box latent weights | **Inspectable nearest neighbors and explicit provenance** |

---

## 🎯 3. Research Objectives & Hypotheses

### Primary Research Questions (RQs)
- **RQ1 (Predictive Parity):** Can non-parametric vector retrieval over frozen foundation representations equal or exceed the accuracy of fully fine-tuned parametric convolutional and vision transformer models?
- **RQ2 (Noise Immunity):** Can automated prototype consistency and relative margin verification insulate external memory against high-rate label corruption without manual supervision?
- **RQ3 (Hardware Efficiency):** How do inference latency, throughput, energy footprint, and memory scale across dedicated GPU and CPU constraints?

### Testable Scientific Hypotheses
- **Hypothesis H1 (Noise Robustness):** Under increasing synthetic label noise ($0\%$ to $30\%$), the Intelligent Memory Orchestrator (IMO) preserves test accuracy by quarantining corrupted instances, whereas naive vector retrieval suffers rapid degradation.
- **Hypothesis H2 (Computational Acceleration):** Non-parametric vector ingestion achieves $>200\times$ acceleration in knowledge update time compared to two-stage gradient fine-tuning.
- **Hypothesis H3 (Hardware Invariance):** Vector graph traversal operates in host cache memory with sub-millisecond latency ($<0.20$ ms) invariant to GPU acceleration.

---

## 🏗️ 4. Proposed Framework Architecture

The framework decouples continual knowledge acquisition from parametric model weights by organizing dataflow into two strictly separated, left-to-right operational streams:
1. **Stage A: Incremental Knowledge Ingestion & Memory Curation Stream (Continual Learning):** Labeled candidate samples $(\mathbf{x}_t, y_t)$ are projected through the invariant frozen backbone (CLIP ViT-B/32, $\nabla_\theta = \mathbf{0}$) and contrastive metric adapter ($g_\phi$) before undergoing multi-stage geometric verification by the Intelligent Memory Orchestrator (IMO). Verified samples commit into active ChromaDB HNSW vector memory $\mathcal{M}$ and update running prototypes $\mathbf{p}_c$, while corrupted samples divert safely to the isolated quarantine buffer $\mathcal{Q}$.
2. **Stage B: Sub-Millisecond Online Inference & Dynamic Evidence Fusion Stream (Query Decision):** Unlabeled inference queries $\mathbf{x}_q$ evaluate dual complementary evidence paths—a parametric model prior $P_{\text{clf}}$ and a sub-millisecond non-parametric retrieval posterior $P_{\text{ret}}$ from active memory $\mathcal{M}$ ($0.15\text{ ms}$, 0 MB GPU VRAM)—synthesized via validation-calibrated Dynamic Evidence Fusion ($\alpha^* = 0.60, k^* = 1$) to achieve 100.0% test accuracy.

![System Architecture](paper_latex/figures/fig1_architecture.png)

---

## 🔄 5. Step-by-Step Methodological Workflow

```mermaid
flowchart TD
    subgraph S1["1. Representation Extraction"]
        X["Input Visual Sample x\n(RGB 224x224)"] --> ENC["Frozen Semantic Visual Backbone\nCLIP ViT-B/32 (f_θ, ∇_θ = 0)"]
        ENC --> E["Normalized Embedding\ne = f_θ(x) / ||f_θ(x)|| ∈ R^512"]
        E --> ADAPT["Metric Retrieval Adapter (g_φ)\nz = Normalize(e + g_φ(e))"]
    end

    subgraph S2["2. Incremental Knowledge Ingestion (IMO Quality Firewall)"]
        ADAPT -->|"Candidate (z_t, y_t)"| G1{"Gate 1: Consistency Check\ncos(z_t, p_y) ≥ τ_c (0.65)?"}
        G1 -->|Fail| Q["Quarantine Buffer (Q)\nIsolated Audit Log"]
        G1 -->|Pass| G2{"Gate 2: Margin Check\ncos(z_t, p_y) - max cos(z_t, p_j) ≥ τ_m (0.00)?"}
        G2 -->|Fail| Q
        G2 -->|Pass| G3{"Gate 3: Outlier Density\nMean Dist(K=5) ≤ τ_outlier (0.45)?"}
        G3 -->|Fail| Q
        G3 -->|Pass All Gates| MEM["Commit to ChromaDB HNSW Index (M)\nUpdate Running Class Centroid p_y"]
    end

    subgraph S3["3. Inference & Decision Fusion"]
        ADAPT -.->|"Query z_q"| HNSW["HNSW ANN Retrieval\nCosine Metric Space (Top-k)"]
        MEM <--> HNSW
        HNSW --> PRET["Retrieval Posterior: P_ret(y|x)"]
        X --> BASE["Parametric Base Classifier\n(e.g., EfficientNetB0)"]
        BASE --> PCLF["Parametric Posterior: P_clf(y|x)"]
        PRET --> FUSE["Dynamic Evidence Fusion\nP(y|x) = α* P_clf + (1-α*) P_ret\n(α* = 0.60, k* = 1)"]
        PCLF --> FUSE
        FUSE --> PRED["Final Prediction: y* = argmax P(y|x)"]
    end
```

### Mathematical Formulation of Technical Components

#### 1. Invariant Semantic Representation & Metric Adapter
The frozen visual backbone extracts normalized unit embeddings:
$$\mathbf{e} = \frac{f_\theta(\mathbf{x})}{\|f_\theta(\mathbf{x})\|_2} \in \mathbb{R}^{512}, \quad \|\mathbf{e}\|_2 = 1, \quad \nabla_\theta = \mathbf{0}$$

A lightweight residual projection head $g_\phi$ maps $\mathbf{e}$ into the task-adapted metric space:
$$\mathbf{z} = \text{Normalize}\left(\mathbf{e} + g_\phi(\mathbf{e})\right) = \frac{\mathbf{e} + \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2}{\|\mathbf{e} + \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2\|_2}$$

Trained on $\mathcal{D}_{\text{train}}$ via Supervised Contrastive Loss ($\tau = 0.07$):
$$\mathcal{L}_{\text{supcon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i^\top \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i^\top \mathbf{z}_a / \tau)}$$

#### 2. Intelligent Memory Orchestrator (IMO) Multi-Stage Ingestion Pipeline
To protect memory against label noise and distribution drift, every incoming candidate $(\mathbf{z}_t, y_t)$ must pass three sequential quality gates:

![IMO Ingestion Pipeline](paper_latex/figures/fig6_imo_pipeline.png)

1. **Running Class Prototypes:** For each category $c \in \mathcal{Y}$, the IMO maintains a normalized running centroid $\mathbf{p}_c$:
   $$\mathbf{p}_c^{(t)} = \frac{n_c \mathbf{p}_c^{(t-1)} + \mathbf{z}_t}{\|n_c \mathbf{p}_c^{(t-1)} + \mathbf{z}_t\|_2}, \quad n_c \leftarrow n_c + 1$$
2. **Gate 1 (Prototype Consistency):** Verifies that candidate embedding $\mathbf{z}_t$ aligns with its claimed category prototype:
   $$\text{Gate}_1: \quad \mathbf{z}_t^\top \mathbf{p}_{y_t} \ge \tau_c \quad (\tau_c = 0.65)$$
3. **Gate 2 (Relative Margin Conflict Check):** Ensures candidate similarity to claimed class exceeds competing class prototypes:
   $$\text{Gate}_2: \quad \mathbf{z}_t^\top \mathbf{p}_{y_t} - \max_{j \ne y_t} \left(\mathbf{z}_t^\top \mathbf{p}_j\right) \ge \tau_m \quad (\tau_m = 0.00)$$
4. **Gate 3 (Outlier Density Filtering):** Checks local $K$-nearest neighbor distance against active memory ($K=5$):
   $$\text{Gate}_3: \quad \bar{D} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{z}_t)} D_{\text{cos}}(\mathbf{z}_t, \mathbf{z}_i) \le \tau_{\text{outlier}} \quad (\tau_{\text{outlier}} = 0.45)$$
5. **Non-Destructive Quarantine:** Rejected samples are preserved in an isolated buffer with forensic rejection metadata:
   $$\mathcal{Q} = \{(\mathbf{z}_t, y_t, \text{reason}, \text{timestamp}, \text{metadata})\}$$

#### 3. Dynamic Evidence Fusion Engine
During inference, retrieval posteriors are computed over the top-$k^*$ nearest neighbors retrieved from ChromaDB:
$$P_{\text{ret}}(y=k \mid \mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_{k^*}(\mathbf{z})} \mathbb{I}(y_i = k) \cdot \exp\left(\mathbf{z}^\top \mathbf{z}_i / T\right)}{\sum_{j \in \mathcal{N}_{k^*}(\mathbf{z})} \exp\left(\mathbf{z}^\top \mathbf{z}_j / T\right)} \quad (T = 0.10)$$

Final predictions combine parametric and non-parametric posteriors via convex consensus:
$$P(y = k \mid \mathbf{x}) = \alpha^* P_{\text{clf}}(y = k \mid \mathbf{x}) + (1 - \alpha^*) P_{\text{ret}}(y = k \mid \mathbf{x})$$
Hyperparameters calibrated on hold-out validation data ($\mathcal{D}_{\text{val}}$): $\alpha^* = 0.60, k^* = 1$.

---

## 📊 6. Dataset & Leak-Free Partitioning Protocol

Experiments utilize the standardized **Microsoft Cats vs. Dogs** benchmark ($N = 2,500$ verified RGB images):
- **Class Balance:** Exactly 1,250 Cat images (50.0%) and 1,250 Dog images (50.0%).
- **Integrity Audit:** 0 corrupt files; 2,500 unique SHA-256 hashes (zero duplicate images).

### Deterministic 3-Way Stratified Partition (Seed 777)

$$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad \mathcal{D}_i \cap \mathcal{D}_j = \emptyset \quad \forall i \ne j$$

| Partition Split | Proportion | Total Samples | Cats Count | Dogs Count | Experimental Function |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Split ($\mathcal{D}_{\text{train}}$)** | 70% | 1,752 | 876 | 876 | Two-stage backbone fine-tuning; metric adapter fitting; vector memory index pool |
| **Validation Split ($\mathcal{D}_{\text{val}}$)** | 15% | 374 | 187 | 187 | Early stopping checkpointing; retrieval hyperparameter calibration ($k^*, \alpha^*$) |
| **Hold-Out Test Split ($\mathcal{D}_{\text{test}}$)** | 15% | 374 | 187 | 187 | **Strictly isolated evaluation set**; identical hold-out test across all 8 models |

*Vector Index Leakage Verification:* Inspection of ChromaDB persistent indices verified that exactly **0 test instances** exist in memory, guaranteeing 100% leak-free evaluation.

---

## 🤖 7. Evaluated Visual Paradigms

Eight architectural paradigms spanning four major visual learning families were evaluated:

1. **Lightweight Edge CNN:** `MobileNetV3-Small` (928k parameters) — Depthwise separable convolutions + SE attention.
2. **Compound Scaled CNN:** `EfficientNetB0` (4.01M parameters) — NAS-optimized inverted residual blocks (MBConv).
3. **Residual CNN Standard:** `ResNet18` (11.18M parameters) — Classical residual connections ($\mathbf{x} + \mathcal{F}(\mathbf{x})$).
4. **Vision Transformer:** `ViT-B/16` (85.80M parameters) — Pure multi-head self-attention over $16 \times 16$ image patches.
5. **Vision-Language Foundation:** `CLIP (ViT-B/32)` (151.28M parameters) — Zero-shot linear probe over contrastive features.
6. **Naive Vector Retrieval:** `CLIP + ChromaDB` (0 parameters) — Unfiltered HNSW majority voting.
7. **Fine-Tuned Retrieval Adapter:** `Metric Adapter + ChromaDB` (526k parameters) — Supervised contrastive metric alignment.
8. **Proposed Framework:** `CLIP + ChromaDB + IMO + Evidence Fusion` (0 backbone params, $\nabla_\theta = \mathbf{0}$) — **Proposed Thesis Contribution**.

---

## 🏆 8. Experimental Results & Empirical Findings

All models were evaluated on the exact same hold-out test set ($N=374$, 187 cats / 187 dogs) on an NVIDIA GeForce RTX 2060 GPU (`cuda:0`).

### Primary Multi-Model GPU Leaderboard

| Model Architecture | Category | Accuracy (%) | Precision | Recall | Macro F1 | ROC-AUC | Train / Index Time | Inference Latency | Throughput (img/s) | Parameters ($N_{\text{total}}$) | Checkpoint Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Edge CNN | 97.33% | 0.9733 | 0.9733 | 0.9733 | 0.9963 | 116.05 s | 3.30 ms | 302.8 | 928,162 | 3.66 MB |
| **EfficientNetB0** | Compound CNN | 98.66% | 0.9840 | 0.9893 | 0.9867 | 0.9998 | 175.18 s | 3.75 ms | 266.9 | 4,010,110 | 15.58 MB |
| **ResNet18** | Residual CNN | 99.20% | 0.9894 | 0.9947 | 0.9920 | 0.9990 | 149.20 s | 3.69 ms | 271.0 | 11,177,538 | 42.71 MB |
| **ViT-B/16** | Vision Transformer | 99.47% | 0.9947 | 0.9947 | 0.9947 | 0.9997 | 479.95 s | 10.98 ms | 91.0 | 85,800,194 | 327.36 MB |
| **CLIP (ViT-B/32)** | Foundation Vision-Language | 99.73% | 0.9947 | 1.0000 | 0.9973 | 1.0000 | 173.16 s | 4.43 ms | 225.7 | 151,278,339 | 577.23 MB |
| **Naive Retrieval** | Vector Index (ChromaDB) | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.80 s** | **0.18 ms\*** | **5,653.9\*** | **0 (Frozen)** | 11.61 MB |
| **Fine-Tuned Retrieval** | Contrastive Adapter + DB | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.69 s** | **0.14 ms\*** | **7,152.5\*** | 526,337 | 11.60 MB |
| **Proposed IMO Framework** | **Vector Index + IMO** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.27 s** | **0.15 ms\*** | **6,726.0\*** | **0 (Frozen)** | **11.59 MB** |

> **\*Methodological Disclosure on Retrieval Latency:** The 0.14 -- 0.18 ms figure denotes non-parametric HNSW index lookup given pre-extracted 512-dim visual embeddings. True end-to-end inference from an un-encoded image tensor includes the invariant CLIP encoder forward pass (4.43 ms GPU / 17.40 ms CPU), yielding **4.58 ms on GPU** (4.43 ms encode + 0.15 ms search) and **17.55 ms on CPU**.

![Accuracy and Adaptation Time Comparison](paper_latex/figures/fig2_accuracy_comparison.png)

![Inference Efficiency Frontier](paper_latex/figures/fig3_latency_accuracy_tradeoff.png)

### Key Findings
1. **Confirmation of RQ1:** Non-parametric vector retrieval over frozen CLIP features achieves 100.0% test accuracy, surpassing fine-tuned CNN baselines (97.33% to 99.20%) and ViT-B/16 (99.47%) without altering backbone weights.
2. **Knowledge Ingestion Speedup:** Indexing all 1,752 images into ChromaDB completed in **1.27 seconds**, compared to **116s -- 480s** for parametric backpropagation (over **200x speedup**).
3. **Per-Image Ingestion:** Updating memory with a new sample takes $<5.0$ ms total (4.43 ms forward pass + 0.15 ms HNSW insertion + prototype update), compared to hundreds of seconds of backpropagation.

---

## ⚡ 9. Hardware Resource Profiling (GPU vs. CPU)

To address **RQ3**, each architecture was evaluated under two strictly isolated regimes:
1. **GPU-Accelerated Mode (`cuda:0`):** NVIDIA GeForce RTX 2060 (6 GB GDDR6).
2. **CPU-Constrained Mode (`cpu`):** 24-Thread AMD64 host processor.

| Model Architecture | GPU Latency | GPU Throughput | CPU Latency | CPU Throughput | Hardware Speedup Factor | Peak GPU VRAM | Checkpoint Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 3.30 ms | 302.8 img/s | 3.05 ms | 328.2 img/s | **0.92 $\times$** | 103.96 MB | 3.66 MB |
| **EfficientNetB0** | 3.75 ms | 266.9 img/s | 9.82 ms | 101.9 img/s | **2.62 $\times$** | 394.38 MB | 15.58 MB |
| **ResNet18** | 3.69 ms | 271.0 img/s | 10.07 ms | 99.3 img/s | **2.73 $\times$** | 370.00 MB | 42.71 MB |
| **ViT-B/16** | 10.98 ms | 91.0 img/s | 62.40 ms | 16.0 img/s | **5.68 $\times$** | 1,066.48 MB | 327.36 MB |
| **CLIP (ViT-B/32)** | 4.43 ms | 225.7 img/s | 17.40 ms | 57.5 img/s | **3.93 $\times$** | 1,541.63 MB | 577.23 MB |
| **Naive Retrieval** | **0.18 ms\*** | **5,653.9 img/s\*** | **0.18 ms\*** | **5,653.9 img/s\*** | **1.00 $\times$** | **0.00 MB** | 11.61 MB |
| **Fine-Tuned Retrieval** | **0.14 ms\*** | **7,152.5 img/s\*** | **0.14 ms\*** | **7,152.5 img/s\*** | **1.00 $\times$** | **0.00 MB** | 11.60 MB |
| **Proposed IMO Framework** | **0.15 ms\*** | **6,726.0 img/s\*** | **0.15 ms\*** | **6,726.0 img/s\*** | **1.00 $\times$** | **0.00 MB** | **11.59 MB** |

$$\text{Speedup Factor} = \frac{\text{CPU Latency}}{\text{GPU Latency}}$$

![Hardware Profiling CPU vs GPU](paper_latex/figures/fig4_cpu_vs_gpu_speedup.png)

### Hardware Insights
- **Vision Transformers Exhibit Severe GPU Dependency (5.68x):** Self-attention requires dense matrix operations mapping to Tensor Cores; on CPU, latency surges to 62.40 ms (throughput drops to 16.0 img/s).
- **MobileNetV3-Small is Bandwidth-Bound (0.92x):** At batch size 1, CPU execution (3.05 ms) slightly outperforms GPU (3.30 ms) due to zero kernel launch and PCIe synchronization overhead.
- **Vector Retrieval Latency Is Hardware-Invariant (1.00x):** HNSW graph traversal is pointer-chasing in host L3 cache, executing in **0.15 ms** on both CPU and GPU while consuming **0.00 MB of GPU VRAM**.

---

## 🛡️ 10. Noise Robustness Stress Test (Hypothesis H1)

To evaluate **RQ2** and validate **Hypothesis H1**, synthetic symmetric label noise was injected into the training pool (1,752 instances) at rates $r \in \{0.00, 0.10, 0.20, 0.30\}$ (fixed seed 777).

| Injected Noise Rate | Naive Retrieval Accuracy | Proposed IMO Accuracy | IMO Quarantined Instances | IMO Acceptance Rate | IMO Accuracy Retention Margin |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Clean)** | 100.00% | 100.00% | 0 / 1,752 | 100.0% | +0.00% |
| **10% Noise** | 98.40% | 100.00% | 175 / 1,752 | 90.0% | **+1.60%** |
| **20% Noise** | 91.98% | 100.00% | 350 / 1,752 | 80.0% | **+8.02%** |
| **30% Noise** | 80.48% | 100.00% | 525 / 1,752 | 70.0% | **+19.52%** |

![Noise Robustness Stress Test](paper_latex/figures/fig5_noise_robustness_curve.png)

### Scientific Verification of Hypothesis H1
- **Vulnerability of Unregulated Memory:** Without gating, Naive Retrieval degrades catastrophically from 100.0% to **80.48%** under 30% noise.
- **Perfect Interception by IMO:** The Proposed IMO Framework intercepted and quarantined **exactly 100% of injected corrupted instances** (175, 350, and 525 samples), preserving pristine active memory and maintaining **100.00% test accuracy** (+19.52% retention margin).

---

## 💡 11. Scientific Contributions & Novelty

1. **Decoupled Continual Learning Paradigm:** Formulates an architectural separation between invariant semantic representations ($\nabla_\theta = \mathbf{0}$) and additive vector memory, mathematically eliminating catastrophic forgetting.
2. **Intelligent Memory Orchestrator (IMO):** An automated multi-gate geometric firewall combining running class prototypes, relative margin verification, and outlier density filtering to quarantine corrupted data.
3. **Sub-Second Metric Alignment:** A lightweight residual contrastive projection head ($<0.35\%$ params) optimized via Supervised Contrastive Loss in $<1$ second.
4. **Leak-Free 8-Paradigm Benchmark:** A reproducible, leak-free 3-way stratified partition across CNNs, Transformers, Foundation Models, and Vector Indices.
5. **Zero-VRAM Hardware Proof:** Proves that non-parametric vector memory operates with zero GPU VRAM and hardware-invariant sub-millisecond retrieval latency.

---

## ⚠️ 12. Threats to Validity & Limitations

1. **Extreme Out-of-Distribution Domains:** If target visual distributions differ radically from CLIP's pre-training corpus (e.g., synthetic aperture radar or specialized cytology), linear separability of frozen embeddings may decrease.
2. **Vector Memory Scalability:** While HNSW search scales logarithmically $\mathcal{O}(\log N)$, RAM footprint grows linearly ($\approx 6.6$ MB per 1,000 samples). For billion-scale lifelong deployments, product quantization (PQ) or inverted file indexing should be considered.
3. **Dataset Scope:** Experiments are validated on the 2,500-sample Microsoft Cats vs. Dogs benchmark. Generalization across fine-grained 1,000-class taxonomies forms an important future research vector.

---

## 🗂️ 13. Repository Organization

```
Retrieval-Incremental-Learning/
├── configs/                             # Experiment, model, and hardware configurations
│   ├── experiment_config.yaml           # Primary hyperparameters (splits, seeds, IMO bounds)
│   ├── hardware_config.yaml             # GPU memory fractions and thread configs
│   └── models.yaml                      # Model architecture specifications
├── paper_latex/                         # Springer Nature JIIS LaTeX publication manuscript
│   ├── main.tex                         # Complete publication manuscript
│   ├── references.bib                   # Peer-reviewed bibliographical database
│   ├── sn-jnl.cls / *.bst               # Official Springer Nature document class & styles
│   └── figures/                         # Vector PDFs and 300 DPI publication figures (Fig 1-6)
├── paper_latex.zip                      # Overleaf-ready compilation bundle
├── scripts/
│   └── generate_scientific_figures.py   # Master scientific figure generator
├── tcontext/                            # Core research implementation module
│   ├── artifacts/                       # Empirical benchmark CSVs, JSONs, and split caches
│   │   ├── benchmark_results_cuda_0.csv # Official GPU benchmark measurements
│   │   ├── benchmark_results_cpu.csv    # Official CPU benchmark measurements
│   │   ├── benchmark_summary.json       # Consolidated multi-model summary
│   │   ├── noise_experiment_results.json# Noise stress test measurements
│   │   └── stratified_split_full_dataset_seed_777.json # Deterministic split cache
│   ├── data/                            # Dataset ingestion and verification
│   │   ├── dataset_loader.py            # Deterministic 3-way stratified splitter
│   │   └── full_dataset/                # 2,500 verified images (1,250 cats, 1,250 dogs)
│   ├── evaluation/                      # Evaluation suite
│   │   ├── metrics.py                   # Classification metrics (Acc, F1, ROC-AUC)
│   │   ├── noise_benchmarks.py          # Synthetic noise corruption benchmark
│   │   └── resource_tracker.py          # Latency, throughput, RAM & VRAM profiler
│   ├── experiments/
│   │   └── run_benchmark.py             # Master multi-model benchmark orchestrator
│   ├── models/                          # Deep neural network classifier suite
│   │   ├── base_model.py                # Abstract BaseClassifier with 2-stage curriculum
│   │   ├── cnn_models.py                # MobileNetV3-Small, EfficientNetB0, ResNet18
│   │   ├── vit_model.py                 # Vision Transformer ViT-B/16
│   │   └── clip_model.py                # CLIP ViT-B/32 multimodal feature extractor
│   ├── reports/                         # Formal thesis and validation reports
│   │   ├── MODEL_COMPARISON.md          # Multi-model evaluation analysis
│   │   ├── HARDWARE_BENCHMARK.md        # CPU vs. GPU profiling report
│   │   ├── EXPERIMENTAL_METHODOLOGY.md  # Scientific methodology & equations
│   │   └── FINAL_QA_REPORT.md           # Independent QA & validation audit report
│   ├── retrieval/                       # External vector memory & IMO implementation
│   │   ├── imo.py                       # Intelligent Memory Orchestrator
│   │   ├── retrieval_engine.py          # Metric retrieval adapter & fusion engine
│   │   └── vector_store.py              # Persistent ChromaDB HNSW interface
│   └── web_app.py                       # Streamlit interactive research dashboard
├── tests/                               # Comprehensive automated test suite (29 tests)
│   ├── test_hardware_and_invariance.py  # Parameter invariance and hardware tests
│   ├── test_integration_pipeline.py     # End-to-end pipeline integration tests
│   ├── test_unit_imo.py                 # IMO multi-gate verification tests
│   └── ...                              # Data, model, metric, and vector store tests
├── launch_app.ps1                       # One-click dashboard launcher
├── run_demo.ps1                         # One-click demo benchmark runner
├── requirements.txt                     # Pinned runtime dependencies
├── LICENSE                              # MIT License
└── README.md                            # Primary documentation
```

---

## 💻 14. Installation & Reproduction Guide

### Prerequisites
- Windows 10/11 (x64) or Linux (x86_64)
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (Optional, CPU mode supported)

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/RoushanKhalid/Retrieval-Incremental-Learning.git
cd Retrieval-Incremental-Learning

# Install runtime dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Research Dashboard
```powershell
# One-click launcher
.\launch_app.ps1

# Or run directly with Streamlit
streamlit run tcontext\web_app.py --server.port 8501
```

### 3. Reproduce Multi-Model Benchmark (8 Paradigms)
```powershell
# Execute full multi-model benchmark across all 2,500 images
python tcontext\experiments\run_benchmark.py --epochs 15 --seed 777
```

### 4. Regenerate Master Scientific Figures
```powershell
# Regenerate all publication-quality vector PDFs and 300 DPI PNGs
python scripts\generate_scientific_figures.py
```

### 5. Execute Automated Test Suite (29 Tests)
```powershell
# Run all unit, integration, and invariance tests
pytest tests\ -v
```

---

## 📜 15. Citation & References

If you find this research or code helpful in your work, please cite:

```bibtex
@article{khalid2026retrieval,
  title={A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning},
  author={Khalid, Sk. Roushan and Arman, Md. Shohel},
  journal={Journal of Intelligent Information Systems (Under Review)},
  year={2026},
  publisher={Springer Nature}
}
```

### Key References
- **[1] Radford et al. (2021):** *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
- **[2] Malkov & Yashunin (2020):** *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs.* IEEE TPAMI.
- **[3] Khosla et al. (2020):** *Supervised Contrastive Learning.* NeurIPS.
- **[4] Kirkpatrick et al. (2017):** *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS.
- **[5] Parisi et al. (2019):** *Continual Lifelong Learning with Neural Networks: A Review.* Neural Networks.
- **[6] Tan & Le (2019):** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- **[7] Dosovitskiy et al. (2021):** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.

---

## 📄 License
This research repository is licensed under the [MIT License](LICENSE).
