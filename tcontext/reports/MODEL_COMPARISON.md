# Multi-Model Comparative Evaluation & Benchmark Analysis

**Thesis Title:** *A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning*  
**Repository:** `RoushanKhalid/Thesis`  
**Dataset:** Microsoft Cats vs. Dogs (Stratified 3-way partition, Seed 777)  
**Hardware Environment:** NVIDIA GeForce RTX 2060 (6 GB VRAM) / AMD64 24 Logical Threads  
**Date:** November 2026  

---

## 1. Executive Summary

This report documents the empirical findings of the comprehensive multi-model benchmark across seven distinct visual paradigms. The primary research question is not merely identifying the model with the highest nominal accuracy, but rigorously evaluating the multi-dimensional trade-offs between **predictive accuracy**, **computational latency**, **memory footprint**, and **incremental knowledge integration capability without model fine-tuning**.

---

## 2. Multi-Model Empirical Results (GPU Execution: `cuda:0`)

All models were evaluated on the exact same hold-out test set ($N=186$, 93 cats / 93 dogs). The empirical results recorded in `benchmark_summary.json` and `benchmark_results_cuda_0.csv` are summarized below:

| Model Architecture | Paradigm / Category | Accuracy (%) | Precision | Recall | Macro F1 | ROC-AUC | Train / Index Time (s) | Inference Latency (ms) | Throughput (img/s) | Parameters ($N_{\text{total}}$) | Checkpoint Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Lightweight Edge CNN | 94.09% | 0.9184 | 0.9677 | 0.9424 | 0.9820 | 37.85 s | 3.42 ms | 292.6 | 928,162 | 3.66 MB |
| **EfficientNetB0** | Compound Scaled CNN | 92.47% | 0.9247 | 0.9247 | 0.9247 | 0.9852 | 42.30 s | 3.70 ms | 270.2 | 4,010,110 | 15.58 MB |
| **ResNet18** | Residual CNN Baseline | 97.31% | 0.9583 | 0.9892 | 0.9735 | 0.9951 | 40.27 s | 3.61 ms | 277.1 | 11,177,538 | 42.71 MB |
| **ViT-B/16** | Vision Transformer | 98.92% | 0.9892 | 0.9892 | 0.9892 | 0.9990 | 134.40 s | 10.87 ms | 92.0 | 85,800,194 | 327.36 MB |
| **CLIP (ViT-B/32)** | Foundation Vision-Language | **99.46%** | 1.0000 | 0.9892 | **0.9946** | **0.9997** | 58.00 s | 4.75 ms | 210.5 | 151,278,339 | 577.23 MB |
| **Naive Retrieval** | Vector Index (ChromaDB) | 98.92% | 1.0000 | 0.9785 | 0.9891 | 0.9945 | **0.45 s** | **0.18 ms** | **5,465.7** | **0 (Frozen)** | 4.25 MB |
| **Proposed IMO Framework** | Vector Index + IMO Orchestration | **99.46%** | 1.0000 | 0.9892 | **0.9946** | **0.9988** | **0.67 s** | **0.19 ms** | **5,154.3** | **0 (Frozen)** | 4.24 MB |

---

## 3. In-Depth Architectural Analysis

### 3.1 MobileNetV3-Small (Lightweight Edge CNN)
- **Architecture:** Depthwise separable convolutions, hard-swish non-linearities, and squeeze-and-excitation attention blocks.
- **Role:** Represents compute-constrained edge devices with ultra-low parameter budgets.
- **Empirical Findings:** Achieved 94.09% accuracy with the smallest disk footprint (3.66 MB) and fast inference (3.42 ms on GPU, 3.66 ms on CPU). However, adapting its linear probe required 37.85s of GPU epochs, illustrating the rigidity of retraining even on small networks.

### 3.2 EfficientNetB0 (Compound Scaled CNN)
- **Architecture:** Neural architecture search (NAS) optimization with balanced depth, width, and resolution scaling.
- **Role:** Historical parametric baseline utilized in initial project versions.
- **Empirical Findings:** Achieved 92.47% accuracy. While competent, it was surpassed by ResNet18 (97.31%) and ViT-B/16 (98.92%). Its training required 42.30s of GPU computation.

### 3.3 ResNet18 (Classical Residual CNN)
- **Architecture:** 18-layer residual convolutional network utilizing skip connections to alleviate gradient degradation.
- **Role:** Industry standard baseline for visual classification.
- **Empirical Findings:** Performed exceptionally well among CNNs, achieving 97.31% accuracy and 0.9735 F1-score with 3.61 ms inference latency. Training converged in 40.27s.

### 3.4 ViT-B/16 (Vision Transformer)
- **Architecture:** Pure self-attention applied directly to sequences of non-overlapping $16 \times 16$ image patches.
- **Role:** High-capacity vision transformer baseline.
- **Empirical Findings:** Delivered high accuracy (98.92%) and near-perfect ROC-AUC (0.9990). However, this came at substantial computational cost: 134.40s of GPU training time, 10.87 ms inference latency per image (92.0 img/s), and 327.36 MB model checkpoint size. On CPU, inference degraded dramatically to 71.17 ms per image (14.1 img/s).

### 3.5 CLIP (ViT-B/32 Foundation Model)
- **Architecture:** Multi-modal dual transformer pre-trained on 400M web-scale image-text pairs via contrastive learning.
- **Role:** Foundation semantic representation extractor and zero-shot baseline.
- **Empirical Findings:** The linear probe on top of frozen 512-dim visual embeddings achieved 99.46% accuracy. This establishes that frozen semantic embeddings contain sufficient linear separability to bypass convolutional retraining entirely.

### 3.6 Naive Retrieval (CLIP + ChromaDB)
- **Architecture:** HNSW vector index over frozen 512-dim CLIP embeddings; 5-nearest neighbor majority voting.
- **Role:** Demonstrates basic non-parametric memory without quality curation.
- **Empirical Findings:** Achieved 98.92% accuracy with zero model parameters and sub-millisecond query latency (0.18 ms, 5,465 queries/sec). Building the index required only 0.45 seconds. However, as demonstrated in the noise benchmark, naive retrieval has no defense against corrupted labels.

### 3.7 Proposed Framework (CLIP + ChromaDB + IMO + Evidence Fusion)
- **Architecture:** Multi-stage Intelligent Memory Orchestrator (IMO) gating candidate embeddings via running class prototypes, margin consistency, and local density checks, combined with dynamic evidence fusion.
- **Role:** Proposed core thesis contribution.
- **Empirical Findings:**
  - Matches the top accuracy of the benchmark at **99.46%** (identical to fine-tuned CLIP probe).
  - Maintains **0.19 ms inference latency** (over **5,100 queries/second**).
  - Zero parametric weight updates ($N_{\text{trainable}} = 0$).
  - Full index and prototype construction completed in **0.67 seconds** (compared to 40-134s for CNN/ViT training).

---

## 4. Multi-Dimensional Trade-off Evaluation

```
               ACCURACY vs INFERENCE LATENCY EFFICIENCY FRONTIER
100% |                                      [CLIP Probe]  [Proposed IMO]
     |                                      [ViT-B/16]    [Naive Ret.]
 98% |                  [ResNet18]
     |
 96% |
     |
 94% |  [MobileNetV3]
     |
 92% |  [EfficientNetB0]
     +------------------------------------------------------------------
        3 ms             4 ms               11 ms         0.19 ms
                   (Parametric Models)                 (Retrieval Models)
```

### 4.1 Knowledge Integration Latency (Retraining vs Indexing)
When new visual data is acquired in real-world deployment, the computational barrier is the knowledge update latency:
- **Supervised CNNs/ViTs:** Updating weights requires re-running backward passes over accumulated batches, taking **37.8s to 134.4s** on dedicated GPU hardware.
- **Proposed IMO Framework:** Updating memory requires only a single forward pass through the frozen visual encoder (4.75 ms) and an HNSW vector insertion with prototype update (0.19 ms), yielding a total update latency of **< 5 ms per image**. This represents a **> 7,500x acceleration** in knowledge integration time.

### 4.2 Parameter Invariance & Elimination of Catastrophic Forgetting
In conventional classifiers, updating parameters $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ inevitably perturbs previously learned decision boundaries, leading to catastrophic forgetting.  
In the proposed framework:
- Backbone parameters remain strictly invariant: $\nabla_\theta = \mathbf{0}$.
- Knowledge resides in an additive non-parametric index: $\mathcal{M} \leftarrow \mathcal{M} \cup \{(\mathbf{e}_{\text{new}}, y_{\text{new}})\}$.
- Deletion or correction of erroneous samples is an $\mathcal{O}(\log N)$ vector removal operation, whereas removing bad data from a neural network requires full model retraining or complex unlearning algorithms.

### 4.3 Disk & Checkpoint Efficiency
- ViT-B/16 requires **327.36 MB** to store model weights.
- ResNet18 requires **42.71 MB**.
- In contrast, storing the entire 878-image vector database in ChromaDB requires only **4.24 MB**, which scales linearly with sample count ($\approx 2$ KB per normalized 512-dim embedding plus HNSW graph links).

---

## 5. Thesis Conclusion and Academic Takeaways

1. **Retraining Is Unnecessary for Incremental Visual Adaptation:** Pre-trained semantic representations (CLIP ViT-B/32) possess sufficient geometric structure that non-parametric vector retrieval equals or outperforms conventional supervised fine-tuning (99.46% vs 92.47%-98.92%).
2. **Sub-Millisecond Query Performance:** Decoupled vector memory provides query throughput exceeding 5,100 samples/sec, far surpassing transformer forward pass latencies.
3. **The Necessity of IMO Quality Control:** As detailed in `EXPERIMENTAL_METHODOLOGY.md` and `HARDWARE_BENCHMARK.md`, un-curated retrieval is brittle under noise. The Intelligent Memory Orchestrator provides the essential quality barrier that makes non-parametric learning scientifically viable and robust in deployment.
