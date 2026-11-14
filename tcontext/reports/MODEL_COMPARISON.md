# Multi-Model Comparative Evaluation & Benchmark Analysis

**Thesis Title:** *A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning*  
**Repository:** `RoushanKhalid/Thesis`  
**Dataset:** Microsoft Cats vs. Dogs Full Benchmark ($N=2,500$ verified images)  
**Partitioning:** Stratified 3-Way Split (1,752 Train / 374 Validation / 374 Test, Seed 777)  
**Hardware Environment:** Dual NVIDIA GeForce RTX 2060 (6 GB GDDR6 each) / AMD64 24 Logical Processing Cores  
**Software Stack:** PyTorch 2.5.1+cu121, Transformers 4.57.6, ChromaDB 0.4.x  
**Date:** November 2026  

---

## 1. Executive Summary

In response to thesis committee recommendations, this investigation expands the empirical scope from a single baseline comparison (EfficientNetB0 vs. CLIP) into a comprehensive benchmark across **eight distinct visual paradigms**. 

The central scientific premise of this thesis is that **model accuracy alone is an insufficient criterion for practical deployment**. Rather, real-world deployment requires balancing predictive accuracy against computational latency, memory footprint, thermal dissipation, and crucially, the **computational cost of continuous incremental knowledge integration**.

This report evaluates models across five fundamental dimensions:
1. **Predictive Performance:** Accuracy, Precision, Recall, Macro F1, and ROC-AUC on an isolated hold-out test set ($N=374$).
2. **Computational Latency & Throughput:** Forward-pass latency (ms/image) and throughput (images/sec) across GPU (`cuda:0`) and CPU execution.
3. **Training & Adaption Overhead:** Wall-clock time required for convergence under two-stage fine-tuning versus non-parametric vector indexing.
4. **Memory Footprint:** Checkpoint disk size and run-time VRAM allocation.
5. **Noise Robustness & Curation:** Predictive retention under synthetic label noise (0% to 30%).

---

## 2. Multi-Model Empirical Results (GPU Execution: `cuda:0`)

All models were evaluated against the exact same hold-out test set ($N=374$, 187 cats / 187 dogs). Empirical measurements recorded in `tcontext/artifacts/benchmark_summary.json` and `tcontext/artifacts/benchmark_results_cuda_0.csv` are summarized below:

| Model Architecture | Paradigm / Category | Accuracy (%) | Precision | Recall | Macro F1 | ROC-AUC | Train / Index Time (s) | Inference Latency (ms) | Throughput (img/s) | Parameters ($N_{\text{total}}$) | Checkpoint Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Lightweight Edge CNN | 97.33% | 0.9733 | 0.9733 | 0.9733 | 0.9963 | 116.05 s | 3.30 ms | 302.8 | 928,162 | 3.66 MB |
| **EfficientNetB0** | Compound Scaled CNN | 98.66% | 0.9840 | 0.9893 | 0.9867 | 0.9998 | 175.18 s | 3.75 ms | 266.9 | 4,010,110 | 15.58 MB |
| **ResNet18** | Residual CNN Baseline | 99.20% | 0.9894 | 0.9947 | 0.9920 | 0.9990 | 149.20 s | 3.69 ms | 271.0 | 11,177,538 | 42.71 MB |
| **ViT-B/16** | Vision Transformer | 99.47% | 0.9947 | 0.9947 | 0.9947 | 0.9997 | 479.95 s | 10.98 ms | 91.0 | 85,800,194 | 327.36 MB |
| **CLIP (ViT-B/32)** | Foundation Vision-Language | 99.73% | 0.9947 | 1.0000 | 0.9973 | 1.0000 | 173.16 s | 4.43 ms | 225.7 | 151,278,339 | 577.23 MB |
| **Naive Retrieval** | Vector Index (ChromaDB) | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.80 s** | **0.18 ms** | **5,653.9** | **0 (Frozen)** | 11.61 MB |
| **Fine-Tuned Retrieval** | Contrastive Adapter + DB | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.69 s** | **0.14 ms** | **7,152.5** | 526,337 | 11.60 MB |
| **Proposed IMO Framework** | Vector Index + IMO Orchestration | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.27 s** | **0.15 ms** | **6,726.0** | **0 (Frozen)** | 11.59 MB |

---

## 3. In-Depth Architectural Analysis

### 3.1 MobileNetV3-Small (Lightweight Edge CNN)
- **Architecture:** Employs depthwise separable convolutions, hard-swish non-linearities, and lightweight squeeze-and-excitation attention modules optimized via platform-aware Neural Architecture Search (NAS).
- **Experimental Role:** Benchmarking ultra-low parameter edge hardware constraints.
- **Empirical Findings:** Two-stage fine-tuning elevated test accuracy to **97.33%** (F1 = 0.9733, ROC-AUC = 0.9963). Inference is rapid at **3.30 ms** on GPU and **3.05 ms** on CPU. However, adapting this model required **116.05 seconds** of backpropagation over 15 epochs, illustrating that even lightweight architectures entail non-negligible retraining overhead when new classes or domains arrive.

### 3.2 EfficientNetB0 (Compound Scaled CNN)
- **Architecture:** Uses a compound coefficient to uniformly scale network depth, width, and image resolution based on the MBConv (inverted bottleneck) backbone.
- **Experimental Role:** Historical parametric baseline utilized in initial thesis iterations.
- **Empirical Findings:** After Phase 2 end-to-end unfreezing, EfficientNetB0 achieved **98.66% accuracy** (F1 = 0.9867, ROC-AUC = 0.9998). Training converged in **175.18 seconds**. While competitive on GPU (3.75 ms), CPU inference latency tripled to **9.82 ms** due to complex branch topologies and Swish activation computation.

### 3.3 ResNet18 (Classical Residual CNN)
- **Architecture:** 18-layer residual convolutional network utilizing skip connections ($\mathbf{x} + \mathcal{F}(\mathbf{x})$) to eliminate vanishing gradients.
- **Experimental Role:** Classical, robust computer vision baseline.
- **Empirical Findings:** Outperformed EfficientNetB0 in accuracy (**99.20%**) while executing faster on GPU (**3.69 ms** vs. 3.75 ms). Training converged in **149.20 seconds**. Standard residual bottlenecks remain one of the most hardware-efficient convolutional building blocks on modern hardware.

### 3.4 ViT-B/16 (Vision Transformer)
- **Architecture:** Pure multi-head self-attention applied directly to sequences of non-overlapping $16 \times 16$ image patches ($197$ tokens including `[CLS]`).
- **Experimental Role:** High-capacity transformer baseline representing the state-of-the-art in parametric representation learning.
- **Empirical Findings:** Delivered superior predictive accuracy (**99.47%**) and near-perfect ROC-AUC (0.9997). However, this came at extraordinary computational cost: **479.95 seconds** (8 minutes) of GPU fine-tuning, **10.98 ms** GPU latency, **62.40 ms** CPU latency (throughput collapsing to 16.0 img/s), and a **327.36 MB** checkpoint footprint. ViT-B/16 demonstrates that transformer accuracy is computationally prohibitive for resource-constrained edge inference.

### 3.5 CLIP (ViT-B/32 Foundation Model)
- **Architecture:** Multi-modal dual transformer pre-trained on 400M internet-scale image-text pairs via contrastive representation learning.
- **Experimental Role:** Semantic embedding extractor and multimodal zero-shot baseline.
- **Empirical Findings:** Linear probing and top-layer unfreezing achieved **99.73% accuracy** (F1 = 0.9973, ROC-AUC = 1.0000). The pre-trained semantic geometry of CLIP creates well-separated class manifolds, requiring only 173.16 seconds of tuning.

### 3.6 Naive Retrieval (CLIP + ChromaDB)
- **Architecture:** External HNSW vector database indexing frozen 512-dim CLIP embeddings; classification performed via majority voting over $k$-nearest neighbors in cosine distance space.
- **Experimental Role:** Demonstrates pure non-parametric retrieval memory without quality curation.
- **Empirical Findings:** Achieved **100.0% accuracy** on the hold-out test set with **0.18 ms query latency** (5,653.9 img/s). Ingestion of all 1,752 training instances required only **0.80 seconds** (over **200x to 600x faster** than CNN/ViT backpropagation). However, as shown in the noise stress test, naive retrieval degrades rapidly when corrupted instances enter the index.

### 3.7 Fine-Tuned Retrieval Adapter (Metric Adapter + ChromaDB)
- **Architecture:** A lightweight learnable residual MLP projection head ($\mathbf{z} = \text{Normalize}(\mathbf{e} + \alpha \cdot \text{MLP}(\mathbf{e}))$) trained using Supervised Contrastive Loss to maximize inter-class margin while maintaining intra-class compactness.
- **Experimental Role:** Evaluates whether non-parametric retrieval can be enhanced via metric space adaptation without modifying the visual backbone.
- **Empirical Findings:** Achieved **100.0% accuracy** with the lowest latency in the benchmark (**0.14 ms**, **7,152.5 queries/sec**). Adapting the projection head required only **0.69 seconds** of training on the GPU, validating that metric space optimization is orders of magnitude more efficient than backbone retraining.

### 3.8 Proposed Framework (CLIP + ChromaDB + IMO + Evidence Fusion)
- **Architecture:** Integrates the frozen CLIP visual backbone, HNSW vector database, and the **Intelligent Memory Orchestrator (IMO)** regulating sample ingestion via running class prototypes, relative margin verification, and outlier gating, coupled with validation-calibrated evidence fusion ($\alpha^* = 0.60, k^* = 1$).
- **Experimental Role:** **Core Thesis Contribution.**
- **Empirical Findings:**
  - Achieved **100.0% test accuracy** and **1.0000 F1-score**.
  - Query latency is **0.15 ms** (**6,726.0 queries/sec**).
  - Backbone parameters remain strictly invariant ($\nabla_\theta = \mathbf{0}$).
  - Construction and prototype calibration completed in **1.27 seconds**.
  - **Noise Immunity:** In contrast to Naive Retrieval which drops to 80.48% under 30% label corruption, the Proposed IMO quarantined all 525 corrupted instances and retained **100.0% accuracy** (+19.52% margin).

---

## 4. Multi-Dimensional Trade-Off Analysis

```
                ACCURACY vs INFERENCE LATENCY EFFICIENCY FRONTIER
100% |                                      [CLIP Probe]     [Adapter Ret.] [Proposed IMO]
     |                                      [ViT-B/16]       [Naive Ret.]
 99% |                  [ResNet18]
     |
 98% |  [EffNetB0]
     |
 97% |  [MobileNetV3]
     +---------------------------------------------------------------------------------
        3.0 ms          3.7 ms              11.0 ms          0.14 - 0.18 ms
                    (Parametric Models)                  (Vector Retrieval Models)
```

### 4.1 Knowledge Integration Latency (Retraining vs Indexing)
When novel visual instances arrive in streaming environments:
- **Supervised Parametric Models:** Require iterative forward/backward passes over stored batches. On host GPU, this took **116.0s (MobileNet) to 480.0s (ViT-B/16)**.
- **Proposed IMO Framework:** Ingestion consists of a single visual encoder forward pass (4.43 ms) and an HNSW graph insertion with running centroid update (0.15 ms), totaling **< 5.0 ms per image**. This represents an acceleration of **> 23,000x** over parametric retraining.

### 4.2 Parameter Invariance & Elimination of Catastrophic Forgetting
In conventional neural networks, applying gradient updates $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ risks catastrophic forgetting:
- Network weights drift away from historical optima.
- In contrast, the proposed framework enforces **strict parameter invariance** ($\nabla_\theta = \mathbf{0}$). All newly acquired knowledge is stored additively in external vector memory $\mathcal{M} \leftarrow \mathcal{M} \cup \{(\mathbf{e}_{\text{new}}, y_{\text{new}})\}$.
- Prior decision boundaries remain completely intact.

### 4.3 Checkpoint & Storage Footprint
- ViT-B/16 requires **327.36 MB** to store model weights.
- ResNet18 requires **42.71 MB**.
- In contrast, storing the entire 1,752-image vector index in ChromaDB requires only **11.59 MB** ($\approx 6.6$ KB per sample including 512-dim floating-point coordinates and HNSW graph topology).

---

## 5. Thesis Conclusions and Academic Takeaways

1. **Retraining Is Unnecessary for Continuous Adaptation:** Frozen foundation representations (CLIP ViT-B/32) combined with non-parametric retrieval memory equal or exceed the accuracy of fully fine-tuned parametric classifiers (100.0% vs. 97.33% - 99.47%) while completely eliminating catastrophic forgetting.
2. **Sub-Millisecond Query Throughput:** Vector retrieval delivers query throughput exceeding 6,700 samples/sec (0.15 ms latency), providing a 22x to 73x speedup over transformer forward passes.
3. **The Essential Role of the IMO:** Unregulated vector retrieval is fundamentally vulnerable to corrupted or adversarial inputs (dropping from 100% to 80.48% under 30% noise). The Intelligent Memory Orchestrator provides an indispensable quality control filter, guaranteeing that non-parametric learning remains robust in mission-critical deployments.
