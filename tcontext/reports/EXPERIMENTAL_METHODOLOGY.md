# Scientifically Rigorous Experimental Methodology

**Thesis Title:** *A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning*  
**Repository:** `RoushanKhalid/Thesis`  
**Author:** Thesis Research Candidate  
**Date:** November 2026  

---

## 1. Executive Summary & Research Problem

Conventional deep learning frameworks adapt to novel visual knowledge through gradient-based parameter fine-tuning or end-to-end retraining. While effective in static benchmark scenarios, gradient adaptation in dynamic deployment settings introduces severe scientific and practical shortcomings:
1. **Catastrophic Forgetting:** Overwriting connection weights causes catastrophic degradation of prior knowledge distributions unless costly rehearsal buffers or regularization constraints are introduced.
2. **Computational Inefficiency:** Retraining backbones requires significant computational energy, extended wall-clock latency, and thermal dissipation, rendering edge or resource-constrained deployment infeasible.
3. **Vulnerability to Label Corruption:** Standard gradient descent blindly incorporates mislabeled or noisy samples into parametric weights, polluting representations globally.

This research formulates and validates an alternative paradigm: **A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning**. By freezing high-capacity semantic visual representations and delegating knowledge retention to an external vector memory curated by an **Intelligent Memory Orchestrator (IMO)**, the framework achieves sub-millisecond knowledge updates ($< 0.20$ ms), completely invariant backbone parameters, and robust rejection of noisy or malicious data.

---

## 2. Dataset Specification & Partitioning Protocol

### 2.1 Microsoft Cats vs. Dogs Benchmark
To preserve scientific comparability with historical thesis baselines while eliminating previous data leakage vulnerabilities, experiments utilize the standardized Microsoft Cats vs. Dogs vision benchmark:
- **Total Dataset Size:** 1,250 high-resolution visual images.
- **Class Balance:** Exact 50/50 balance (625 Cat samples, 625 Dog samples).
- **Format:** RGB, variable resolutions normalized during preprocessing.

### 2.2 Deterministic 3-Way Stratified Partitioning
Prior implementations suffered from validation leakage (evaluating final accuracy on validation data used for early stopping). We enforce a strict, reproducible three-way stratified split:

$$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

| Partition | Percentage | Sample Count | Cats Count | Dogs Count | Role in Experimental Protocol |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Set ($\mathcal{D}_{\text{train}}$)** | 70% | 878 | 439 | 439 | Baseline weight adaptation; vector memory indexing pool |
| **Validation Set ($\mathcal{D}_{\text{val}}$)** | 15% | 186 | 93 | 93 | Learning rate scheduling, early stopping, threshold tuning |
| **Test Set ($\mathcal{D}_{\text{test}}$)** | 15% | 186 | 93 | 93 | **Strict hold-out set**; evaluated identically across all models |

The partition is deterministically fixed with random seed `777` and cached in `tcontext/artifacts/stratified_split_seed_777.json`. Every model and retrieval index is evaluated against the exact same 186 hold-out test images.

### 2.3 Input Preprocessing & Augmentation Pipelines
To ensure architectural parity:
- **Supervised CNNs & ViT:** Input images are resized to $224 \times 224$ pixels, transformed to float tensors, and normalized using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). Training partitions employ random horizontal flips and $\pm 10^\circ$ affine rotations.
- **CLIP Visual Encoders & Retrieval:** Input images are processed via `CLIPProcessor` (bicubic interpolation to $224 \times 224$, channel rescaling to $[0, 1]$, and standardization with OpenAI CLIP statistics).

---

## 3. Evaluated Model Suites & Architectural Roles

To address board recommendations for architectural diversity, seven representative paradigms were selected:

| Model Architecture | Architectural Category | Parameters | Embedding Dim | Adaptation Paradigm | Thesis Role |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **MobileNetV3-Small** | Lightweight Edge CNN | 928,162 | N/A | Linear Probe on Frozen Features | Edge/Mobile Baseline |
| **EfficientNetB0** | Compound Scaled CNN | 4,010,110 | N/A | Linear Probe on Frozen Features | Original Parametric Baseline |
| **ResNet18** | Residual CNN | 11,177,538 | N/A | Linear Probe on Frozen Features | Classical Standard Baseline |
| **ViT-B/16** | Vision Transformer | 85,800,194 | 768 | Linear Probe on Frozen Features | High-Capacity Attention Baseline |
| **CLIP (ViT-B/32)** | Vision-Language Foundation | 151,278,339 | 512 | Frozen Zero-Shot & Linear Probe | Multi-Modal Semantic Baseline |
| **Naive Retrieval** | Vector Index (ChromaDB) | 0 | 512 | Unfiltered Embedding Insertion | Unregulated Memory Baseline |
| **Proposed Framework** | Vector Index + IMO Orchestrator | 0 | 512 | Gated Prototype Insertion + Evidence Fusion | **Proposed Thesis Contribution** |

---

## 4. Mathematical Formulation of the Proposed Framework

The proposed framework integrates two complementary components: an **External Semantic Vector Memory** and an **Intelligent Memory Orchestrator (IMO)**, coupled with a **Dynamic Evidence Fusion Engine**.

```
                +------------------------------------+
                |  Input Query Image x              |
                +-----------------+------------------+
                                  |
                +-----------------v------------------+
                | Frozen CLIP Visual Encoder (f_θ)   |
                +-----------------+------------------+
                                  | e = f_θ(x) / ||f_θ(x)||
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
| Parametric Classifier |                   | External Vector Memory|
| P_clf(y | x)          |                   | ChromaDB (HNSW Cosine)|
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            |                                           | Top-k Query
            |                                           v
            |                               +-----------------------+
            |                               | Intelligent Memory    |
            |                               | Orchestrator (IMO)    |
            |                               | P_ret(y | x)          |
            |                               +-----------+-----------+
            |                                           |
            +---------------------+---------------------+
                                  |
                                  v
                    +---------------------------+
                    | Dynamic Evidence Fusion   |
                    | P(y|x) = α P_clf + (1-α) P_ret
                    +-------------+-------------+
                                  |
                                  v
                        Final Prediction y*
```

### 4.1 External Vector Memory
Given an image $\mathbf{x}$, a frozen CLIP visual backbone $f_\theta$ extracts a $d$-dimensional normalized semantic embedding:

$$\mathbf{e} = \frac{f_\theta(\mathbf{x})}{\|f_\theta(\mathbf{x})\|_2} \in \mathbb{R}^d \quad (d = 512)$$

Embeddings are indexed into an external vector database $\mathcal{M} = \{(\mathbf{e}_i, y_i, \text{meta}_i)\}_{i=1}^N$ structured with Hierarchical Navigable Small World (HNSW) graphs under cosine distance:

$$D_{\text{cos}}(\mathbf{e}_q, \mathbf{e}_i) = 1 - \mathbf{e}_q^\top \mathbf{e}_i$$

### 4.2 Intelligent Memory Orchestrator (IMO)
To prevent memory corruption, every candidate sample $(\mathbf{e}_t, y_t)$ presented for incremental insertion must pass three verification stages before commitment:

#### Stage 1: Running Normalized Class Prototypes
For each class $k \in \mathcal{C}$, the IMO maintains a running class prototype $\mathbf{p}_k$. When an accepted sample $(\mathbf{e}_t, y_t=k)$ is committed, the prototype updates incrementally:

$$\mathbf{p}_k^{(t)} = \frac{n \mathbf{p}_k^{(t-1)} + \mathbf{e}_t}{\|n \mathbf{p}_k^{(t-1)} + \mathbf{e}_t\|_2}, \quad n \leftarrow n + 1$$

#### Stage 2: Prototype Consistency & Relative Margin Gating
A candidate is checked for both absolute similarity and relative class margin:

$$\text{Gate}_1: \quad \mathbf{e}_t^\top \mathbf{p}_{y_t} \ge \tau_c \quad (\tau_c = 0.65)$$

$$\text{Gate}_2: \quad \mathbf{e}_t^\top \mathbf{p}_{y_t} - \max_{j \ne y_t} \left( \mathbf{e}_t^\top \mathbf{p}_j \right) \ge \tau_m \quad (\tau_m = 0.00)$$

If a sample claimed to be class $k$ exhibits greater cosine similarity to a competing prototype $\mathbf{p}_j$ ($j \ne k$), it is identified as a corrupted or ambiguous candidate.

#### Stage 3: Nearest Neighbor Conflict & Outlier Filtering
Using a local neighborhood query $\mathcal{N}_k(\mathbf{e}_t)$ of size $K=5$:
- **Majority Neighborhood Conflict:** If the majority vote of neighboring stored vectors conflicts with candidate label $y_t$, the candidate is rejected.
- **Outlier Density Gate:** If the mean cosine distance $\bar{D} = \frac{1}{K} \sum_{i \in \mathcal{N}_k} D_{\text{cos}}(\mathbf{e}_t, \mathbf{e}_i) > \tau_{\text{outlier}}$ ($\tau_{\text{outlier}} = 0.45$), the candidate is treated as an out-of-distribution outlier.

#### Non-Destructive Quarantine Buffer
Candidates failing any gate are not discarded; they are logged into an isolated Quarantine Buffer $\mathcal{Q}$:

$$\mathcal{Q} = \left\{ (\mathbf{e}_t, y_t, \text{reason}, \text{timestamp}, \text{metadata}) \right\}$$

This preserves data for downstream active learning or human expert audit without polluting active inference memory.

### 4.3 Dynamic Evidence Fusion Engine
During inference, predictions combine the parametric posterior $P_{\text{clf}}(y=k \mid \mathbf{x})$ and non-parametric retrieval posterior $P_{\text{ret}}(y=k \mid \mathbf{x})$:

$$P(y = k \mid \mathbf{x}) = \alpha P_{\text{clf}}(y = k \mid \mathbf{x}) + (1 - \alpha) P_{\text{ret}}(y = k \mid \mathbf{x})$$

where $\alpha \in [0, 1]$ is a tunable fusion parameter:
- $\alpha = 1.0$: Pure parametric deep learning.
- $\alpha = 0.0$: Pure non-parametric retrieval memory.
- $\alpha = 0.5$: Balanced multi-modal consensus.

---

## 5. Comprehensive Metrics Formulation

### 5.1 Predictive Performance Metrics
All metrics are evaluated on the identical hold-out test set $\mathcal{D}_{\text{test}}$ ($N=186$):
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall:** $\frac{TP}{TP + FN}$
- **Macro F1-Score:** $\frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve computed over continuous probability scores.

### 5.2 Computational & Hardware Profiling Metrics
- **Parameter Counts:** Total parameters ($N_{\text{total}}$) and trainable parameters ($N_{\text{trainable}}$).
- **Model Checkpoint Size:** Binary size on disk (MB).
- **Training Time:** Elapsed wall-clock seconds to complete training and early stopping.
- **Inference Latency:** Measured over 20 warmup cycles followed by 186 hold-out test evaluations. Reported as Mean, Median, and 95th Percentile latency in milliseconds per image (ms/img).
- **Throughput:** Processed samples per second ($\frac{N_{\text{test}}}{\text{Total Inference Seconds}}$).
- **Peak Hardware Consumption:**
  - Peak VRAM allocated on GPU via `torch.cuda.max_memory_allocated()`.
  - Peak RAM footprint via Windows native Win32 `GetProcessMemoryInfo` API.

---

## 6. Controlled Hardware Profiling Protocol

To evaluate real-world edge feasibility, every model is benchmarked under two isolated execution environments:
1. **GPU-Accelerated Mode:** Primary execution on `NVIDIA GeForce RTX 2060` (6 GB dedicated GDDR6 VRAM, CUDA 12.1).
2. **CPU-Only Mode:** Secondary execution constrained strictly to host processor (`AMD64 Family 25 Model 33 Stepping 0`, 24 logical cores) by explicitly routing tensors and module graphs to `torch.device("cpu")`.

Speedup factors are computed as:

$$\text{Speedup} = \frac{\text{Latency}_{\text{CPU}}}{\text{Latency}_{\text{GPU}}}$$

---

## 7. Noise Robustness Verification Protocol (Hypothesis H1)

**Hypothesis H1:** *Under increasing label corruption, the Intelligent Memory Orchestrator (IMO) preserves high predictive accuracy by routing corrupted samples into the quarantine buffer, whereas naive retrieval suffers rapid accuracy degradation.*

To empirically evaluate H1:
1. Four synthetic label noise rates are injected into the training pool: $r \in \{0.00, 0.10, 0.20, 0.30\}$.
2. Binary class labels are inverted uniformly at rate $r$ with fixed seed `777`.
3. Two parallel systems are instantiated under identical conditions:
   - **System A (Naive Retrieval):** Ingests all $N$ training samples indiscriminately into ChromaDB.
   - **System B (Proposed IMO Framework):** Filters candidates through prototype consistency and margin gates before insertion.
4. Both systems are evaluated against the clean, uncorrupted test partition $\mathcal{D}_{\text{test}}$.
5. Accuracy retention and quarantine counts are recorded and analyzed.
