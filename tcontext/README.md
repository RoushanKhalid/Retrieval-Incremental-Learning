# `tcontext` — Multi-Modal Incremental Learning Framework

Internal architecture and module reference for the master's thesis research project:
**"A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning"**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1+cu121](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

---

## 🏛️ System Architecture

The framework decouples visual representation learning from incremental knowledge adaptation:
1. **Parametric Foundation Backbones:** Pre-trained and fine-tuned feature extractors (MobileNetV3-Small, EfficientNetB0, ResNet18, ViT-B/16, and CLIP ViT-B/32).
2. **Metric Retrieval Adapter:** Learnable contrastive projection head ($\mathbb{R}^{512} \to \mathbb{R}^{512}$) optimizing semantic cluster compactness without altering visual encoder weights.
3. **External Vector Memory:** Persistent ChromaDB instance operating in cosine similarity space via Hierarchical Navigable Small World (HNSW) graphs.
4. **Intelligent Memory Orchestrator (IMO):** Automated continuous quality guardian enforcing:
   - **Prototype Consistency Gating:** Ensures candidate embeddings correlate with running class centroids ($\cos(\mathbf{e}, \mathbf{p}_c) \ge \tau_c$).
   - **Relative Margin Verification:** Verifies positive class proximity exceeds nearest negative prototype ($\cos(\mathbf{e}, \mathbf{p}_c) - \cos(\mathbf{e}, \mathbf{p}_{c'}) \ge \tau_m$).
   - **Quarantine Buffer:** Isolates suspicious, mislabeled, or out-of-distribution instances without polluting the active index.
   - **Dynamic Evidence Fusion:** Combines parametric base probability posteriors with non-parametric retrieval evidence via temperature-scaled Bayesian softmax weighting.

```
                            +-----------------------------------+
                            |        Input Query Image          |
                            +-----------------+-----------------+
                                              |
                            +-----------------v-----------------+
                            | Frozen Semantic Visual Backbone   |
                            |       (CLIP ViT-B/32)             |
                            +-----------------+-----------------+
                                              | e ∈ R^512
                            +-----------------v-----------------+
                            |     Metric Retrieval Adapter      |
                            |   (Supervised Contrastive MLP)    |
                            +-----------------+-----------------+
                                              | z = Normalize(e + W_a e)
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
    +---------------------------------+               +---------------------------------+
    |    Parametric Base Classifier   |               |     Non-Parametric Memory       |
    |        (e.g., EfficientNetB0)   |               |     (ChromaDB HNSW Graph)       |
    +----------------+----------------+               +----------------+----------------+
                     | P_clf(y|x)                                      | Top-k Neighbors
                     |                                                 v
                     |                                +---------------------------------+
                     |                                | Intelligent Memory Orchestrator |
                     |                                | (Prototype & Margin Gating)     |
                     |                                +----------------+----------------+
                     |                                                 | P_ret(y|x)
                     +------------------------+------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |      Dynamic Evidence Fusion      |
                            |  P(y|x) = α P_clf + (1-α) P_ret   |
                            +-----------------------------------+
```

---

## 📁 Directory Structure & Module Reference

```
tcontext/
├── artifacts/                  # Generated benchmark CSVs, JSON summaries, and plots
│   ├── benchmark_results_cpu.csv
│   ├── benchmark_results_cuda_0.csv
│   ├── benchmark_summary.json
│   ├── metric_retrieval_adapter.pt
│   ├── noise_experiment_results.json
│   └── *.png
├── data/                       # Data ingestion and leak-free partitioning
│   ├── dataset_loader.py       # Deterministic 3-way stratified partitioner (70/15/15)
│   ├── full_dataset/           # 2,500 validated images (1,250 cats, 1,250 dogs)
│   └── hf_parquet/             # HuggingFace parquet mirror archives
├── evaluation/                 # Metrics, noise testing, and hardware profilers
│   ├── metrics.py              # Classification metrics (Acc, Precision, Recall, F1, ROC-AUC)
│   ├── noise_benchmarks.py     # Synthetic label corruption stress testing (0% to 30%)
│   └── resource_tracker.py     # Hardware metrics (RAM, VRAM, latency, throughput)
├── experiments/                # Unified experimental runners
│   └── run_benchmark.py        # Master multi-model benchmark script
├── models/                     # Deep learning classifier suite
│   ├── base_model.py           # Abstract BaseClassifier with 2-stage fine-tuning
│   ├── cnn_models.py           # MobileNetV3-Small, EfficientNetB0, ResNet18
│   ├── clip_model.py           # CLIP ViT-B/32 vision encoder & linear probe
│   └── vit_model.py            # Vision Transformer ViT-B/16
├── reports/                    # Formal thesis reports and audit logs
│   ├── END_TO_END_AUDIT.md     # Comprehensive repository audit
│   ├── EXPERIMENTAL_METHODOLOGY.md # Formal scientific protocol & equations
│   ├── HARDWARE_BENCHMARK.md   # CPU vs. GPU profiling report
│   ├── IMPLEMENTATION_CHANGELOG.md # Engineering changelog
│   └── MODEL_COMPARISON.md     # Multi-model evaluation analysis
├── retrieval/                  # External memory & orchestration
│   ├── imo.py                  # Intelligent Memory Orchestrator
│   ├── retrieval_engine.py     # RetrievalFramework & MetricRetrievalAdapter
│   └── vector_store.py         # ChromaVectorStore wrapper
├── vector_db/                  # Local persistent ChromaDB directories
├── visualization/              # Publication graphics generator
│   └── plot_results.py         # Matplotlib / Seaborn visualization routines
└── web_app.py                  # Interactive 6-module Streamlit research dashboard
```

---

## 🧪 Experimental Protocol

### 1. Leak-Free 3-Way Stratified Partitioning
Data is partitioned deterministically using seed `777`:
- **Train Split (70%, 1,752 samples):** Used exclusively for linear probing, backbone fine-tuning, and vector memory index construction.
- **Validation Split (15%, 374 samples):** Used for learning rate scheduling, early stopping checkpoint selection, and retrieval hyperparameter calibration ($k^*, \alpha^*$).
- **Test Split (15%, 374 samples):** Completely held out. Evaluated identically across all 8 architectural paradigms under equivalent conditions.

### 2. Two-Stage Fine-Tuning
Parametric models follow a disciplined two-stage training curriculum:
- **Phase 1 (Linear Probe Warmup):** The backbone remains frozen while the classification head is trained with learning rate $\eta_{\text{head}} = 3 \times 10^{-4}$ for 4-5 epochs.
- **Phase 2 (End-to-End Fine-Tuning):** The backbone is unfrozen and trained with discriminative learning rate $\eta_{\text{backbone}} = 1 \times 10^{-5}$ and Cosine Annealing schedule with early stopping.

### 3. Metric Retrieval Adaptation
To tailor general-purpose CLIP embeddings to the target visual domain without altering backbone weights:
- A learnable residual projection head $\mathbf{z} = \text{Normalize}(\mathbf{e} + \alpha \cdot \text{MLP}(\mathbf{e}))$ is optimized using Supervised Contrastive loss on $\mathcal{D}_{\text{train}}$.
- Hyperparameters ($k^* \in \{1, 3, 5, 7, 9\}$ and fusion weight $\alpha^* \in [0.0, 1.0]$) are calibrated strictly on $\mathcal{D}_{\text{val}}$.

---

## 🚀 Execution Commands

Run full multi-model benchmark with fine-tuning on GPU:
```powershell
python tcontext\experiments\run_benchmark.py --epochs 15 --seed 777
```

Launch the interactive Streamlit dashboard:
```powershell
streamlit run tcontext\web_app.py --server.port 8501
```
