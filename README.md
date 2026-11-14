# A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1+cu121](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

**Thesis Project Repository**  
GitHub: [`RoushanKhalid/Thesis`](https://github.com/RoushanKhalid/Thesis)  
Domain: Multi-Modal Incremental Learning, Non-Parametric Memory, Resource-Constrained Machine Learning.

---

## 📖 Research Abstract & Overview

Adapting deep neural networks to continuous streams of visual knowledge typically requires iterative backpropagation and fine-tuning. This paradigm introduces three critical scientific and operational hurdles:
1. **Catastrophic Forgetting:** Weight updates degrade previously learned decision boundaries unless computationally expensive replay buffers or regularization constraints are introduced.
2. **Computational Inefficiency:** Retraining backbones consumes substantial energy, memory, and wall-clock time, creating an insurmountable bottleneck for edge and resource-constrained devices.
3. **Vulnerability to Label Noise:** Standard gradient descent indiscriminately absorbs mislabeled and adversarial inputs into network weights.

This thesis presents **A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning**. By freezing a high-capacity semantic visual encoder (OpenAI CLIP ViT-B/32) and coupling it with an external vector database (ChromaDB) regulated by an **Intelligent Memory Orchestrator (IMO)**, the framework achieves:
- **Sub-Millisecond Knowledge Integration (< 0.20 ms):** New visual instances are ingested instantly via HNSW index insertions without a single backward pass.
- **Strict Parameter Invariance:** Model backbone parameters remain completely frozen ($\nabla_\theta = \mathbf{0}$), eliminating catastrophic forgetting by design.
- **Noise Robustness:** Prototype consistency and relative class margin gating actively filter corrupted samples into an isolated Quarantine Buffer, preserving **100.0% accuracy** under 30% label corruption (+19.52% higher than naive retrieval).

---

## 🏆 Multi-Model Benchmark Results

Evaluated on the full **Microsoft Cats vs. Dogs** benchmark ($N=2,500$ verified images) under a strict, leak-free **3-way stratified partition** (70% Train: 1,752, 15% Validation: 374, 15% Hold-Out Test: 374) with fixed seed `777`.

### Primary GPU Performance Leaderboard (`cuda:0` - NVIDIA GeForce RTX 2060)

| Model Architecture | Category | Accuracy (%) | Precision | Recall | Macro F1 | ROC-AUC | Train / Index Time | Inference Latency | Throughput (img/s) | Parameters ($N_{\text{total}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Edge CNN (Fine-Tuned) | 97.33% | 0.9733 | 0.9733 | 0.9733 | 0.9963 | 116.05 s | 3.30 ms | 302.8 | 928,162 |
| **EfficientNetB0** | Compound CNN (Fine-Tuned) | 98.66% | 0.9840 | 0.9893 | 0.9867 | 0.9998 | 175.18 s | 3.75 ms | 266.9 | 4,010,110 |
| **ResNet18** | Residual CNN (Fine-Tuned) | 99.20% | 0.9894 | 0.9947 | 0.9920 | 0.9990 | 149.20 s | 3.69 ms | 271.0 | 11,177,538 |
| **ViT-B/16** | Vision Transformer (Fine-Tuned) | 99.47% | 0.9947 | 0.9947 | 0.9947 | 0.9997 | 479.95 s | 10.98 ms | 91.0 | 85,800,194 |
| **CLIP (ViT-B/32)** | Foundation Vision-Language | 99.73% | 0.9947 | 1.0000 | 0.9973 | 1.0000 | 173.16 s | 4.43 ms | 225.7 | 151,278,339 |
| **Naive Retrieval** | Vector Index (ChromaDB) | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.80 s** | **0.18 ms** | **5,653.9** | **0 (Frozen)** |
| **Fine-Tuned Retrieval** | Contrastive Metric Adapter + ChromaDB | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.69 s** | **0.14 ms** | **7,152.5** | 526,337 |
| **Proposed IMO Framework** | Vector Index + IMO Orchestrator | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.27 s** | **0.15 ms** | **6,726.0** | **0 (Frozen)** |

---

## ⚡ Hardware Profiling: GPU vs. CPU Constraints

Experiments systematically profile resource utilization across host CPU (24 logical threads) and dedicated GPU acceleration (NVIDIA RTX 2060 6GB VRAM):

| Model Architecture | GPU Latency | CPU Latency | Hardware Speedup ($\times$) | Peak GPU VRAM | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 3.30 ms | 3.05 ms | **0.92 $\times$** | 103.96 MB | 3.66 MB |
| **EfficientNetB0** | 3.75 ms | 9.82 ms | **2.62 $\times$** | 394.38 MB | 15.58 MB |
| **ResNet18** | 3.69 ms | 10.07 ms | **2.73 $\times$** | 370.00 MB | 42.71 MB |
| **ViT-B/16** | 10.98 ms | 62.40 ms | **5.68 $\times$** | 1,066.48 MB | 327.36 MB |
| **CLIP (ViT-B/32)** | 4.43 ms | 17.40 ms | **3.93 $\times$** | 1,541.63 MB | 577.23 MB |
| **Fine-Tuned Retrieval** | **0.14 ms** | **0.14 ms** | **1.00 $\times$** | **0.0 MB** | **11.60 MB** |
| **Proposed IMO Framework** | **0.15 ms** | **0.15 ms** | **1.00 $\times$** | **0.0 MB** | **11.59 MB** |

### Key Hardware Insights:
- **ViT-B/16 shows heavy GPU dependency (5.68x speedup):** Dense self-attention matrix operations cause CPU latency to jump to 62.40 ms, making pure Vision Transformers impractical for real-time edge CPU deployment.
- **MobileNetV3-Small is CPU-friendly (0.92x speedup):** Depthwise convolutions are memory-bandwidth bound at batch size 1; host CPU vector units handle it at 3.05 ms without GPU launch overhead.
- **Vector Retrieval is CPU/GPU Invariant (0.14 - 0.18 ms):** HNSW graph traversal operates via CPU cache-accelerated pointer chasing, yielding over 6,700 queries/sec with zero GPU VRAM consumption.

---

## 🛡️ Noise Robustness Benchmark (Hypothesis H1)

To evaluate memory integrity under adversarial or erroneous data entry, synthetic label noise was injected into the training pool (1,752 images) at rates of 0%, 10%, 20%, and 30%:

| Injected Noise Rate | Naive Retrieval Accuracy | Proposed IMO Accuracy | IMO Quarantined Samples | Accuracy Retention Margin |
| :---: | :---: | :---: | :---: | :---: |
| **0% (Clean)** | 100.00% | 100.00% | 0 | +0.00% |
| **10% Noise** | 98.40% | 100.00% | 175 | **+1.60%** |
| **20% Noise** | 91.98% | 100.00% | 350 | **+8.02%** |
| **30% Noise** | 80.48% | 100.00% | 525 | **+19.52%** |

> **Hypothesis H1 Validated:** Under 30% label corruption, naive retrieval collapses by -19.52% (from 100% down to 80.48%), whereas the Proposed IMO Framework intercepts and quarantines exactly all 525 corrupted candidates, maintaining **100.00% test accuracy** on the hold-out test set.

---

## 🗂️ Modular Repository Architecture

```
Thesis/
├── configs/
│   ├── experiment_config.yaml       # Hyperparameters, random seed, batch sizes
│   ├── hardware_config.yaml         # Device placement, precision, threads
│   └── models.yaml                  # Model architecture registry & weights
├── tcontext/
│   ├── artifacts/                   # Benchmark CSVs, summary JSONs, generated plots
│   │   ├── benchmark_results_cpu.csv
│   │   ├── benchmark_results_cuda_0.csv
│   │   ├── benchmark_summary.json
│   │   ├── cpu_vs_gpu_speedup.png
│   │   ├── latency_vs_accuracy_tradeoff.png
│   │   ├── multi_model_accuracy_comparison.png
│   │   ├── noise_experiment_results.json
│   │   └── noise_robustness_curve.png
│   ├── data/
│   │   ├── dataset_loader.py        # 3-way deterministic stratified partitioner
│   │   └── full_dataset/            # 2,500 validated balanced benchmark images
│   ├── evaluation/
│   │   ├── metrics.py               # Classification metrics (Acc, F1, ROC-AUC)
│   │   ├── noise_benchmarks.py      # Synthetic label noise experiment runner
│   │   └── resource_tracker.py      # Win32 RAM, CUDA VRAM, and Latency profiler
│   ├── experiments/
│   │   └── run_benchmark.py         # End-to-end multi-model benchmark runner
│   ├── models/
│   │   ├── base_model.py            # Abstract BaseClassifier interface & 2-stage fine-tuning
│   │   ├── cnn_models.py            # MobileNetV3-Small, EfficientNetB0, ResNet18
│   │   ├── clip_model.py            # CLIP visual backbone & linear probe
│   │   └── vit_model.py             # Vision Transformer (ViT-B/16)
│   ├── reports/
│   │   ├── END_TO_END_AUDIT.md      # Initial comprehensive audit report
│   │   ├── EXPERIMENTAL_METHODOLOGY.md # Formal scientific protocol
│   │   ├── MODEL_COMPARISON.md      # Multi-model evaluation analysis
│   │   ├── HARDWARE_BENCHMARK.md    # CPU vs GPU hardware profiling report
│   │   └── IMPLEMENTATION_CHANGELOG.md # Detailed audit and modernization log
│   ├── retrieval/
│   │   ├── imo.py                   # Intelligent Memory Orchestrator
│   │   ├── retrieval_engine.py      # Retrieval, Contrastive Adapter & Evidence Fusion pipeline
│   │   └── vector_store.py          # ChromaDB persistent client wrapper
│   ├── visualization/
│   │   └── plot_results.py          # Publication-ready matplotlib/seaborn figures
│   └── web_app.py                   # Modernized Streamlit interactive dashboard
├── launch_app.ps1                   # One-click Streamlit launcher
├── run_demo.ps1                     # Full benchmark pipeline & demo launcher
└── README.md                        # Master thesis documentation
```

---

## 🚀 Quickstart & Execution Guide

### Prerequisites
- Windows 10/11 (x64)
- Python 3.11 with CUDA 12.1 enabled
- Recommended: Dedicated NVIDIA GPU (minimum 4 GB VRAM)

### 1. One-Click Dashboard Launch (Recommended)
Launch the interactive Streamlit dashboard on `http://localhost:8501`:
```powershell
.\launch_app.ps1
```

### 2. Run the Full Multi-Model Benchmark
To re-run the entire experimental benchmark across all 8 models on both GPU and CPU:
```powershell
.\run_demo.ps1 -Seed 777 -ForceRebuild
```
Or directly via Python:
```powershell
python tcontext\experiments\run_benchmark.py --seed 777 --epochs 15
```

### 3. Interactive Web Dashboard Features
The Streamlit application (`tcontext/web_app.py`) provides 6 interactive modules:
1. **🏆 Multi-Model Leaderboard:** Side-by-side metric tables, efficiency frontier scatter plots, and ROC curves.
2. **⚡ Hardware Profiling:** Real-time CPU vs. GPU latency, throughput, and memory consumption comparisons.
3. **🧠 Retrieval & IMO Architecture:** Prototype consistency and relative class margin gating visualization.
4. **🛡️ Noise Robustness:** Interactive degradation curves under synthetic label corruption.
5. **🔍 Live Inference & Memory Demo:** Test arbitrary images across multiple models and add new instances into memory with live IMO quality gating.
6. **💻 Benchmark Commands:** Direct CLI references and execution tools.

---

## 📚 Detailed Thesis Reports

For exhaustive mathematical formulations, hardware logs, and empirical breakdowns, refer to:
- [`tcontext/reports/END_TO_END_AUDIT.md`](tcontext/reports/END_TO_END_AUDIT.md): Initial repository audit and vulnerability identification.
- [`tcontext/reports/EXPERIMENTAL_METHODOLOGY.md`](tcontext/reports/EXPERIMENTAL_METHODOLOGY.md): Mathematical formulations, dataset partitioning, and evaluation protocol.
- [`tcontext/reports/MODEL_COMPARISON.md`](tcontext/reports/MODEL_COMPARISON.md): In-depth comparison across the 8 visual paradigms.
- [`tcontext/reports/HARDWARE_BENCHMARK.md`](tcontext/reports/HARDWARE_BENCHMARK.md): Hardware profiling analysis under CPU and GPU constraints.
- [`tcontext/reports/IMPLEMENTATION_CHANGELOG.md`](tcontext/reports/IMPLEMENTATION_CHANGELOG.md): Complete changelog of refactoring and modernization.
