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
- **Noise Robustness:** Prototype consistency and relative class margin gating actively filter corrupted samples into an isolated Quarantine Buffer, preserving **98.39% accuracy** under 30% label corruption (+17.74% higher than naive retrieval).

---

## 🏆 Multi-Model Benchmark Results

Evaluated on the standardized **Microsoft Cats vs. Dogs** benchmark ($N=1,250$) under a strict, leak-free **3-way stratified partition** (70% Train, 15% Validation, 15% Hold-Out Test) with fixed seed `777`.

### Primary GPU Performance Leaderboard (`cuda:0` - NVIDIA GeForce RTX 2060)

| Model Architecture | Category | Accuracy (%) | Precision | Recall | Macro F1 | ROC-AUC | Train / Index Time | Inference Latency | Throughput (img/s) | Parameters ($N_{\text{total}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Edge CNN | 94.09% | 0.9184 | 0.9677 | 0.9424 | 0.9820 | 37.85 s | 3.42 ms | 292.6 | 928,162 |
| **EfficientNetB0** | Compound CNN | 92.47% | 0.9247 | 0.9247 | 0.9247 | 0.9852 | 42.30 s | 3.70 ms | 270.2 | 4,010,110 |
| **ResNet18** | Residual CNN | 97.31% | 0.9583 | 0.9892 | 0.9735 | 0.9951 | 40.27 s | 3.61 ms | 277.1 | 11,177,538 |
| **ViT-B/16** | Vision Transformer | 98.92% | 0.9892 | 0.9892 | 0.9892 | 0.9990 | 134.40 s | 10.87 ms | 92.0 | 85,800,194 |
| **CLIP (ViT-B/32)** | Foundation Vision-Language | **99.46%** | 1.0000 | 0.9892 | **0.9946** | **0.9997** | 58.00 s | 4.75 ms | 210.5 | 151,278,339 |
| **Naive Retrieval** | Vector Index (ChromaDB) | 98.92% | 1.0000 | 0.9785 | 0.9891 | 0.9945 | **0.45 s** | **0.18 ms** | **5,465.7** | **0 (Frozen)** |
| **Proposed IMO Framework** | Vector Index + IMO Orchestrator | **99.46%** | 1.0000 | 0.9892 | **0.9946** | **0.9988** | **0.67 s** | **0.19 ms** | **5,154.3** | **0 (Frozen)** |

---

## ⚡ Hardware Profiling: GPU vs. CPU Constraints

Experiments systematically profile resource utilization across host CPU (24 logical threads) and dedicated GPU acceleration (NVIDIA RTX 2060 6GB VRAM):

| Model Architecture | GPU Latency | CPU Latency | Hardware Speedup ($\times$) | Peak GPU VRAM | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 3.42 ms | 3.66 ms | **1.07 $\times$** | 100.4 MB | 3.66 MB |
| **EfficientNetB0** | 3.70 ms | 11.51 ms | **3.11 $\times$** | 374.3 MB | 15.58 MB |
| **ResNet18** | 3.61 ms | 10.49 ms | **2.91 $\times$** | 306.3 MB | 42.71 MB |
| **ViT-B/16** | 10.87 ms | 71.17 ms | **6.55 $\times$** | 668.2 MB | 327.36 MB |
| **CLIP (ViT-B/32)** | 4.75 ms | 20.02 ms | **4.21 $\times$** | 1,079.7 MB | 577.23 MB |
| **Proposed IMO Framework** | **0.19 ms** | **0.19 ms** | **1.00 $\times$** | **0.0 MB** | **4.24 MB** |

### Key Hardware Insights:
- **ViT-B/16 shows heavy GPU dependency (6.55x speedup):** Quadratic self-attention matrix operations cause CPU latency to spike to 71.17 ms, making it impractical for CPU-only edge devices.
- **MobileNetV3-Small is CPU-friendly (1.07x speedup):** Depthwise convolutions are memory-bandwidth bound at batch size 1, running almost identically on CPU (3.66 ms) and GPU (3.42 ms).
- **Vector Retrieval is CPU/GPU Invariant (0.19 ms):** HNSW graph search operates entirely via CPU pointer chasing, yielding over 5,100 queries/sec with zero GPU VRAM consumption.

---

## 🛡️ Noise Robustness Benchmark (Hypothesis H1)

To evaluate memory integrity under adversarial or erroneous data entry, synthetic label noise was injected into the training pool at rates of 0%, 10%, 20%, and 30%:

| Injected Noise Rate | Naive Retrieval Accuracy | Proposed IMO Accuracy | IMO Quarantined Samples | Accuracy Retention Margin |
| :---: | :---: | :---: | :---: | :---: |
| **0% (Clean)** | 98.92% | 98.92% | 13 | +0.00% |
| **10% Noise** | 98.39% | 98.92% | 99 | **+0.53%** |
| **20% Noise** | 93.55% | 98.39% | 187 | **+4.84%** |
| **30% Noise** | 80.65% | 98.39% | 273 | **+17.74%** |

> **Hypothesis H1 Validated:** Under 30% label corruption, naive retrieval collapses by -18.27%, whereas the Proposed IMO Framework intercepts and quarantines 273 corrupted candidates, maintaining **98.39% test accuracy**.

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
│   │   └── sampled_25_percent_seed_777/ # 1,250 balanced benchmark images
│   ├── evaluation/
│   │   ├── metrics.py               # Classification metrics (Acc, F1, ROC-AUC)
│   │   ├── noise_benchmarks.py      # Synthetic label noise experiment runner
│   │   └── resource_tracker.py      # Win32 RAM, CUDA VRAM, and Latency profiler
│   ├── experiments/
│   │   └── run_benchmark.py         # End-to-end multi-model benchmark runner
│   ├── models/
│   │   ├── base_model.py            # Abstract BaseClassifier interface
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
│   │   ├── retrieval_engine.py      # Retrieval & Evidence Fusion pipeline
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
To re-run the entire experimental benchmark across all 7 models on both GPU and CPU:
```powershell
.\run_demo.ps1 -Seed 777 -ForceRebuild
```
Or directly via Python:
```powershell
python tcontext\experiments\run_benchmark.py --seed 777
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
- [`tcontext/reports/END_TO_END_AUDIT.md`](file:///tcontext/reports/END_TO_END_AUDIT.md): Initial repository audit and vulnerability identification.
- [`tcontext/reports/EXPERIMENTAL_METHODOLOGY.md`](file:///tcontext/reports/EXPERIMENTAL_METHODOLOGY.md): Mathematical formulations, dataset partitioning, and evaluation protocol.
- [`tcontext/reports/MODEL_COMPARISON.md`](file:///tcontext/reports/MODEL_COMPARISON.md): In-depth comparison across the 7 visual paradigms.
- [`tcontext/reports/HARDWARE_BENCHMARK.md`](file:///tcontext/reports/HARDWARE_BENCHMARK.md): Hardware profiling analysis under CPU and GPU constraints.
- [`tcontext/reports/IMPLEMENTATION_CHANGELOG.md`](file:///tcontext/reports/IMPLEMENTATION_CHANGELOG.md): Complete changelog of refactoring and modernization.
