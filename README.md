# A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1+cu121](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Thesis Research Repository & Official Implementation**  
GitHub Repository: [`RoushanKhalid/Retrieval-Incremental-Learning`](https://github.com/RoushanKhalid/Retrieval-Incremental-Learning)  
Target Journal: *Journal of Intelligent Information Systems (Springer JIIS)*  
Author: Sk. Roushan Khalid (ID: 221-35-877)  
Department: Software Engineering, Daffodil International University

---

## 📌 Repository Description & Topics

> **Description:** A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning using CLIP representations, ChromaDB HNSW vector indexing, and an Intelligent Memory Orchestrator (IMO) for noise-resistant continual adaptation under resource constraints.

**GitHub Topics / Tags:**  
`continual-learning` • `incremental-learning` • `retrieval-augmented-generation` • `vector-database` • `chromadb` • `clip` • `vision-language` • `edge-ai` • `intelligent-information-systems` • `pytorch` • `hnsw` • `zero-shot`

---

## 📖 Research Overview

Adapting deep neural networks to continuous streams of visual knowledge typically requires iterative backpropagation. This introduces three critical hurdles:
1. **Catastrophic Forgetting:** Weight updates degrade historical decision boundaries.
2. **Computational Overhead:** Retraining backbones consumes significant energy and wall-clock latency.
3. **Vulnerability to Label Noise:** Standard gradient descent absorbs mislabeled inputs into network weights.

This repository decouples representation extraction from knowledge retention:
- **Invariant Semantic Encoder:** Frozen CLIP ViT-B/32 backbone ($\nabla_\theta = \mathbf{0}$) guarantees zero catastrophic forgetting.
- **Metric Retrieval Adapter:** Lightweight contrastive projection head trained via supervised contrastive loss.
- **Intelligent Memory Orchestrator (IMO):** Quality gatekeeper enforcing prototype consistency, relative margin verification, and density filtering to quarantine corrupted samples.
- **Dynamic Evidence Fusion:** Calibrated Bayesian consensus between parametric and non-parametric posteriors.

---

## 🏆 Multi-Model Benchmark Results

Evaluated on the full **Microsoft Cats vs. Dogs** benchmark ($N=2,500$ verified images) under a leak-free **3-way stratified partition** (70% Train: 1,752, 15% Validation: 374, 15% Hold-Out Test: 374, Seed `777`).

### GPU Performance Leaderboard (`cuda:0` - NVIDIA GeForce RTX 2060)

| Model Architecture | Category | Accuracy (%) | Precision | Recall | Macro F1 | ROC-AUC | Train / Index Time | Inference Latency | Throughput (img/s) | Parameters ($N_{\text{total}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | Edge CNN (Fine-Tuned) | 97.33% | 0.9733 | 0.9733 | 0.9733 | 0.9963 | 116.05 s | 3.30 ms | 302.8 | 928,162 |
| **EfficientNetB0** | Compound CNN (Fine-Tuned) | 98.66% | 0.9840 | 0.9893 | 0.9867 | 0.9998 | 175.18 s | 3.75 ms | 266.9 | 4,010,110 |
| **ResNet18** | Residual CNN (Fine-Tuned) | 99.20% | 0.9894 | 0.9947 | 0.9920 | 0.9990 | 149.20 s | 3.69 ms | 271.0 | 11,177,538 |
| **ViT-B/16** | Vision Transformer (Fine-Tuned) | 99.47% | 0.9947 | 0.9947 | 0.9947 | 0.9997 | 479.95 s | 10.98 ms | 91.0 | 85,800,194 |
| **CLIP (ViT-B/32)** | Foundation Vision-Language | 99.73% | 0.9947 | 1.0000 | 0.9973 | 1.0000 | 173.16 s | 4.43 ms | 225.7 | 151,278,339 |
| **Naive Retrieval** | Vector Index (ChromaDB) | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.80 s** | **0.18 ms** | **5,653.9** | **0 (Frozen)** |
| **Fine-Tuned Retrieval** | Contrastive Adapter + DB | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.69 s** | **0.14 ms** | **7,152.5** | 526,337 |
| **Proposed IMO Framework** | Vector Index + IMO Orchestrator | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.27 s** | **0.15 ms** | **6,726.0** | **0 (Frozen)** |

---

## ⚡ Hardware Profiling: GPU vs. CPU Constraints

| Model Architecture | GPU Latency | CPU Latency | Hardware Speedup ($\times$) | Peak GPU VRAM | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 3.30 ms | 3.05 ms | **0.92 $\times$** | 103.96 MB | 3.66 MB |
| **EfficientNetB0** | 3.75 ms | 9.82 ms | **2.62 $\times$** | 394.38 MB | 15.58 MB |
| **ResNet18** | 3.69 ms | 10.07 ms | **2.73 $\times$** | 370.00 MB | 42.71 MB |
| **ViT-B/16** | 10.98 ms | 62.40 ms | **5.68 $\times$** | 1,066.48 MB | 327.36 MB |
| **CLIP (ViT-B/32)** | 4.43 ms | 17.40 ms | **3.93 $\times$** | 1,541.63 MB | 577.23 MB |
| **Fine-Tuned Retrieval** | **0.14 ms** | **0.14 ms** | **1.00 $\times$** | **0.0 MB** | **11.60 MB** |
| **Proposed IMO Framework** | **0.15 ms** | **0.15 ms** | **1.00 $\times$** | **0.0 MB** | **11.59 MB** |

---

## 🛡️ Noise Robustness Benchmark (Hypothesis H1)

Evaluated under symmetric synthetic label noise across all 1,752 training instances:

| Injected Noise Rate | Naive Retrieval Accuracy | Proposed IMO Accuracy | IMO Quarantined Samples | Accuracy Retention Margin |
| :---: | :---: | :---: | :---: | :---: |
| **0% (Clean)** | 100.00% | 100.00% | 0 | +0.00% |
| **10% Noise** | 98.40% | 100.00% | 175 | **+1.60%** |
| **20% Noise** | 91.98% | 100.00% | 350 | **+8.02%** |
| **30% Noise** | 80.48% | 100.00% | 525 | **+19.52%** |

---

## 🗂️ Repository Architecture

```
Retrieval-Incremental-Learning/
├── configs/                         # Experiment, hardware, and model configs
├── paper_latex/                     # Complete Springer Nature JIIS LaTeX paper source
├── paper_latex.zip                  # Ready-to-compile archive for Overleaf / LaTeX
├── tcontext/
│   ├── artifacts/                   # Benchmark CSVs, summary JSONs, and figures
│   ├── data/full_dataset/           # 2,500 validated benchmark images
│   ├── evaluation/                  # Metrics, noise benchmark, and profiler
│   ├── experiments/run_benchmark.py # Master benchmark runner
│   ├── models/                      # PyTorch models & 2-stage fine-tuning
│   ├── reports/                     # Formal thesis & audit reports
│   ├── retrieval/                   # IMO, contrastive adapter & ChromaDB engine
│   └── web_app.py                   # Streamlit interactive research dashboard
├── launch_app.ps1                   # One-click dashboard launcher
├── run_demo.ps1                     # One-click benchmark runner
└── README.md                        # Project documentation
```

---

## 🚀 Quickstart & Execution

```powershell
# 1. Launch Interactive Dashboard
.\launch_app.ps1

# 2. Run Full Multi-Model Benchmark
python tcontext\experiments\run_benchmark.py --epochs 15 --seed 777
```

---

## 📖 Citation

If you use this framework or benchmark in your research, please cite:

```bibtex
@article{khalid2026retrieval,
  title={A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning},
  author={Khalid, Sk. Roushan and Arman, Md. Shohel},
  journal={Journal of Intelligent Information Systems (Under Review)},
  year={2026},
  publisher={Springer}
}
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
