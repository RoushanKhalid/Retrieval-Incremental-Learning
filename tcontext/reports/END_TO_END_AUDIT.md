# Comprehensive End-to-End Thesis Repository Audit

**Project Title:** A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning  
**Repository:** `RoushanKhalid/Thesis`  
**Author / Candidate:** Sk. Roushan Khalid (ID: 221-35-877)  
**Supervisor:** Md. Shohel Arman, Assistant Professor, Department of Software Engineering, Daffodil International University  
**Audit Date:** November 2026  
**Auditor:** Antigravity Autonomous Systems Specialist (Google DeepMind)

---

## Executive Summary

An exhaustive end-to-end audit of the repository was conducted prior to making any code modifications. The audit examined every script, configuration file, launch script, generated artifact, dataset directory, documentation claim, and the local execution hardware environment.

While the thesis proposes a compelling and timely research question—evaluating non-parametric retrieval memory and an Intelligent Memory Orchestrator (IMO) as an alternative to continual fine-tuning—the audit identified severe scientific validity discrepancies, architectural flaws, environment configuration bugs, and missing algorithmic components. Crucially:
1. **The central thesis contribution—the Intelligent Memory Orchestrator (IMO)—is largely absent from the experimental benchmark pipeline.** The reported 98.80% retrieval accuracy is simply raw CLIP ViT-B/32 zero-shot feature extraction paired with ChromaDB 5-NN majority voting.
2. **Evaluation suffers from validation set leakage.** The 20% validation split was used simultaneously for training early stopping / learning-rate scheduling and for final benchmark reporting; no held-out, unseen test set exists.
3. **The baseline comparison is scientifically confounded.** EfficientNetB0 (ImageNet-1k pretrained, linear probe on 1,000 images) was compared against CLIP ViT-B/32 (pretrained on 400M internet image-text pairs). The performance gap reflects the representation power of CLIP's foundational pretraining, not the inherent superiority of retrieval over parametric classification.
4. **All historical experiments ran 100% on CPU, despite two NVIDIA GeForce RTX 2060 GPUs being present.** Native Windows TensorFlow 2.21 has dropped GPU support, and the installed PyTorch package was a CPU-only build (`2.12.1+cpu`).
5. **The repository cannot run from a fresh clone.** `quick500_experiment.py` crashes because it hardcodes a dependency on an uncommitted TensorFlow Datasets cache in the user's home directory. Furthermore, the PowerShell launch scripts crash due to WindowsApps 0-byte execution aliases.

Below is the complete forensic audit with prioritized findings, technical analysis, and an implementation roadmap.

---

## 1. Current Architecture and Execution Flow

### 1.1 Directory Layout
```
Thesis/
├── .gitignore                      # Git ignore patterns
├── launch_app.ps1                  # Streamlit launcher script
├── run_demo.ps1                    # Pipeline orchestrator + launcher script
├── README.md                       # Root documentation
├── thesis_enrichment.md            # Conceptual thesis design & theoretical blueprint
├── proposal/
│   └── Sk. Roushan Khalid 221-35-877.pdf  # Defense presentation slides
├── logs/                           # Execution logs
└── tcontext/
    ├── README.md                   # Context-level documentation
    ├── requirements.txt            # Python dependency specification
    ├── dataset_utils.py            # Dataset extraction, cleaning, and caching utility
    ├── quick500_experiment.py      # Monolithic experiment script (783 lines)
    ├── comparative_eda.py          # Post-run EDA and plotting script (233 lines)
    ├── query_demo.py               # CLI query / manual memory insert tool (121 lines)
    ├── web_app.py                  # Streamlit web dashboard (852 lines)
    ├── data/
    │   ├── sampled_10_percent/     # 500 images (250 cats, 250 dogs)
    │   └── sampled_25_percent_seed_777/ # 1,250 images (625 cats, 625 dogs)
    └── reports/                    # Historical markdown logs, CSVs, and PNG charts
```

### 1.2 Execution Flow Analysis
The execution flow is structured around two PowerShell entry points:
- **`launch_app.ps1`**: Aims to start the Streamlit server directly. It kills processes listening on port 8501 and executes `streamlit run tcontext\web_app.py`.
- **`run_demo.ps1`**: Checks whether `tcontext\model\cats_vs_dogs_model_quick500.keras` and `tcontext\vector_db` exist. If missing, it invokes `python tcontext\quick500_experiment.py --seed 777` followed by `python tcontext\comparative_eda.py`, and finally launches Streamlit.
- **`quick500_experiment.py` Execution Path**:
  1. Calls `set_seed(seed)`.
  2. Calls `prepare_quick_subset(seed)` -> calls `dataset_utils.get_dataset_root()`.
  3. `get_dataset_root()` looks for `Path.home() / "tensorflow_datasets" / "downloads" / "cats_vs_dogs" / "*.zip"`.
  4. Generates exploratory data analysis (EDA) plots in `tcontext/artifacts/`.
  5. Instantiates TensorFlow EfficientNetB0, freezes base weights, fits on 80% train split for 12 epochs with EarlyStopping and ReduceLROnPlateau.
  6. Evaluates classifier on the 20% validation split.
  7. Instantiates PyTorch HuggingFace CLIP (`openai/clip-vit-base-patch32`), extracts embeddings for train images, and writes to ChromaDB collection `clip_image_embeddings`.
  8. Queries ChromaDB with validation embeddings, performs 5-NN majority voting, and records accuracy, precision, recall, and F1.
  9. Writes `sampled25_summary.json`, `sampled25_comparison_metrics.csv`, and markdown reports.

---

## 2. Existing Model Implementation Assessment

| Model / Component | Library / Framework | Pretrained Weights | Parameter Status | Role in Repository | Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNetB0** | `tensorflow.keras.applications` | ImageNet-1k | Base frozen (4.05M params), Dense head trained (1,281 params) | Parametric Classifier Baseline | Working (CPU only) |
| **CLIP ViT-B/32** | `transformers` / `torch` | OpenAI WebImageText (400M pairs) | Fully frozen (87.8M visual encoder params) | Non-Parametric Feature Extractor | Working (CPU only) |
| **ChromaDB Vector Store** | `chromadb` (PersistentClient) | N/A | Non-parametric HNSW index | External Memory Storage | Working |
| **Intelligent Memory Orchestrator (IMO)** | Custom | N/A | Logic gating | Thesis Core Contribution | **MISSING in Benchmark Pipeline** |

### Critical Observations:
1. **Absence of IMO in Benchmark Pipeline:** In `quick500_experiment.py`, there is no prototype calculation, no conflict detection, no quarantine mechanism, no density-based outlier filtering, and no evidence fusion. All training embeddings are directly inserted into ChromaDB without gating. In `web_app.py`, there is only a 15-line function `_passes_memory_threshold` that rejects manual user uploads if the EfficientNet classifier's confidence is $< 0.75$. This does not constitute an orchestrator.
2. **Framework Fragmentation:** The repository unnecessarily splits between TensorFlow (for EfficientNetB0) and PyTorch (for CLIP). This doubles memory footprint, complicates hardware acceleration on Windows, and prevents sharing common batch data pipelines.
3. **No Representation Parity:** The repository does not test EfficientNetB0 as a feature extractor in the vector database, nor does it test CLIP with a linear classification head. It compares a small CNN trained on 1,000 images against a massive Vision-Language Transformer.

---

## 3. Dataset and Experimental Methodology Assessment

### 3.1 Data Acquisition and Caching
- **The TFDS Dependency Bug:** `dataset_utils.py` contains `_find_local_tfds_zip()`, which searches `C:\Users\<User>\tensorflow_datasets\downloads\cats_vs_dogs`. This folder does not exist on this machine or in standard clones. Consequently, executing `quick500_experiment.py` raises `FileNotFoundError`.
- **Existing Static Subsets:** The repository already tracks two sampled subsets in Git:
  - `tcontext/data/sampled_10_percent`: 500 images (250 cats, 250 dogs).
  - `tcontext/data/sampled_25_percent_seed_777`: 1,250 images (625 cats, 625 dogs).
  Instead of utilizing these tracked images, `quick500_experiment.py` attempted to re-sample from an uncommitted raw archive.

### 3.2 Partitioning & Data Leakage (Critical Scientific Concern)
- In `quick500_experiment.py`:
  ```python
  (train_paths, train_labels), (val_paths, val_labels) = split_dataset_files(data_root, seed=seed, val_ratio=0.2)
  ```
  The dataset of 1,250 images is split into 1,000 training images (80%) and 250 validation images (20%).
- **Leakage Point 1:** The classifier is fitted on `train_ds` with `validation_data=val_ds`, utilizing `EarlyStopping(monitor="val_loss", restore_best_weights=True)`. Model checkpoint selection is directly driven by `val_ds`.
- **Leakage Point 2:** The final reported classifier metrics (Accuracy: 0.9800, F1: 0.9796, ROC-AUC: 0.9996) are evaluated on this identical `val_ds`.
- **Leakage Point 3:** The retrieval system is also evaluated on `val_paths`.
- **Scientific Implication:** There is **no independent test set**. In academic machine learning, evaluating the final model on the validation set that determined early stopping introduces selection bias and invalidates generalizability claims.
- **Resolution Required:** Establish a fixed three-way stratified split: **Train (70%) / Validation (15%) / Test (15%)**. The Test partition must remain strictly isolated until final model evaluation.

### 3.3 Flawed Incremental Cost Modeling
- Lines 532–537 of `quick500_experiment.py` calculate:
  ```python
  "deep_learning_refit_5_images_seconds": float((train_seconds / max(1, len(train_labels))) * 5)
  ```
  This calculates refit time as 5/1000th of full training time (~3.52 s).
- In reality, retraining a neural network on 1,005 images takes *longer* than original training (~705 s), while fine-tuning on only 5 images requires epoch iterations, optimizer re-initialization, and causes catastrophic forgetting.
- The retrieval update cost ($O(\text{insert})$) should be compared against actual retraining / fine-tuning epochs, not a linear mathematical artifact.

---

## 4. Computational Resource Utilization Assessment

### 4.1 Hardware Environment
- **CPU:** AMD64 architecture, 24 logical execution cores.
- **System RAM:** 32 GB.
- **GPU Accelerator:** **2x NVIDIA GeForce RTX 2060 (6,144 MiB VRAM each)**.
  - Driver: 610.88
  - CUDA Toolkit: v13.2 installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`.
  - CUDA Driver UMD: 13.3.

### 4.2 Software Environment & GPU Invisibility
- **Python Installation:** Python 3.11.3 located at `C:\Users\DSLAB\AppData\Local\Programs\Python\Python311\python.exe`.
- **TensorFlow Status:** `tensorflow==2.21.0`. Starting with TensorFlow 2.11, Google officially deprecated and removed native GPU support on Windows. Native Windows TensorFlow only supports CPU execution (`tf.config.list_physical_devices('GPU')` returns `[]`).
- **PyTorch Status:** The installed PyTorch wheel is `torch==2.12.1+cpu`. Consequently, `torch.cuda.is_available()` returns `False`.
- **Finding:** Despite the workstation possessing two high-performance NVIDIA GPUs, **100% of the historical experiments ran on the CPU**.
- The historical run took **703.95 seconds (~11.7 minutes)** to train 12 epochs on 1,000 images, and 133 seconds to extract 1,000 CLIP embeddings.
- On an RTX 2060 GPU with native CUDA acceleration in PyTorch, feature extraction and classifier training on 1,000 images will take under 15 seconds.

---

## 5. Code Quality and Maintainability Issues

1. **Monolithic Architecture:** `quick500_experiment.py` (783 lines) contains data preparation, model architecture, training loops, evaluation, EDA plotting, confusion matrices, ROC curves, JSON generation, and report generation in a single flat file.
2. **Duplicated Code:**
   - Image loading and preprocessing logic is duplicated in `quick500_experiment.py`, `dataset_utils.py`, `query_demo.py`, and `web_app.py`.
   - Metric formatting and plotting are duplicated between `quick500_experiment.py` and `comparative_eda.py`.
3. **Fragile PowerShell Scripts:**
   - `run_demo.ps1` invokes `(Get-Command python).Source`, which picks up `C:\Users\DSLAB\AppData\Local\Microsoft\WindowsApps\python.exe` (a 0-byte redirection shim to the Windows Store), throwing a fatal error.
   - `launch_app.ps1` runs `streamlit run` directly, but `streamlit.exe` is not present in PATH.
4. **Hardcoded Assumptions:** Arbitrary thresholds (`min_confidence = 0.75`, `k = 5`, `val_ratio = 0.2`, `epochs = 12`) are hardcoded without configuration files or ablation support.

---

## 6. Documentation Inconsistencies

1. **Claims vs Code Discrepancies:**
   - `README.md` and `thesis_enrichment.md` state that an Intelligent Memory Orchestrator (IMO) handles confidence gating, prototype consistency, and outlier filtering. In the codebase, this is nonexistent in the benchmark and only a single confidence check in `web_app.py`.
   - `README.md` claims "O(1) incremental updates" for vector retrieval. HNSW insertion is empirically and theoretically $O(\log N)$ in search space, plus $O(D)$ embedding latency.
2. **Missing Experimental Context:**
   - The reported accuracy figures (98.00% vs 98.80%) omit the fact that both were evaluated on the validation set, and that EfficientNetB0 was evaluated with a single train/val run rather than cross-validated or tested on held-out samples.
3. **Artifact Disconnect:**
   - `tcontext/reports/` contains PNG files with irregular filenames containing spaces and double extensions (e.g., `comparison_confusion_matricespng.png`, `deep learning curvepng.png`).

---

## 7. Audit Findings Classification

| Finding ID | Severity | Category | Description |
| :--- | :--- | :--- | :--- |
| **F-01** | **CRITICAL** | Scientific Validity | Validation set leakage: 20% validation set used for early stopping and final metric reporting. |
| **F-02** | **CRITICAL** | Scientific Validity | Missing Intelligent Memory Orchestrator (IMO) implementation in experimental benchmark pipeline. |
| **F-03** | **CRITICAL** | Scientific Validity | Confounded baseline comparison: 400M-pretrained CLIP vs 1,000-image fine-tuned EfficientNetB0 without representation controls. |
| **F-04** | **CRITICAL** | Functional Bug | `quick500_experiment.py` crashes due to missing `~/tensorflow_datasets` archive. |
| **F-05** | **CRITICAL** | Functional Bug | PowerShell scripts crash due to WindowsApps 0-byte python.exe shim and missing streamlit in PATH. |
| **F-06** | **HIGH** | Hardware / Compute | GPU acceleration completely disabled: TF Windows CPU-only + PyTorch CPU wheel on dual-RTX 2060 system. |
| **F-07** | **HIGH** | Methodology | Scientifically invalid incremental refit cost estimation formula ($t_{train} \times 5/N$). |
| **F-08** | **HIGH** | Methodology | Single seed / split; lack of statistical variance, confidence intervals, or repeated runs. |
| **F-09** | **MEDIUM** | Code Quality | Monolithic, duplicated scripts (`quick500_experiment.py` and `web_app.py`). |
| **F-10** | **MEDIUM** | Architecture | Framework fragmentation (TF for CNN, PyTorch for CLIP) hindering unified benchmarking. |
| **F-11** | **LOW** | Documentation | Irregular file names with spaces and double extensions in reports directory. |

---

## 8. Recommended Multi-Model Benchmark Strategy

To address the thesis board's recommendation and fulfill the thesis core research question under limited computational resources (CPU vs GPU constraints), the comparison must expand beyond a two-model setup to evaluate **three distinct paradigm levels**:

### Level 1: Predictive Performance
Accuracy, Precision, Recall, Macro F1, ROC-AUC, and Confusion Matrix on a strictly held-out test set.

### Level 2: Computational Resource Efficiency
Parameters, Checkpoint Size, Embedding Dimension, Feature Extraction Time, Training Time, Inference Latency (Mean, Median, p95), Throughput (samples/sec), Peak RAM, Peak VRAM, and Execution Device Verification (CPU vs GPU).

### Level 3: Incremental Knowledge Integration (Thesis Core Contribution)
Insertion Latency per item, Retrieval Latency, Memory Footprint Growth, Resistance to Injected Label Noise (10%, 20%, 30%), and IMO Quality Gating (Acceptance Rate, Quarantine Rate, Prototype Consistency).

### Recommended Model Suite (5–7 Representative Approaches)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Multi-Model Benchmark Suite                          │
├────────────────────────────┬─────────────────────────────┬───────────────────────┤
│ Approach Paradigm          │ Model Architecture          │ Primary Research Role │
├────────────────────────────┼─────────────────────────────┼───────────────────────┤
│ 1. Lightweight CNN         │ MobileNetV3-Small / Large   │ Ultra-low CPU latency │
│ 2. Standard CNN Baseline   │ EfficientNetB0              │ Existing baseline     │
│ 3. Residual CNN Baseline   │ ResNet18                    │ Classic inductive bias│
│ 4. Vision Transformer      │ ViT-B/16 (or ViT-Tiny/Small)│ Attention mechanisms  │
│ 5. Vision-Language Encoder │ CLIP (ViT-B/32 visual head) │ Multi-modal semantic  │
│ 6. Baseline Retrieval      │ Frozen CLIP + ChromaDB      │ Naive memory baseline │
│ 7. Proposed Framework      │ CLIP + ChromaDB + IMO       │ Thesis contribution   │
└────────────────────────────┴─────────────────────────────┴───────────────────────┘
```

**Unified Framework Recommendation:**  
Implement all model backbones within **PyTorch / Torchvision / Transformers**. This eliminates the Windows native TensorFlow GPU limitation, enables direct GPU execution on the dual RTX 2060 cards via CUDA, unifies the batching pipeline, and ensures standard, reproducible benchmarking across all architectures.

---

## 9. Recommended Modular Architecture

```
Thesis/
├── configs/
│   ├── experiment_config.yaml       # Dataset, splits, seeds, hyperparameters
│   └── hardware_config.yaml         # Device placement, batch sizes, threads
├── data/                            # Canonical dataset location
│   └── cats_vs_dogs_25pct/          # 1,250 verified images
├── tcontext/
│   ├── data/
│   │   ├── dataset_loader.py        # Reproducible 3-way split (Train/Val/Test)
│   │   └── preprocessing.py         # Standardized transforms per model
│   ├── models/
│   │   ├── base_classifier.py       # Abstract base interface
│   │   ├── cnn_models.py            # MobileNetV3, EfficientNetB0, ResNet18
│   │   ├── vit_model.py             # Vision Transformer
│   │   └── clip_model.py            # CLIP visual encoder
│   ├── retrieval/
│   │   ├── vector_store.py          # ChromaDB / persistent vector memory
│   │   ├── imo.py                   # Intelligent Memory Orchestrator (Prototypes, Gating, Quarantine)
│   │   └── hybrid_engine.py         # Evidence fusion (base + retrieval)
│   ├── evaluation/
│   │   ├── metrics.py               # Classification metrics (Acc, F1, AUC)
│   │   ├── resource_tracker.py      # Latency, throughput, RAM, VRAM, GPU util
│   │   └── noise_benchmarks.py      # Robustness under corrupted/noisy updates
│   ├── experiments/
│   │   ├── run_benchmark.py         # Central unified experiment runner
│   │   ├── run_cpu_benchmark.py     # Controlled CPU execution
│   │   └── run_gpu_benchmark.py     # GPU-accelerated execution
│   ├── visualization/
│   │   └── plot_benchmarks.py       # Publication-ready comparative figures
│   ├── web_app.py                   # Upgraded multi-model Streamlit dashboard
│   └── reports/                     # Generated experiment outputs & docs
├── tests/
│   └── test_pipeline.py             # Automated end-to-end sanity tests
├── launch_app.ps1                   # Fixed robust launcher
├── run_demo.ps1                     # Fixed robust pipeline runner
├── requirements.txt                 # Clean pinned dependencies
└── README.md                        # Thesis documentation
```

---

## 10. Prioritized Implementation Roadmap

1. **Phase 1: Environment & Tooling Normalization**
   - Configure Python 3.11 environment with proper PATH resolution.
   - Verify PyTorch with CUDA support for native GPU execution on the RTX 2060 GPUs.
   - Establish clean offline dataset loading directly from `tcontext/data/sampled_25_percent_seed_777`.
2. **Phase 2: Formalize Experimental Protocol**
   - Create deterministic 3-way partition: Train (70%), Validation (15%), Test (15%).
   - Guarantee zero test set leakage into hyperparameter or threshold selection.
3. **Phase 3: Implement Modular Multi-Model Suite**
   - Implement unified PyTorch model loaders: MobileNetV3, EfficientNetB0, ResNet18, ViT, CLIP.
   - Implement feature extraction and classification heads.
4. **Phase 4: Implement the Intelligent Memory Orchestrator (IMO)**
   - Prototype calculation and cosine consistency gating ($\cos(z, P_y) \ge \tau_c$).
   - Conflict detection and quarantine routing.
   - Density/outlier filtering (Isolation Forest / Cosine kNN distance).
   - Fusion engine: $p(c|x) = \lambda p_{base}(c|x) + (1-\lambda) p_{ret}(c|x)$.
5. **Phase 5: Execute Controlled CPU and GPU Benchmarks**
   - Run multi-model evaluation across CPU-only and GPU-accelerated modes.
   - Collect empirical resource measurements: parameter counts, latencies (mean, p95), throughput, RAM, VRAM, and model disk sizes.
   - Run incremental noise injection experiment (0%, 10%, 20%, 30% label corruption) comparing Naive Retrieval vs IMO-enhanced Retrieval.
6. **Phase 6: Update Dashboard, Artifacts, and Documentation**
   - Update Streamlit dashboard to dynamically load all multi-model results and resource profiles.
   - Fix PowerShell launch scripts to execute seamlessly.
   - Produce all required markdown thesis reports (`EXPERIMENTAL_METHODOLOGY.md`, `MODEL_COMPARISON.md`, `HARDWARE_BENCHMARK.md`, `IMPLEMENTATION_CHANGELOG.md`).
   - Update root `README.md`.
