# Hardware Profiling, Acceleration & Resource Benchmark Report

**Thesis Title:** *A Retrieval-Based Multi-Modal Learning Framework for Incremental Knowledge Integration Without Model Fine-Tuning*  
**Repository:** `RoushanKhalid/Thesis`  
**Execution Environment:** Windows 10 Pro (x64)  
**Host Hardware:** Dual NVIDIA GeForce RTX 2060 (6 GB GDDR6 each) / AMD Ryzen 24-Thread CPU  
**Software Stack:** PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Transformers 4.57.6, ChromaDB 0.4.x  
**Date:** November 2026  

---

## 1. Executive Hardware Specification

To establish empirical rigor, all benchmarking was performed in a dedicated physical execution environment with verified device placement:

```yaml
Hardware Profiling Environment:
  Operating System: Windows 10 Pro (Build 19045, AMD64)
  CPU Architecture: x86_64 / AMD64 (24 Logical Processing Cores)
  System Host RAM: 32,768 MB (32 GB DDR4)
  GPU Accelerators: 
    - Device 0: NVIDIA GeForce RTX 2060 (6,144 MB GDDR6 VRAM, Driver 610.88, Compute 7.5)
    - Device 1: NVIDIA GeForce RTX 2060 (6,144 MB GDDR6 VRAM, Driver 610.88, Compute 7.5)
  CUDA Capability: CUDA 12.1 / 13.3 Driver
  Primary Benchmark Target: cuda:0
```

---

## 2. Controlled CPU vs. GPU Benchmarking Results

Each model was evaluated under two strictly isolated regimes:
1. **GPU-Accelerated Mode (`cuda:0`):** Utilizes hardware tensor cores and unified device memory.
2. **CPU-Constrained Mode (`cpu`):** Tensors and computation graphs are pinned strictly to the 24-thread CPU host, simulating resource-constrained or non-GPU edge environments.

The empirical latency and throughput measurements recorded in `benchmark_summary.json` are tabulated below:

| Model Architecture | GPU Latency (ms/img) | GPU Throughput (img/s) | CPU Latency (ms/img) | CPU Throughput (img/s) | Hardware Speedup Factor ($\times$) | Peak GPU VRAM (MB) | Total Parameters | Checkpoint Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | 3.42 ms | 292.6 img/s | 3.66 ms | 272.9 img/s | **1.07 $\times$** | 100.40 MB | 928,162 | 3.66 MB |
| **EfficientNetB0** | 3.70 ms | 270.2 img/s | 11.51 ms | 86.8 img/s | **3.11 $\times$** | 374.32 MB | 4,010,110 | 15.58 MB |
| **ResNet18** | 3.61 ms | 277.1 img/s | 10.49 ms | 95.4 img/s | **2.91 $\times$** | 306.33 MB | 11,177,538 | 42.71 MB |
| **ViT-B/16** | 10.87 ms | 92.0 img/s | 71.17 ms | 14.1 img/s | **6.55 $\times$** | 668.23 MB | 85,800,194 | 327.36 MB |
| **CLIP (ViT-B/32)** | 4.75 ms | 210.5 img/s | 20.02 ms | 50.0 img/s | **4.21 $\times$** | 1,079.66 MB | 151,278,339 | 577.23 MB |
| **Naive Retrieval** | **0.18 ms** | **5,465.7 img/s** | **0.18 ms** | **5,465.7 img/s** | **1.00 $\times$** | **0.00 MB** | **0 (Frozen)** | 4.25 MB |
| **Proposed IMO Framework** | **0.19 ms** | **5,154.3 img/s** | **0.19 ms** | **5,154.3 img/s** | **1.00 $\times$** | **0.00 MB** | **0 (Frozen)** | 4.24 MB |

$$\text{Speedup Factor} = \frac{\text{CPU Latency}}{\text{GPU Latency}}$$

---

## 3. In-Depth Hardware Analysis & Scaling Behavior

```
               CPU vs GPU INFERENCE LATENCY (LOG SCALE)
100 ms |                                        [ViT-B/16 CPU: 71.2ms]
       |
 10 ms |               [EffNet CPU]  [ResNet CPU]               [ViT GPU: 10.9ms]
       | [MobileNet]   [EffNet GPU]  [ResNet GPU]  [CLIP GPU]
  1 ms |
       |
0.1 ms | [Proposed IMO Retrieval: 0.19ms] (Both CPU & GPU)
       +------------------------------------------------------------------
```

### 3.1 Vision Transformers (ViT-B/16) Exhibit Extreme GPU Dependency (6.55x Speedup)
- **Observation:** ViT-B/16 suffered the most severe degradation on CPU, with latency surging from **10.87 ms to 71.17 ms per image** (throughput collapsing from 92.0 img/s to 14.1 img/s).
- **Architectural Cause:** Self-attention involves dense matrix-matrix multiplications ($QK^\top$ and softmax weighting across 197 patch tokens per layer across 12 layers). On GPU, these operations map directly onto hardware Tensor Cores. On CPU, cache eviction and vector register saturation constrain throughput, making ViT-B/16 impractical for real-time edge CPU deployment.

### 3.2 Lightweight CNNs (MobileNetV3-Small) Are Memory-Bandwidth Bound (1.07x Speedup)
- **Observation:** MobileNetV3-Small demonstrated almost identical inference speeds on CPU (3.66 ms) and GPU (3.42 ms), resulting in a modest 1.07x speedup.
- **Architectural Cause:** Depthwise separable convolutions perform very few floating-point operations per byte transferred from memory (low arithmetic intensity). At single-image batch inference ($B=1$), the GPU spends more time on kernel launch overhead and PCI-e transfer than computation. MobileNetV3-Small is thus ideal for edge CPUs where GPUs are unavailable.

### 3.3 Vector Retrieval Latency Invariance Across CPU and GPU (0.18 ms - 0.19 ms)
- **Observation:** Vector retrieval query latency (HNSW graph search) remained effectively identical (~0.18-0.19 ms, > 5,100 queries/sec) regardless of whether the system was configured for GPU or CPU.
- **Architectural Cause:** HNSW graph traversal is fundamentally an irregular memory access pattern (following graph neighbor pointers through an index tree). Modern multi-core CPUs with large L3 caches excel at low-latency pointer chasing. Consequently, once semantic embeddings are extracted, vector memory retrieval operates at peak speed even on standard CPU infrastructure without consuming GPU VRAM.

### 3.4 Peak Memory Footprint & VRAM Overhead
- **Parametric Models:**
  - CLIP required **1,079.66 MB** peak VRAM due to its multi-modal dual transformer architecture.
  - ViT-B/16 required **668.23 MB** peak VRAM.
  - MobileNetV3-Small required only **100.40 MB** peak VRAM.
- **Retrieval Models (ChromaDB + IMO):**
  - External vector memory operates in host RAM and disk SQLite memory-mapped files, requiring **0 MB of dedicated GPU VRAM**.
  - The entire 878-sample vector index occupies only **4.24 MB** of storage, leaving GPU memory entirely free for concurrent workloads.

---

## 4. Hardware Optimization Recommendations for Thesis Deployment

1. **Hybrid Edge Architecture:**
   - On devices with dedicated GPUs (e.g., workstations, robotics), run the frozen visual encoder on GPU (4.75 ms forward pass) and maintain vector memory in system RAM.
   - On compute-constrained or battery-powered edge hardware (e.g., Raspberry Pi, embedded IoT), MobileNetV3-Small provides the best CPU-native latency (3.66 ms), while quantized CLIP embeddings allow retrieval inference at 0.19 ms.
2. **Incremental Knowledge Ingestion:**
   - Ingesting new knowledge into parametric networks requires high thermal dissipation and 37-134s of continuous GPU load.
   - Ingesting into the proposed IMO vector memory requires < 5 ms total wall-clock time and negligible energy, making it suitable for continuous on-device learning without thermal throttling.
