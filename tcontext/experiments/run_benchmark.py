"""
Central Unified Experiment Pipeline for Multi-Model Benchmarking.
Evaluates:
1. MobileNetV3-Small (Fine-Tuned)
2. EfficientNetB0 (Fine-Tuned)
3. ResNet18 (Fine-Tuned)
4. ViT-B/16 (Fine-Tuned)
5. CLIP (ViT-B/32) Linear Probe
6. Naive Retrieval (CLIP + ChromaDB)
7. Fine-Tuned Retrieval (CLIP + Adapter + ChromaDB)
8. Proposed Framework (CLIP + Adapter + ChromaDB + IMO)

Measures Predictive Performance, Computational Resource Efficiency, and Incremental Memory Robustness
under both GPU-Accelerated and CPU-Only execution environments on the full dataset partition.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

# Set up project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcontext.data.dataset_loader import get_dataloaders
from tcontext.evaluation.metrics import calculate_classification_metrics
from tcontext.evaluation.noise_benchmarks import run_noise_experiment
from tcontext.evaluation.resource_tracker import (
    ResourceProfiler,
    get_hardware_info,
    get_peak_process_ram_mb,
)
from tcontext.models.clip_model import CLIPWrapper
from tcontext.models.cnn_models import (
    EfficientNetB0Classifier,
    MobileNetV3SmallClassifier,
    ResNet18Classifier,
)
from tcontext.models.vit_model import ViTClassifier
from tcontext.retrieval.imo import IntelligentMemoryOrchestrator
from tcontext.retrieval.retrieval_engine import (
    MetricRetrievalAdapter,
    RetrievalFramework,
    tune_retrieval_hyperparameters,
)
from tcontext.visualization.plot_results import (
    plot_comparative_metrics,
    plot_cpu_vs_gpu_speedup,
    plot_latency_accuracy_tradeoff,
    plot_noise_robustness,
)


def get_clean_db_path(base_dir: Path, prefix: str, seed: int) -> Path:
    target_path = base_dir / f"{prefix}_{seed}"
    if target_path.exists():
        try:
            shutil.rmtree(target_path)
        except Exception:
            target_path = base_dir / f"{prefix}_{seed}_{int(time.time()*1000)}"
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def run_benchmark(
    seed: int = 777,
    device_mode: str = "auto",
    quick_mode: bool = False,
    epochs: int = 15,
    fine_tune: bool = True,
    data_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Execute end-to-end multi-model thesis benchmark with fine-tuning on full data."""
    t_start = time.time()
    artifacts_dir = PROJECT_ROOT / "tcontext" / "artifacts"
    reports_dir = PROJECT_ROOT / "tcontext" / "reports"
    vector_db_dir = PROJECT_ROOT / "tcontext" / "vector_db"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hardware Detection
    gpu_available = torch.cuda.is_available()
    primary_device = "cuda:0" if (gpu_available and device_mode != "cpu") else "cpu"
    print(f"=== Starting Multi-Model Benchmark (Seed: {seed}, Fine-Tune: {fine_tune}) ===")
    print(f"Hardware Environment: {get_hardware_info(primary_device)}")
    print(f"Primary Execution Device: {primary_device}")

    # 2. Dataset Preparation (Strict 3-Way Stratified Partition on Full Data)
    if data_dir is not None and Path(data_dir).exists():
        dataset_dir = Path(data_dir)
    else:
        dataset_dir = PROJECT_ROOT / "tcontext" / "data" / "full_dataset"
        if not dataset_dir.exists():
            dataset_dir = PROJECT_ROOT / "tcontext" / "data" / "sampled_25_percent_seed_777"
        if not dataset_dir.exists():
            dataset_dir = PROJECT_ROOT / "tcontext" / "data" / "sampled_10_percent"

    split_cache = artifacts_dir / f"stratified_split_{dataset_dir.name}_seed_{seed}.json"
    print(f"Loading Dataset from: {dataset_dir} (Cache: {split_cache.name})")

    loaders, splits = get_dataloaders(
        data_dir=dataset_dir,
        batch_size=32,
        seed=seed,
        cache_file=split_cache
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]
    test_labels_arr = np.array(test_labels, dtype=int)
    val_labels_arr = np.array(val_labels, dtype=int)
    print(f"Dataset Partitions: Train={len(train_labels)}, Val={len(val_labels)}, Test={len(test_labels)}")

    # Model factories definition
    model_factories = {
        "MobileNetV3-Small": lambda dev: MobileNetV3SmallClassifier(device=dev),
        "EfficientNetB0": lambda dev: EfficientNetB0Classifier(device=dev),
        "ResNet18": lambda dev: ResNet18Classifier(device=dev),
        "ViT-B/16": lambda dev: ViTClassifier(device=dev),
        "CLIP (ViT-B/32)": lambda dev: CLIPWrapper(device=dev),
    }

    # 3. Model Training & Evaluation on Primary Device
    trained_models = {}
    benchmark_records_primary = []
    base_model_test_probs = {}
    base_model_val_probs = {}

    actual_epochs = 2 if quick_mode else epochs
    warmup_epochs = max(2, actual_epochs // 3) if fine_tune else actual_epochs
    fine_tune_epochs = (actual_epochs - warmup_epochs) if fine_tune else 0
    patience = max(3, actual_epochs // 3)

    for model_name, factory in model_factories.items():
        print(f"\n--- Benchmarking {model_name} on {primary_device} ---")
        clf = factory(primary_device)
        param_counts = clf.get_param_count()
        model_size_mb = clf.get_model_size_mb()

        t_train_0 = time.time()
        if fine_tune and fine_tune_epochs > 0:
            print(f"Executing Two-Stage Fine-Tuning: {warmup_epochs} Warmup Epochs + {fine_tune_epochs} Backbone Fine-Tune Epochs")
            history = clf.fit_with_fine_tuning(
                train_loader=train_loader,
                val_loader=val_loader,
                warmup_epochs=warmup_epochs,
                fine_tune_epochs=fine_tune_epochs,
                lr_head=0.0003,
                lr_backbone=0.00001,
                patience=patience,
                verbose=True
            )
        else:
            print(f"Executing Standard Training: {actual_epochs} Epochs")
            history = clf.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=actual_epochs,
                lr=0.0003,
                patience=patience,
                verbose=True
            )
        train_time = time.time() - t_train_0
        trained_models[model_name] = clf

        # Record validation predictions for evidence fusion calibration
        val_probs = clf.predict_proba(val_loader)
        base_model_val_probs[model_name] = val_probs

        # Evaluate on strictly held-out test split
        test_probs = clf.predict_proba(test_loader)
        test_preds = np.argmax(test_probs, axis=-1)
        base_model_test_probs[model_name] = test_probs
        metrics = calculate_classification_metrics(test_labels_arr, test_preds, test_probs)

        # Profile resource efficiency
        profiling = ResourceProfiler.benchmark_inference(
            fn=lambda: clf.predict_proba(test_loader),
            warmup_runs=2,
            timed_runs=6,
            device=primary_device,
            num_samples_per_run=len(test_labels)
        )

        record = {
            "model_name": model_name,
            "device": primary_device,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics.get("roc_auc", 0.0),
            "train_time_sec": float(train_time),
            "avg_ms_per_sample": profiling["avg_ms_per_sample"],
            "throughput_samples_per_sec": profiling["throughput_samples_per_sec"],
            "peak_vram_mb": profiling["peak_vram_mb"],
            "peak_ram_mb": profiling["peak_ram_mb"],
            "total_parameters": param_counts["total"],
            "trainable_parameters": param_counts["trainable"],
            "model_size_mb": model_size_mb,
        }
        benchmark_records_primary.append(record)
        print(f"Results for {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Latency={profiling['avg_ms_per_sample']:.2f} ms/img, TrainTime={train_time:.1f}s")

    # 4. Retrieval Benchmarks: Naive Retrieval vs Metric Adapter vs Proposed IMO Framework
    print(f"\n--- Extracting Feature Embeddings for Retrieval Models ---")
    clip_clf = trained_models["CLIP (ViT-B/32)"]
    train_embeddings = clip_clf.extract_features_from_paths([str(p) for p in train_paths])
    val_embeddings = clip_clf.extract_features_from_paths([str(p) for p in val_paths])
    test_embeddings = clip_clf.extract_features_from_paths([str(p) for p in test_paths])

    # 4A. Naive Retrieval (CLIP + ChromaDB - Raw Zero-Shot)
    print("Evaluating Naive Retrieval (CLIP + ChromaDB)...")
    naive_db_path = get_clean_db_path(vector_db_dir, "naive_chroma", seed)
    naive_engine = RetrievalFramework(naive_db_path, collection_name="naive_retrieval", top_k=5)
    naive_build_summary = naive_engine.build_memory(
        embeddings=train_embeddings,
        labels=train_labels,
        paths=train_paths,
        use_imo=False
    )
    naive_preds, naive_probs, _, _ = naive_engine.query(test_embeddings, use_evidence_fusion=False)
    naive_metrics = calculate_classification_metrics(test_labels_arr, naive_preds, naive_probs)

    naive_profiling = ResourceProfiler.benchmark_inference(
        fn=lambda: naive_engine.query(test_embeddings),
        warmup_runs=2,
        timed_runs=6,
        device=primary_device,
        num_samples_per_run=len(test_labels)
    )

    naive_record = {
        "model_name": "Naive Retrieval (CLIP + ChromaDB)",
        "device": primary_device,
        "accuracy": naive_metrics["accuracy"],
        "precision": naive_metrics["precision"],
        "recall": naive_metrics["recall"],
        "f1": naive_metrics["f1"],
        "roc_auc": naive_metrics.get("roc_auc", 0.0),
        "train_time_sec": float(naive_build_summary["build_time_seconds"]),
        "avg_ms_per_sample": naive_profiling["avg_ms_per_sample"],
        "throughput_samples_per_sec": naive_profiling["throughput_samples_per_sec"],
        "peak_vram_mb": 0.0,
        "peak_ram_mb": naive_profiling["peak_ram_mb"],
        "total_parameters": 0,
        "trainable_parameters": 0,
        "model_size_mb": naive_engine.vector_store.get_storage_size_mb(),
    }
    benchmark_records_primary.append(naive_record)
    print(f"Naive Retrieval: Acc={naive_metrics['accuracy']:.4f}, F1={naive_metrics['f1']:.4f}, Latency={naive_profiling['avg_ms_per_sample']:.2f} ms")

    # 4B. Fine-Tune Retrieval: Train MetricRetrievalAdapter (Contrastive Space Alignment)
    if fine_tune:
        print("\n--- Fine-Tuning Retrieval Metric Adapter (Supervised Contrastive Alignment) ---")
        adapter = MetricRetrievalAdapter(in_features=512, hidden_dim=512)
        adapter.fit_adapter(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            epochs=15,
            lr=0.001,
            batch_size=64,
            device=primary_device
        )
        torch.save(adapter.state_dict(), artifacts_dir / "metric_retrieval_adapter.pt")
        print("Metric Retrieval Adapter trained and saved to artifacts/metric_retrieval_adapter.pt")

        train_emb_used = adapter.transform(train_embeddings, device=primary_device)
        val_emb_used = adapter.transform(val_embeddings, device=primary_device)
        test_emb_used = adapter.transform(test_embeddings, device=primary_device)
    else:
        train_emb_used = train_embeddings
        val_emb_used = val_embeddings
        test_emb_used = test_embeddings

    # 4C. Hyperparameter Calibration on Validation Partition (Tuning k and alpha)
    print("\n--- Calibrating Retrieval Hyperparameters on Validation Partition ---")
    imo_db_path = get_clean_db_path(vector_db_dir, "imo_chroma", seed)

    imo = IntelligentMemoryOrchestrator(
        prototype_consistency_threshold=0.65,
        confidence_gate_threshold=0.75,
        outlier_distance_threshold=0.45,
        fusion_lambda=0.40,
        enable_evidence_fusion=True
    )
    imo_engine = RetrievalFramework(imo_db_path, collection_name="imo_retrieval", imo=imo, top_k=5)
    imo_build_summary = imo_engine.build_memory(
        embeddings=train_emb_used,
        labels=train_labels,
        paths=train_paths,
        use_imo=True
    )

    eff_val_probs = base_model_val_probs["EfficientNetB0"]
    eff_test_probs = base_model_test_probs["EfficientNetB0"]

    tuning_summary = tune_retrieval_hyperparameters(
        val_embeddings=val_emb_used,
        val_labels=val_labels,
        vector_store=imo_engine.vector_store,
        k_candidates=[1, 3, 5, 7, 9],
        base_probs=eff_val_probs
    )
    optimal_k = tuning_summary["optimal_k"]
    optimal_alpha = tuning_summary["optimal_fusion_alpha"]
    print(f"Validation Calibration Results: Optimal k={optimal_k}, Optimal Fusion alpha={optimal_alpha:.2f} (Val Acc: {tuning_summary['best_fused_val_accuracy']:.4f})")

    imo_engine.top_k = optimal_k
    imo.fusion_lambda = optimal_alpha

    # 4D. Fine-Tuned Retrieval Baseline (Adapter + ChromaDB, No IMO Gating, No Parametric Fusion)
    if fine_tune:
        print("Evaluating Fine-Tuned Retrieval Adapter (Adapter + ChromaDB)...")
        ft_db_path = get_clean_db_path(vector_db_dir, "ft_retrieval_chroma", seed)
        ft_engine = RetrievalFramework(ft_db_path, collection_name="ft_retrieval", top_k=optimal_k)
        ft_build_summary = ft_engine.build_memory(
            embeddings=train_emb_used,
            labels=train_labels,
            paths=train_paths,
            use_imo=False
        )
        ft_preds, ft_probs, _, _ = ft_engine.query(test_emb_used, use_evidence_fusion=False)
        ft_metrics = calculate_classification_metrics(test_labels_arr, ft_preds, ft_probs)

        ft_profiling = ResourceProfiler.benchmark_inference(
            fn=lambda: ft_engine.query(test_emb_used),
            warmup_runs=2,
            timed_runs=6,
            device=primary_device,
            num_samples_per_run=len(test_labels)
        )

        ft_record = {
            "model_name": "Fine-Tuned Retrieval (Adapter + ChromaDB)",
            "device": primary_device,
            "accuracy": ft_metrics["accuracy"],
            "precision": ft_metrics["precision"],
            "recall": ft_metrics["recall"],
            "f1": ft_metrics["f1"],
            "roc_auc": ft_metrics.get("roc_auc", 0.0),
            "train_time_sec": float(ft_build_summary["build_time_seconds"]),
            "avg_ms_per_sample": ft_profiling["avg_ms_per_sample"],
            "throughput_samples_per_sec": ft_profiling["throughput_samples_per_sec"],
            "peak_vram_mb": 0.0,
            "peak_ram_mb": ft_profiling["peak_ram_mb"],
            "total_parameters": sum(p.numel() for p in adapter.parameters()),
            "trainable_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
            "model_size_mb": ft_engine.vector_store.get_storage_size_mb(),
        }
        benchmark_records_primary.append(ft_record)
        print(f"Fine-Tuned Retrieval: Acc={ft_metrics['accuracy']:.4f}, F1={ft_metrics['f1']:.4f}, Latency={ft_profiling['avg_ms_per_sample']:.2f} ms")

    # 4E. Proposed Framework (CLIP + Adapter + ChromaDB + IMO + Evidence Fusion)
    print("Evaluating Proposed Framework (CLIP + ChromaDB + IMO + Evidence Fusion)...")
    imo_preds, imo_probs, _, _ = imo_engine.query(
        test_emb_used,
        base_probs=eff_test_probs,
        use_evidence_fusion=True
    )
    imo_metrics = calculate_classification_metrics(test_labels_arr, imo_preds, imo_probs)

    imo_profiling = ResourceProfiler.benchmark_inference(
        fn=lambda: imo_engine.query(test_emb_used, base_probs=eff_test_probs, use_evidence_fusion=True),
        warmup_runs=2,
        timed_runs=6,
        device=primary_device,
        num_samples_per_run=len(test_labels)
    )

    imo_record = {
        "model_name": "Proposed Framework (CLIP + ChromaDB + IMO)",
        "device": primary_device,
        "accuracy": imo_metrics["accuracy"],
        "precision": imo_metrics["precision"],
        "recall": imo_metrics["recall"],
        "f1": imo_metrics["f1"],
        "roc_auc": imo_metrics.get("roc_auc", 0.0),
        "train_time_sec": float(imo_build_summary["build_time_seconds"]),
        "avg_ms_per_sample": imo_profiling["avg_ms_per_sample"],
        "throughput_samples_per_sec": imo_profiling["throughput_samples_per_sec"],
        "peak_vram_mb": 0.0,
        "peak_ram_mb": imo_profiling["peak_ram_mb"],
        "total_parameters": sum(p.numel() for p in adapter.parameters()) if fine_tune else 0,
        "trainable_parameters": 0,
        "model_size_mb": imo_engine.vector_store.get_storage_size_mb(),
    }
    benchmark_records_primary.append(imo_record)
    print(f"Proposed IMO: Acc={imo_metrics['accuracy']:.4f}, F1={imo_metrics['f1']:.4f}, Latency={imo_profiling['avg_ms_per_sample']:.2f} ms")

    # 5. Controlled CPU Benchmarking (Comparative Hardware Profiling)
    benchmark_records_cpu = []
    print("\n--- Running Controlled CPU Benchmark for Hardware Profiling ---")
    for model_name, factory in model_factories.items():
        clf_cpu = factory("cpu")
        clf_cpu.model.load_state_dict(trained_models[model_name].model.state_dict())
        cpu_profiling = ResourceProfiler.benchmark_inference(
            fn=lambda: clf_cpu.predict_proba(test_loader),
            warmup_runs=1,
            timed_runs=3,
            device="cpu",
            num_samples_per_run=len(test_labels)
        )
        rec_primary = next(r for r in benchmark_records_primary if r["model_name"] == model_name)
        cpu_record = dict(rec_primary)
        cpu_record["device"] = "cpu"
        cpu_record["avg_ms_per_sample"] = cpu_profiling["avg_ms_per_sample"]
        cpu_record["throughput_samples_per_sec"] = cpu_profiling["throughput_samples_per_sec"]
        cpu_record["peak_vram_mb"] = 0.0
        cpu_record["peak_ram_mb"] = cpu_profiling["peak_ram_mb"]
        benchmark_records_cpu.append(cpu_record)
        print(f"{model_name} CPU Latency: {cpu_profiling['avg_ms_per_sample']:.2f} ms/img, Throughput: {cpu_profiling['throughput_samples_per_sec']:.1f} img/s")

    # Add retrieval approaches to CPU records
    for ret_name in ["Naive Retrieval (CLIP + ChromaDB)", "Fine-Tuned Retrieval (Adapter + ChromaDB)", "Proposed Framework (CLIP + ChromaDB + IMO)"]:
        match = [r for r in benchmark_records_primary if r["model_name"] == ret_name]
        if match:
            rec = dict(match[0])
            rec["device"] = "cpu"
            benchmark_records_cpu.append(rec)

    # 6. Noise Robustness Experiment (Hypothesis H1 Verification)
    print("\n--- Running Noise Robustness Benchmark (0%, 10%, 20%, 30% Label Corruption) ---")
    noise_results = run_noise_experiment(
        train_embeddings=train_emb_used,
        train_labels=train_labels,
        train_paths=train_paths,
        test_embeddings=test_emb_used,
        test_labels=test_labels,
        noise_rates=[0.0, 0.10, 0.20, 0.30],
        seed=seed,
        temp_dir=vector_db_dir / "noise_exp"
    )
    print("Noise Experiment Results:")
    for rate, n_acc, i_acc in zip(noise_results["noise_rates"], noise_results["naive_accuracy"], noise_results["imo_accuracy"]):
        print(f"  Noise Rate {int(rate*100)}%: Naive Acc = {n_acc:.4f} | Proposed IMO Acc = {i_acc:.4f} (Delta: {i_acc - n_acc:+.4f})")

    # 7. Save Results and Dataframes
    df_primary = pd.DataFrame(benchmark_records_primary)
    df_cpu = pd.DataFrame(benchmark_records_cpu)

    primary_csv_path = artifacts_dir / f"benchmark_results_{primary_device.replace(':', '_')}.csv"
    cpu_csv_path = artifacts_dir / "benchmark_results_cpu.csv"
    df_primary.to_csv(primary_csv_path, index=False)
    df_cpu.to_csv(cpu_csv_path, index=False)

    noise_json_path = artifacts_dir / "noise_experiment_results.json"
    with open(noise_json_path, "w", encoding="utf-8") as f:
        json.dump(noise_results, f, indent=2)

    # 8. Generate Visualizations
    print("\n--- Generating Publication-Ready Comparative Figures ---")
    plot_comparative_metrics(
        df_primary,
        artifacts_dir / "multi_model_accuracy_comparison.png",
        title=f"Multi-Model Predictive Performance ({primary_device.upper()})"
    )
    plot_latency_accuracy_tradeoff(
        df_primary,
        artifacts_dir / "latency_vs_accuracy_tradeoff.png",
        title="Accuracy vs Latency Efficiency Frontier"
    )
    if "cuda" in primary_device:
        plot_cpu_vs_gpu_speedup(
            df_cpu,
            df_primary,
            artifacts_dir / "cpu_vs_gpu_speedup.png",
            title="CPU vs GPU Inference Latency Comparison"
        )
    plot_noise_robustness(
        noise_results,
        artifacts_dir / "noise_robustness_curve.png",
        title="Resistance to Label Noise: Naive vs Proposed IMO"
    )

    # 9. Generate Comprehensive Summary JSON
    summary = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "total_benchmark_time_seconds": time.time() - t_start,
            "primary_device": primary_device,
            "hardware": get_hardware_info(primary_device),
            "fine_tuning_enabled": fine_tune,
            "epochs_evaluated": actual_epochs,
        },
        "dataset": {
            "dataset_directory": str(dataset_dir),
            "total_samples": len(train_labels) + len(val_labels) + len(test_labels),
            "train_samples": len(train_labels),
            "val_samples": len(val_labels),
            "test_samples": len(test_labels),
        },
        "retrieval_tuning": tuning_summary,
        "primary_benchmark_table": df_primary.to_dict(orient="records"),
        "cpu_benchmark_table": df_cpu.to_dict(orient="records"),
        "noise_experiment": noise_results,
        "imo_curation_summary": imo.get_summary(),
    }

    summary_path = artifacts_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBenchmark completed successfully in {summary['metadata']['total_benchmark_time_seconds']:.2f} seconds.")
    print(f"Artifacts saved to: {artifacts_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Unified Multi-Model Thesis Benchmark.")
    parser.add_argument("--seed", type=int, default=777, help="Deterministic random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per baseline.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to image dataset directory.")
    parser.add_argument("--no-fine-tune", action="store_true", help="Disable backbone fine-tuning.")
    parser.add_argument("--quick", action="store_true", help="Run with reduced epochs for fast validation.")
    args = parser.parse_args()

    run_benchmark(
        seed=args.seed,
        device_mode=args.device,
        quick_mode=args.quick,
        epochs=args.epochs,
        fine_tune=not args.no_fine_tune,
        data_dir=args.data_dir
    )
