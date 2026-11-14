"""
Visualization module for Thesis Multi-Model Benchmarking.
Generates publication-quality comparative figures.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_comparative_metrics(
    results_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "Multi-Model Predictive Performance Comparison"
):
    """Generate comparative bar chart for Accuracy, Precision, Recall, F1."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    plot_df = results_df[["model_name"] + metrics].set_index("model_name")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_xlabel("Model Architecture / Approach", fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latency_accuracy_tradeoff(
    results_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "Accuracy vs Latency Trade-off (Efficiency Frontier)"
):
    """Plot scatter of Inference Latency (ms) vs Test Accuracy."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for _, row in results_df.iterrows():
        name = row["model_name"]
        acc = row["accuracy"]
        lat = row["avg_ms_per_sample"]
        color = "#1f77b4"
        marker = "o"
        if "IMO" in name:
            color = "#2ca02c"
            marker = "*"
        elif "Retrieval" in name:
            color = "#ff7f0e"
            marker = "s"

        ax.scatter(lat, acc, s=150, color=color, marker=marker, edgecolors="black", zorder=4)
        ax.annotate(
            name,
            (lat, acc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if "IMO" in name else "normal"
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Inference Latency per Sample (ms) [Log Scale]", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cpu_vs_gpu_speedup(
    cpu_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    output_path: Path | str,
    title: str = "Hardware Profiling: CPU vs GPU Inference Latency"
):
    """Plot side-by-side comparison of CPU vs GPU latencies."""
    merged = pd.merge(
        cpu_df[["model_name", "avg_ms_per_sample"]].rename(columns={"avg_ms_per_sample": "CPU Latency (ms)"}),
        gpu_df[["model_name", "avg_ms_per_sample"]].rename(columns={"avg_ms_per_sample": "GPU Latency (ms)"}),
        on="model_name"
    ).set_index("model_name")

    fig, ax = plt.subplots(figsize=(10, 5))
    merged.plot(kind="bar", ax=ax, color=["#d95f02", "#1b9e77"], width=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Latency per Sample (ms) [Log Scale]", fontsize=11)
    ax.set_xlabel("Architecture", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_noise_robustness(
    noise_results: Dict[str, Any],
    output_path: Path | str,
    title: str = "Resistance to Injected Label Noise: Naive vs Proposed IMO"
):
    """Plot test accuracy degradation under increasing noise rates."""
    rates = [r * 100 for r in noise_results["noise_rates"]]
    naive_acc = noise_results["naive_accuracy"]
    imo_acc = noise_results["imo_accuracy"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rates, naive_acc, marker="o", linewidth=2.2, color="#d95f02", label="Naive Retrieval (ChromaDB)")
    ax.plot(rates, imo_acc, marker="s", linewidth=2.5, color="#1b9e77", label="Proposed Framework (ChromaDB + IMO)")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Injected Label Noise (%)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
