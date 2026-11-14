"""
Computational Resource Profiler.
Measures latency (mean, median, p95), throughput, RAM (Windows ctypes), VRAM, and hardware attributes.
"""

import ctypes
from ctypes import wintypes
import os
import platform
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_current_process_ram_mb() -> float:
    """Return current process resident RAM (working set) in megabytes."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize / (1024 ** 2))
    except Exception:
        pass
    return 0.0


def get_peak_process_ram_mb() -> float:
    """Return peak process resident RAM in megabytes."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float(counters.PeakWorkingSetSize / (1024 ** 2))
    except Exception:
        pass
    return 0.0


def get_hardware_info(device_str: str = "cpu") -> Dict[str, Any]:
    """Inspect and return current hardware environment details."""
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu_arch": platform.machine(),
        "cpu_count_logical": os.cpu_count() or 1,
        "device_used": device_str,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if "cuda" in device_str:
            dev_idx = 0
            if ":" in device_str:
                try:
                    dev_idx = int(device_str.split(":")[1])
                except Exception:
                    dev_idx = 0
            props = torch.cuda.get_device_properties(dev_idx)
            info["active_gpu_name"] = props.name
            info["active_gpu_total_vram_mb"] = float(props.total_memory / (1024 ** 2))
    return info


class ResourceProfiler:
    """Context and benchmarking tool for timing and hardware tracking."""

    @staticmethod
    def benchmark_inference(
        fn: Callable,
        warmup_runs: int = 5,
        timed_runs: int = 20,
        device: str = "cpu",
        num_samples_per_run: int = 1
    ) -> Dict[str, float]:
        """
        Run warm-up and timed inference iterations.
        Computes mean, median, p95 latency in ms, and throughput in samples/sec.
        """
        # Warmup
        for _ in range(warmup_runs):
            fn()
            if "cuda" in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()

        latencies_ms: List[float] = []
        ram_before = get_current_process_ram_mb()

        if "cuda" in str(device) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(timed_runs):
            t0 = time.perf_counter()
            fn()
            if "cuda" in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        ram_after = get_current_process_ram_mb()
        peak_vram_mb = 0.0
        if "cuda" in str(device) and torch.cuda.is_available():
            peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        latencies = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        median_lat = float(np.median(latencies))

        per_sample_lat_ms = mean_lat / max(1, num_samples_per_run)
        throughput = (num_samples_per_run / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

        return {
            "mean_batch_latency_ms": mean_lat,
            "median_batch_latency_ms": median_lat,
            "p95_batch_latency_ms": p95_lat,
            "avg_ms_per_sample": per_sample_lat_ms,
            "throughput_samples_per_sec": throughput,
            "peak_vram_mb": peak_vram_mb,
            "ram_used_mb": max(0.0, ram_after - ram_before),
            "peak_ram_mb": get_peak_process_ram_mb(),
        }
