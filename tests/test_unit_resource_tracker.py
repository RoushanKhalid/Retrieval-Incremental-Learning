"""
Unit tests for computational resource tracking:
64-bit Windows ctypes RAM measurement, hardware info inspection, and inference benchmarking.
"""

from tcontext.evaluation.resource_tracker import (
    ResourceProfiler,
    get_current_process_ram_mb,
    get_hardware_info,
    get_peak_process_ram_mb,
)


def test_ctypes_ram_measurement_nonzero():
    """Verify that 64-bit ctypes process memory counters return valid non-zero values."""
    ram = get_current_process_ram_mb()
    peak_ram = get_peak_process_ram_mb()

    # The Python runtime process should occupy at least 10 MB and less than 32 GB
    assert ram > 10.0, f"Expected resident RAM > 10 MB, got {ram}"
    assert peak_ram >= ram, f"Peak RAM ({peak_ram}) must be >= current RAM ({ram})"


def test_hardware_info():
    """Verify system hardware inspection returns required platform keys."""
    info = get_hardware_info("cpu")
    assert "os" in info
    assert "cpu_arch" in info
    assert "cpu_count_logical" in info
    assert info["cpu_count_logical"] >= 1
    assert "cuda_available" in info


def test_resource_profiler_timing():
    """Verify ResourceProfiler calculates latency and throughput accurately."""
    def dummy_fn():
        import time
        time.sleep(0.002)

    res = ResourceProfiler.benchmark_inference(
        fn=dummy_fn,
        warmup_runs=1,
        timed_runs=3,
        device="cpu",
        num_samples_per_run=10
    )

    assert "mean_batch_latency_ms" in res
    assert "throughput_samples_per_sec" in res
    assert res["mean_batch_latency_ms"] >= 1.5  # Slept 2ms
    assert res["throughput_samples_per_sec"] > 0.0
