from __future__ import annotations

import time
from statistics import pstdev
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover
    EmissionsTracker = None

try:
    from thop import profile
except ImportError:  # pragma: no cover
    profile = None


def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    runs: int = 30,
) -> float:
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def benchmark_dataloader_latency(
    infer_fn: Callable[[torch.Tensor], Any],
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = 3,
    timed_batches: int | None = None,
) -> dict[str, float]:
    per_sample_latencies: list[float] = []
    total_elapsed = 0.0
    total_samples = 0
    measured_batches = 0

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(dataloader):
            images = images.to(device)

            if batch_index < warmup_batches:
                _ = infer_fn(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                continue

            if timed_batches is not None and measured_batches >= timed_batches:
                break

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = infer_fn(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            batch_size = images.size(0)
            total_elapsed += elapsed
            total_samples += batch_size
            per_sample_latencies.append((elapsed / batch_size) * 1000.0)
            measured_batches += 1

    if total_samples == 0:
        return {"latency_ms": 0.0, "latency_std_ms": 0.0}

    mean_latency_ms = (total_elapsed / total_samples) * 1000.0
    std_latency_ms = pstdev(per_sample_latencies) if len(per_sample_latencies) > 1 else 0.0
    return {
        "latency_ms": mean_latency_ms,
        "latency_std_ms": std_latency_ms,
    }


def estimate_flops(model: torch.nn.Module, sample: torch.Tensor) -> float | None:
    if profile is None:
        return None
    device = next(model.parameters()).device
    sample = sample.to(device)
    flops, _ = profile(model, inputs=(sample,), verbose=False)
    return float(flops)


def track_emissions(fn: Callable[[], Any], project_name: str) -> tuple[Any, float | None]:
    if EmissionsTracker is None:
        return fn(), None

    tracker = EmissionsTracker(project_name=project_name, save_to_file=False, log_level="error")
    tracker.start()
    try:
        result = fn()
    finally:
        emissions = tracker.stop()
    return result, emissions
