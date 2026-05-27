from __future__ import annotations

import time
from typing import Any, Callable

import torch

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
