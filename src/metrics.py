"""Hardware measurements shared by the NAS (Q2) and the final benchmark (Q1 + Q2).

All numbers are *measured* on this machine, not estimated from FLOPs:

* CPU latency      - wall-clock time of one batch-1 forward pass with PyTorch on the CPU
                     (fixed number of threads, warm-up runs first, median of many runs).
* storage size     - bytes of the serialised state_dict (what is written to disk) and,
                     where available, of the exported ONNX file.
* peak memory      - peak bytes allocated during a batch-1 inference. Exact peak tracking is
                     available through the CUDA caching allocator, so it is measured on the GPU
                     ("where supported"). For the CPU we additionally report the growth of the
                     process resident set size (RSS), which is a coarser but still real number.
* MACs             - multiply-accumulate count from forward hooks on Conv2d / Linear layers
                     (reported for context; it is not used for the constraints).
"""
import gc
import io
import statistics
import time

import numpy as np
import psutil
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- size
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model: nn.Module) -> float:
    """Exact on-disk size (MB) of `torch.save(model.state_dict())`, measured in memory."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


# ----------------------------------------------------------------------------- MACs
@torch.no_grad()
def count_macs(model: nn.Module, resolution: int = 32) -> int:
    """Multiply-accumulate operations of one forward pass (Conv2d + Linear layers only)."""
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        k = m.kernel_size[0] * m.kernel_size[1]
        macs += out.numel() * (m.in_channels // m.groups) * k

    def linear_hook(m, inp, out):
        nonlocal macs
        macs += out.numel() * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(1, 3, resolution, resolution, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


# ----------------------------------------------------------------------------- latency
def latency_stats(times_s):
    """Summary statistics (milliseconds) of a list of per-run timings in seconds."""
    ms = np.asarray(times_s) * 1000.0
    return {"mean_ms": float(ms.mean()), "std_ms": float(ms.std()),
            "p50_ms": float(np.percentile(ms, 50)), "p90_ms": float(np.percentile(ms, 90)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "min_ms": float(ms.min())}


@torch.inference_mode()
def measure_cpu_latency(model: nn.Module, resolution: int, threads: int = 1, warmup: int = 20,
                        runs: int = 100, inputs=None) -> dict:
    """Batch-1 CPU latency of a PyTorch model (FP32, eval mode, `threads` intra-op threads).

    The model is copied to the CPU so the caller's (possibly GPU) model is left untouched.
    `inputs` may supply real images (N, 3, R, R); otherwise random tensors are used, which
    give the same timing for these convolutional networks.
    """
    import copy
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        m = copy.deepcopy(model).cpu().eval()
        if inputs is None:
            inputs = torch.randn(8, 3, resolution, resolution)
        n = len(inputs)
        for i in range(warmup):
            m(inputs[i % n:i % n + 1])
        times = []
        for i in range(runs):
            x = inputs[i % n:i % n + 1]
            t0 = time.perf_counter()
            m(x)
            times.append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(old_threads)
    return latency_stats(times)


def measure_ort_latency(session, x_np: np.ndarray, warmup: int = 20, runs: int = 100) -> dict:
    """Batch-1 latency of an ONNX Runtime session; `x_np` holds N sample inputs (N, 3, R, R)."""
    name = session.get_inputs()[0].name
    n = len(x_np)
    for i in range(warmup):
        session.run(None, {name: x_np[i % n:i % n + 1]})
    times = []
    for i in range(runs):
        x = x_np[i % n:i % n + 1]
        t0 = time.perf_counter()
        session.run(None, {name: x})
        times.append(time.perf_counter() - t0)
    return latency_stats(times)


# ----------------------------------------------------------------------------- peak memory
@torch.inference_mode()
def measure_peak_memory_cuda(model: nn.Module, resolution: int) -> dict:
    """Peak GPU memory (MB) of a batch-1 forward pass: weights + activations + workspace.

    Returns None values if CUDA is not available (the "where supported" case).
    """
    if not torch.cuda.is_available():
        return {"peak_mem_mb": None, "activation_mem_mb": None}
    import copy
    # cuDNN autotuning (benchmark mode) temporarily allocates large trial workspaces that have
    # nothing to do with deployment, so it is switched off for this measurement
    old_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        m = copy.deepcopy(model).cuda().eval()
        x = torch.randn(1, 3, resolution, resolution, device="cuda")
        m(x)                                   # warm-up: one-time CUDA/cuDNN initialisation
        torch.cuda.synchronize()
        weights = torch.cuda.memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        m(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del m, x
        torch.cuda.empty_cache()
    finally:
        torch.backends.cudnn.benchmark = old_bench
    return {"peak_mem_mb": peak / 1e6, "activation_mem_mb": (peak - weights) / 1e6}


@torch.inference_mode()
def measure_cpu_rss_delta(model: nn.Module, resolution: int) -> float:
    """Growth of the process resident memory (MB) while loading + running the model on the CPU.

    This is coarse (allocator caching, other threads), so it is recorded for information only;
    the deployment constraint uses the exact CUDA allocator peak.
    """
    import copy
    gc.collect()
    proc = psutil.Process()
    before = proc.memory_info().rss
    m = copy.deepcopy(model).cpu().eval()
    m(torch.randn(1, 3, resolution, resolution))
    after = proc.memory_info().rss
    del m
    return max(0.0, (after - before) / 1e6)


def median_of(dicts, key):
    return statistics.median(d[key] for d in dicts)
