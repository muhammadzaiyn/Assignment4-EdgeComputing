"""Hardware-aware evaluation of candidate architectures (Question 2, requirements 5 and 6).

Before a candidate is trained, `profile_architecture()` builds it and *measures*:

    cpu_latency_ms  median batch-1 FP32 latency on the CPU with 1 thread (PyTorch eager)
    size_mb         size of the serialised FP32 state_dict on disk
    peak_mem_mb     peak inference memory of a batch-1 forward pass (CUDA allocator, exact)
    params, macs    for context and for the report

`DeploymentConstraints.check()` then decides whether the candidate can be deployed on the
target edge device. Candidates that violate any limit are rejected *before* training, which
saves the GPU time that would otherwise be wasted on undeployable networks.

Single-thread CPU latency is used deliberately: many edge devices (micro-controllers,
low-end ARM boards, a busy phone core) effectively give one core to the model, and a single
thread also makes the measurement far less sensitive to background load on the PC.
"""
from dataclasses import asdict, dataclass

import torch

from ..metrics import count_macs, count_params, measure_cpu_latency, measure_peak_memory_cuda, state_dict_size_mb
from .generator import build_network


@dataclass
class DeploymentConstraints:
    """Upper limits a deployable architecture must satisfy (all measured, see above)."""
    max_latency_ms: float = 2.0     # batch-1 CPU latency, 1 thread
    max_size_mb: float = 1.0        # FP32 weights on disk
    max_peak_mem_mb: float = 8.0    # peak inference memory, batch 1

    def violations(self, m: dict) -> list:
        """Signed constraint values used by Optuna (<= 0 means satisfied)."""
        mem = m["peak_mem_mb"] if m.get("peak_mem_mb") is not None else 0.0
        return [m["cpu_latency_ms"] - self.max_latency_ms,
                m["size_mb"] - self.max_size_mb,
                (mem - self.max_peak_mem_mb) if m.get("peak_mem_mb") is not None else 0.0]

    def check(self, m: dict):
        """Return (feasible, list of human-readable reasons for rejection)."""
        reasons = []
        if m["cpu_latency_ms"] > self.max_latency_ms:
            reasons.append(f"latency {m['cpu_latency_ms']:.2f} ms > {self.max_latency_ms} ms")
        if m["size_mb"] > self.max_size_mb:
            reasons.append(f"size {m['size_mb']:.3f} MB > {self.max_size_mb} MB")
        if m.get("peak_mem_mb") is not None and m["peak_mem_mb"] > self.max_peak_mem_mb:
            reasons.append(f"peak memory {m['peak_mem_mb']:.2f} MB > {self.max_peak_mem_mb} MB")
        return len(reasons) == 0, reasons

    def to_dict(self):
        return asdict(self)


def profile_architecture(config: dict, num_classes: int = 10, latency_runs: int = 60,
                         warmup: int = 15) -> dict:
    """Build the candidate on the CPU and measure its deployment cost."""
    model = build_network(config, num_classes).eval()
    res = config["resolution"]
    lat = measure_cpu_latency(model, res, threads=1, warmup=warmup, runs=latency_runs)
    mem = measure_peak_memory_cuda(model, res)
    return {
        "params": count_params(model),
        "macs_m": count_macs(model, res) / 1e6,
        "size_mb": state_dict_size_mb(model),
        "cpu_latency_ms": lat["p50_ms"],
        "cpu_latency_p90_ms": lat["p90_ms"],
        "peak_mem_mb": mem["peak_mem_mb"],
        "activation_mem_mb": mem["activation_mem_mb"],
    }
