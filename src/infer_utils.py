"""Unified model registry for inference, ONNX export and benchmarking (Q1.6, Q2.8).

Every trained, deployable model has one short name:

    mobilenetv2_baseline        mobilenetv2_kd           (CIFAR-100 students)
    squeezenet_baseline         squeezenet_kd
    efficientnet_lite0_baseline efficientnet_lite0_kd
    nas_best                                             (CIFAR-10 NAS model)
    teacher                                              (CIFAR-100 ResNet-50, reference only)

`load_model(name)` rebuilds the right architecture from the config stored in the checkpoint and
returns it together with the metadata the caller needs (dataset, class names, input size).
`predict()` runs a forward pass and returns class probabilities. Both the PyTorch inference
script and the ONNX Runtime script use this file, so they always agree on preprocessing.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .common import CKPT_DIR, ONNX_DIR
from .data import class_names
from .models import DISPLAY_NAMES, STUDENTS, build_student, build_teacher
from .nas import build_network

STUDENT_MODELS = [f"{a}_{m}" for a in STUDENTS for m in ("baseline", "kd")]
ALL_MODELS = STUDENT_MODELS + ["nas_best"]          # deployable models (inference + ONNX)
REFERENCE_MODELS = ["teacher"]                      # evaluated in the benchmark, not deployed


@dataclass
class ModelInfo:
    name: str
    display: str
    dataset: str
    resolution: int
    checkpoint: Path
    classes: list = field(default_factory=list)
    onnx_path: Path = None


def checkpoint_path(name: str) -> Path:
    if name == "teacher":
        return CKPT_DIR / "teacher" / "resnet50_cifar100.pth"
    if name == "nas_best":
        return CKPT_DIR / "nas" / "nas_best.pth"
    arch, mode = name.rsplit("_", 1)
    return CKPT_DIR / mode / f"{arch}_{mode}.pth"


def display_name(name: str) -> str:
    if name == "teacher":
        return DISPLAY_NAMES["resnet50"]
    if name == "nas_best":
        return "NAS-best (CIFAR-10)"
    arch, mode = name.rsplit("_", 1)
    return f"{DISPLAY_NAMES[arch]} ({'KD' if mode == 'kd' else 'baseline'})"


def load_model(name: str, device="cpu"):
    """Rebuild a trained model by name. Returns (model in eval mode, ModelInfo)."""
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint for '{name}' not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if name == "teacher":
        model, dataset, res = build_teacher(cfg["num_classes"], pretrained=False), "cifar100", 32
    elif name == "nas_best":
        model, dataset, res = build_network(cfg["nas_config"], cfg["num_classes"]), "cifar10", cfg["resolution"]
    else:
        model, dataset, res = build_student(cfg["arch"], cfg["num_classes"]), "cifar100", 32
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    info = ModelInfo(name=name, display=display_name(name), dataset=dataset, resolution=res,
                     checkpoint=path, classes=class_names(dataset), onnx_path=ONNX_DIR / f"{name}.onnx")
    return model, info


@torch.inference_mode()
def predict(model, x: torch.Tensor) -> torch.Tensor:
    """Class probabilities for a batch of preprocessed images."""
    dev = next(model.parameters()).device
    return torch.softmax(model(x.to(dev)).float(), dim=1).cpu()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ort_session(path: Path, threads: int = 1):
    """ONNX Runtime CPU session with a fixed number of threads (1 = edge-like single core)."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
