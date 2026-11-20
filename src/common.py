"""Shared helpers used by every script: project paths, seeding, logging and small I/O utilities.

Keeping these in one place means every experiment writes its outputs to the same folders with
the same conventions, which makes the results easy to find and the report easy to rebuild.
"""
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # project root (folder that holds src/)
DATA_DIR = ROOT / "data"                            # dataset cache (never committed)
CKPT_DIR = ROOT / "checkpoints"                     # trained models
ONNX_DIR = ROOT / "onnx_models"                     # exported ONNX graphs
RESULTS_DIR = ROOT / "results"                      # CSV / JSON results
LOG_DIR = RESULTS_DIR / "logs"                      # full console logs of every run
FIG_DIR = RESULTS_DIR / "figures"                   # plots used in the report
REPORT_DIR = ROOT / "report"                        # technical report + screenshots

SEED = 42                                           # single global seed for every experiment

for _d in (DATA_DIR, CKPT_DIR, ONNX_DIR, RESULTS_DIR, LOG_DIR, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- reproducibility
def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    `deterministic=True` additionally forces deterministic cuDNN kernels. It is off by default
    because it slows training noticeably; with a fixed seed the runs are still repeatable to
    within normal GPU floating-point noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    """Training device: the NVIDIA GPU if present, otherwise the CPU (slow, but still works)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- logging
class Tee:
    """Duplicate everything printed to the console into a log file.

    Every script wraps its run in `with Tee(LOG_DIR / "<name>.log"):` so that the complete console
    output is kept next to the results (these logs are also used for the report's console figures).
    """

    def __init__(self, path: Path, mode: str = "w"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode

    def __enter__(self):
        self.file = open(self.path, self.mode, encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"# log started {stamp} | command: python {' '.join(sys.argv)}")
        return self

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc, tb, file=self.file)
        sys.stdout = self.stdout
        self.file.close()
        return False


def banner(title: str) -> None:
    """Print a clearly visible section header in the console / log."""
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


# ----------------------------------------------------------------------------- file helpers
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    """Make NumPy / torch scalars and paths JSON-serialisable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_csv(rows, path: Path, fieldnames=None) -> None:
    """Write a list of dicts to CSV (column order = first row's keys unless given)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def file_size_mb(path: Path) -> float:
    """Size of a file on disk in megabytes (10^6 bytes)."""
    return Path(path).stat().st_size / 1e6


def rel(path: Path) -> str:
    """Path relative to the project root, with forward slashes (for logs and reports)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------------- environment
def system_info() -> dict:
    """Hardware / software description written next to the results for reproducibility."""
    import psutil
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": SEED,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 1),
                    cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    for mod in ("torchvision", "onnx", "onnxruntime", "optuna", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # package not installed
            info[mod] = None
    return info


def _cpu_name() -> str:
    """Readable CPU model name (platform.processor() is vague on Windows)."""
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


class Timer:
    """Tiny context manager: `with Timer() as t: ...; print(t.seconds)`."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False
