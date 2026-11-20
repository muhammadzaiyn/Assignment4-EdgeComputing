"""Automatic self-check of all 13 deliverables listed in the assignment.

For each deliverable it checks that the required files exist and, where it is cheap, that
they actually work: checkpoints are loaded and run, ONNX files are loaded by ONNX Runtime,
CSV files have rows, the NAS model's measured cost respects the constraints, etc.
It prints a PASS/FAIL table and writes results/deliverables_check.csv.

Run:  python scripts/13_verify_deliverables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.common import CKPT_DIR, LOG_DIR, ONNX_DIR, REPORT_DIR, RESULTS_DIR, ROOT, Tee, banner, load_json, write_csv
from src.infer_utils import ALL_MODELS, load_model, ort_session

checks = []


def check(num, deliverable, desc, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # any crash is a failed check
        ok, detail = False, f"{type(e).__name__}: {e}"
    checks.append({"deliverable": num, "name": deliverable, "check": desc, "result": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {num:>2} {deliverable:24s} {desc:52s} {detail}")


def runs(name):
    model, info = load_model(name)
    out = model(torch.randn(2, 3, info.resolution, info.resolution))
    return out.shape == (2, len(info.classes)), f"output {tuple(out.shape)}"


def exists(*paths):
    missing = [str(Path(p).relative_to(ROOT)) for p in paths if not Path(p).exists()]
    return not missing, "ok" if not missing else f"missing {missing}"


def csv_rows(path, min_rows=1):
    d = pd.read_csv(path)
    return len(d) >= min_rows, f"{len(d)} rows"


def main():
    with Tee(LOG_DIR / "13_verify_deliverables.log"):
        banner("Deliverables self-check")
        S = ROOT / "scripts"
        archs = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
        check(1, "Teacher model", "training script + checkpoint exist",
              lambda: exists(S / "02_train_teacher.py", CKPT_DIR / "teacher" / "resnet50_cifar100.pth"))
        check(1, "Teacher model", "checkpoint loads and runs", lambda: runs("teacher"))
        check(1, "Teacher model", "test accuracy recorded (FP32 benchmark)",
              lambda: (True, f"test top-1 {pd.read_csv(RESULTS_DIR / 'benchmark_summary.csv').set_index('model').loc['teacher', 'test_top1']:.2f}%"))
        check(2, "Student architectures", "MobileNetV2 / SqueezeNet / EfficientNet-Lite0 code",
              lambda: exists(ROOT / "src" / "models" / "students.py"))
        for a in archs:
            check(3, "Baseline students", f"{a}_baseline loads and runs", lambda a=a: runs(f"{a}_baseline"))
        check(4, "Knowledge distillation", "KD framework + training script",
              lambda: exists(ROOT / "src" / "kd.py", S / "04_train_students_kd.py"))
        for a in archs:
            check(4, "Knowledge distillation", f"{a}_kd loads and runs", lambda a=a: runs(f"{a}_kd"))
        check(5, "NAS search space", "search space + generator code",
              lambda: exists(ROOT / "src" / "nas" / "search_space.py", ROOT / "src" / "nas" / "generator.py"))
        check(6, "NAS algorithm", "Optuna search script + hardware profiler",
              lambda: exists(S / "05_nas_search.py", ROOT / "src" / "nas" / "hardware.py"))
        check(7, "NAS results", "best architecture JSON + history CSV/JSON",
              lambda: exists(RESULTS_DIR / "nas" / "best_architecture.json", RESULTS_DIR / "nas" / "search_history.csv",
                             RESULTS_DIR / "nas" / "search_history.json"))
        check(7, "NAS results", "search history has trials",
              lambda: csv_rows(RESULTS_DIR / "nas" / "search_history.csv", 20))
        check(8, "Optimized architecture", "nas_best checkpoint loads and runs", lambda: runs("nas_best"))
        check(8, "Optimized architecture", "final model meets deployment constraints",
              lambda: (load_json(RESULTS_DIR / "nas" / "final_results.json")["satisfies_constraints"],
                       str(load_json(RESULTS_DIR / "nas" / "final_results.json")["hardware"]["cpu_latency_ms"])[:5] + " ms"))
        check(9, "Inference pipeline", "PyTorch + ONNX inference scripts",
              lambda: exists(S / "08_inference.py", S / "09_onnx_inference.py", ROOT / "src" / "infer_utils.py"))
        for n in ALL_MODELS:
            check(10, "ONNX deployment", f"{n}.onnx runs in ONNX Runtime",
                  lambda n=n: (ort_session(ONNX_DIR / f"{n}.onnx").get_providers()[0] == "CPUExecutionProvider", "CPU EP"))
        check(10, "ONNX deployment", "export parity checks passed",
              lambda: (bool(pd.read_csv(RESULTS_DIR / "onnx_export.csv").parity_ok.all()), "all models"))
        check(11, "Benchmarking", "benchmark script + summary CSV",
              lambda: exists(S / "10_benchmark.py", RESULTS_DIR / "benchmark_summary.csv"))
        check(11, "Benchmarking", "summary covers all models",
              lambda: csv_rows(RESULTS_DIR / "benchmark_summary.csv", len(ALL_MODELS) + 1))
        check(12, "Technical report", "PDF report exists", lambda: exists(REPORT_DIR / "Technical_Report.pdf"))
        check(12, "Technical report", "screenshots / figures exist",
              lambda: (len(list((REPORT_DIR / "screenshots").glob("*.png"))) >= 5, "console captures"))
        check(13, "README", "README with install + run steps", lambda: exists(ROOT / "README.md", ROOT / "requirements.txt"))

        write_csv(checks, RESULTS_DIR / "deliverables_check.csv")
        n_pass = sum(c["result"] == "PASS" for c in checks)
        print(f"\n{n_pass}/{len(checks)} checks passed")
        if n_pass != len(checks):
            sys.exit(1)


if __name__ == "__main__":
    main()
