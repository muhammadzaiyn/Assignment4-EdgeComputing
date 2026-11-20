"""Run the complete Assignment 4 pipeline in order.

    python run_all.py --full     every experiment with the settings used for the submitted results
                                 (about 3-4 hours on an RTX 5060 Ti; most of it is GPU training)
    python run_all.py --quick    tiny version of every step (a few minutes) to test the plumbing;
                                 its outputs carry a "_quick" suffix and never overwrite real results
    python run_all.py --full --from 7      resume the pipeline from step 7

Steps
   1  scripts/01_prepare_data.py            download CIFAR-10/100, check the data pipeline
   2  scripts/02_train_teacher.py           ResNet-50 teacher on CIFAR-100
   3  scripts/03_train_students_baseline.py three students without KD
   4  scripts/04b_calibrate_feature_weight.py   choose the feature-loss weight beta
      scripts/04_train_students_kd.py       three students with soft + feature KD
   5  scripts/04_train_students_kd.py --mode soft   ablation: soft-target KD only
   6  scripts/05_nas_search.py              hardware-aware NAS (+ automatic training of the best)
   7  scripts/07_export_onnx.py             ONNX export + parity checks
   8  scripts/08_inference.py / 09_onnx_inference.py   inference demos and test-set evaluation
   9  scripts/10_benchmark.py               automated benchmark -> CSV
  10  scripts/11_make_figures.py            figures + console captures
  11  scripts/12_make_report.py             technical report (PDF + DOCX)
  12  scripts/13_verify_deliverables.py     PASS/FAIL check of every deliverable
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "scripts"


def steps(quick):
    q = ["--quick"] if quick else []
    plan = [
        (1, [S / "01_prepare_data.py"]),
        (2, [S / "02_train_teacher.py", *q]),
        (3, [S / "03_train_students_baseline.py", *q]),
        *([] if quick else [(4, [S / "04b_calibrate_feature_weight.py"])]),
        (4, [S / "04_train_students_kd.py", *q]),
        (5, [S / "04_train_students_kd.py", "--mode", "soft", *q]),
        (6, [S / "05_nas_search.py", "--fresh", *q]),
    ]
    if not quick:   # the remaining steps work on the final (non-quick) checkpoints
        plan += [
            (7, [S / "07_export_onnx.py"]),
            (8, [S / "08_inference.py", "--model", "all", "--num-samples", "5"]),
            (8, [S / "08_inference.py", "--model", "all", "--evaluate"]),
            (8, [S / "09_onnx_inference.py", "--model", "all", "--num-samples", "3"]),
            (8, [S / "09_onnx_inference.py", "--model", "all", "--evaluate"]),
            (9, [S / "10_benchmark.py"]),
            (10, [S / "11_make_figures.py"]),
            (11, [S / "12_make_report.py"]),
            (12, [S / "13_verify_deliverables.py"]),
        ]
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--quick", action="store_true")
    p.add_argument("--from", dest="start", type=int, default=1, help="first step to run")
    args = p.parse_args()

    t_all = time.time()
    for num, cmd in steps(args.quick):
        if num < args.start:
            continue
        line = " ".join(str(c.relative_to(ROOT)) if isinstance(c, Path) else c for c in cmd)
        print(f"\n{'#' * 78}\n# step {num}: python {line}\n{'#' * 78}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, check=True)
        print(f"# step {num} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nall steps finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
