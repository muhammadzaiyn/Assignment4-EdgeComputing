"""Q1.5 - Baseline training of the three students WITHOUT knowledge distillation.  (deliverable 3)

Each student (MobileNetV2, SqueezeNet 1.1, EfficientNet-Lite0) is trained from random
initialisation on CIFAR-100 with plain cross-entropy on the ground-truth labels. These are the
reference points that tell us how much the distillation in 04_train_students_kd.py helps.

The optimiser, schedule, augmentation, epochs and seed are *identical* to the KD runs (both
scripts call the same `fit()` engine); the only difference is the loss.

Outputs (per student <arch>)
  checkpoints/baseline/<arch>_baseline.pth         best-validation checkpoint
  results/students/<arch>_baseline_history.csv     per-epoch log
  results/students/<arch>_baseline.json            final val/test accuracy, params, time
  results/logs/03_baseline_<arch>.log              console output

Run:  python scripts/03_train_students_baseline.py                    (all three, 60 epochs)
      python scripts/03_train_students_baseline.py --arch squeezenet  (one student)
      python scripts/03_train_students_baseline.py --quick            (smoke test)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common import CKPT_DIR, LOG_DIR, RESULTS_DIR, SEED, Tee, banner, get_device, save_json, set_seed
from src.data import get_loaders
from src.models import DISPLAY_NAMES, STUDENTS, build_student
from src.train_utils import count_params, evaluate, fit, load_checkpoint

# shared student hyper-parameters (also imported by the KD script so both stay identical)
STUDENT_HPARAMS = {"epochs": 60, "lr": 0.1, "weight_decay": 5e-4, "batch_size": 128, "warmup_epochs": 1}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    p.add_argument("--epochs", type=int, default=STUDENT_HPARAMS["epochs"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def train_baseline(arch, args, device):
    suffix = "_quick" if args.quick else ""
    set_seed(SEED)
    hp = dict(STUDENT_HPARAMS, epochs=2 if args.quick else args.epochs)
    train, val, test, classes = get_loaders("cifar100", device, batch_size=hp["batch_size"],
                                            train_subset=5000 if args.quick else None)
    model = build_student(arch, num_classes=len(classes))
    banner(f"Baseline (no KD): {DISPLAY_NAMES[arch]} on CIFAR-100")
    print(f"params {count_params(model):,} | epochs {hp['epochs']} | lr {hp['lr']} | "
          f"wd {hp['weight_decay']} | batch {hp['batch_size']} | loss: cross-entropy (hard labels)")

    config = {"arch": arch, "num_classes": len(classes), "dataset": "cifar100", "mode": "baseline",
              "kd": None, "seed": SEED, **hp}
    ckpt = CKPT_DIR / "baseline" / f"{arch}_baseline{suffix}.pth"
    out_dir = RESULTS_DIR / "students"
    t0 = time.time()
    fit(model, train, val, epochs=hp["epochs"], lr=hp["lr"], weight_decay=hp["weight_decay"],
        warmup_epochs=hp["warmup_epochs"], device=device, ckpt_path=ckpt, config=config,
        history_csv=out_dir / f"{arch}_baseline_history{suffix}.csv", resume=args.resume)
    minutes = (time.time() - t0) / 60

    state = load_checkpoint(ckpt)
    model.load_state_dict(state["state_dict"])
    val_m, test_m = evaluate(model, val, device), evaluate(model, test, device)
    state.update(test_acc=test_m["top1"], test_top5=test_m["top5"])
    torch.save(state, ckpt)
    res = {"arch": arch, "mode": "baseline", "params": count_params(model), "best_epoch": state["epoch"],
           "val_top1": val_m["top1"], "test_top1": test_m["top1"], "test_top5": test_m["top5"],
           "train_minutes": round(minutes, 2), "checkpoint": f"baseline/{ckpt.name}"}
    save_json(res, out_dir / f"{arch}_baseline{suffix}.json")
    print(f"\n[{arch}] baseline: best epoch {state['epoch']} | val {val_m['top1']:.2f}% | "
          f"test top-1 {test_m['top1']:.2f}% | top-5 {test_m['top5']:.2f}% | {minutes:.1f} min")
    return res


def main():
    args = parse_args()
    device = get_device()
    archs = list(STUDENTS) if args.arch == "all" else [args.arch]
    for arch in archs:
        suffix = "_quick" if args.quick else ""
        with Tee(LOG_DIR / f"03_baseline_{arch}{suffix}.log"):
            train_baseline(arch, args, device)


if __name__ == "__main__":
    main()
