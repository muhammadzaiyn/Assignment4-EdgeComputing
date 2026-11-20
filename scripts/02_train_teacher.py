"""Q1.1 - Train the ResNet-50 teacher on CIFAR-100.                       (deliverable 1)

The teacher starts from ImageNet-pretrained bottleneck stages (see src/models/teacher.py for the
CIFAR stem adaptation) and is fine-tuned end-to-end on the 45k training split. The checkpoint
with the best validation accuracy is kept and evaluated once on the 10k test set.

Outputs
  checkpoints/teacher/resnet50_cifar100.pth      best-validation weights + config + metrics
  results/teacher_history.csv                    per-epoch training log
  results/teacher_results.json                   final val / test accuracy, params, time
  results/logs/02_train_teacher.log              full console output

Run:  python scripts/02_train_teacher.py            (full: 40 epochs, ~15 min on an RTX 5060 Ti)
      python scripts/02_train_teacher.py --quick    (2 epochs on a 5k subset, plumbing test)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common import CKPT_DIR, LOG_DIR, RESULTS_DIR, SEED, Tee, banner, get_device, save_json, set_seed
from src.data import get_loaders
from src.models import build_teacher
from src.train_utils import count_params, evaluate, fit, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05, help="peak learning rate (fine-tuning)")
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch instead")
    p.add_argument("--resume", action="store_true", help="continue from the last epoch checkpoint")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    return p.parse_args()


def main():
    args = parse_args()
    suffix = "_quick" if args.quick else ""
    with Tee(LOG_DIR / f"02_train_teacher{suffix}.log"):
        set_seed(SEED)
        device = get_device()
        epochs = 2 if args.quick else args.epochs
        train, val, test, classes = get_loaders("cifar100", device, batch_size=args.batch_size,
                                                train_subset=5000 if args.quick else None)
        model = build_teacher(num_classes=len(classes), pretrained=not args.no_pretrained)

        banner("Teacher: ResNet-50 (CIFAR stem) on CIFAR-100")
        print(f"device {device} | params {count_params(model):,} | ImageNet init {not args.no_pretrained}")
        print(f"epochs {epochs} | batch {args.batch_size} | lr {args.lr} (warm-up 1 epoch + cosine) | "
              f"wd {args.wd} | train batches/epoch {len(train)}")

        config = {"arch": "resnet50", "num_classes": len(classes), "dataset": "cifar100",
                  "pretrained_init": not args.no_pretrained, "epochs": epochs, "lr": args.lr,
                  "weight_decay": args.wd, "batch_size": args.batch_size, "seed": SEED}
        ckpt = CKPT_DIR / "teacher" / f"resnet50_cifar100{suffix}.pth"
        t0 = time.time()
        best_val, history = fit(model, train, val, epochs=epochs, lr=args.lr, weight_decay=args.wd,
                                device=device, ckpt_path=ckpt, config=config,
                                history_csv=RESULTS_DIR / f"teacher_history{suffix}.csv",
                                resume=args.resume)
        train_minutes = (time.time() - t0) / 60

        # reload the best-validation weights and evaluate once on the test set
        state = load_checkpoint(ckpt)
        model.load_state_dict(state["state_dict"])
        test_m = evaluate(model, test, device)
        val_m = evaluate(model, val, device)
        state.update(test_acc=test_m["top1"], test_top5=test_m["top5"])
        torch.save(state, ckpt)

        results = {"model": "resnet50", "params": count_params(model), "best_epoch": state["epoch"],
                   "val_top1": val_m["top1"], "test_top1": test_m["top1"], "test_top5": test_m["top5"],
                   "train_minutes": round(train_minutes, 2), "checkpoint": ckpt.name, **config}
        save_json(results, RESULTS_DIR / f"teacher_results{suffix}.json")
        banner("Teacher result")
        print(f"best epoch {state['epoch']} | val top-1 {val_m['top1']:.2f}% | "
              f"test top-1 {test_m['top1']:.2f}% | test top-5 {test_m['top5']:.2f}% | "
              f"training time {train_minutes:.1f} min")
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
