"""Q1.3/Q1.4 - Train the three students WITH knowledge distillation.      (deliverable 4)

The frozen ResNet-50 teacher (02_train_teacher.py) supervises each student with the combined
objective implemented in src/kd.py:

    L = (1 - alpha) * CE(labels)                       hard labels
      + alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))   soft targets, temperature T
      + beta  * MSE(regressor(f_s), f_t)               intermediate-feature (hint) distillation

Default: alpha = 0.9, T = 4, and beta taken from 04b_calibrate_feature_weight.py, which balances
the feature term against the classification terms at initialisation. The teacher hint is
ResNet-50 `layer3` (1024 x 8 x 8); the student hint is its own 8 x 8 stage (see students.py).

Modes
  --mode full : hard + soft + feature distillation (the deliverable checkpoints)
  --mode soft : hard + soft only (beta = 0), an ablation that isolates the feature term

Everything else (optimiser, schedule, epochs, augmentation, seed) is identical to the baseline
script, so the comparison isolates the effect of distillation.

Outputs (per student <arch>)
  checkpoints/kd/<arch>_kd.pth                  (mode full)  /  checkpoints/kd/ablation/<arch>_kd_soft.pth
  results/students/<arch>_kd[_soft]_history.csv, results/students/<arch>_kd[_soft].json
  results/logs/04_kd_<arch>[_soft].log

Run:  python scripts/04_train_students_kd.py                      (all three, full KD)
      python scripts/04_train_students_kd.py --arch mobilenetv2 --mode soft
      python scripts/04_train_students_kd.py --quick
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common import CKPT_DIR, LOG_DIR, RESULTS_DIR, SEED, Tee, banner, get_device, save_json, set_seed
from src.data import get_loaders
from src.kd import DistillationLoss, KDConfig
from src.models import DISPLAY_NAMES, STUDENTS, build_student, build_teacher
from src.train_utils import count_params, evaluate, fit, load_checkpoint

# reuse the exact hyper-parameters of the baseline pipeline
_spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("03_train_students_baseline.py"))
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
STUDENT_HPARAMS = _baseline.STUDENT_HPARAMS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    p.add_argument("--mode", default="full", choices=["full", "soft"])
    p.add_argument("--alpha", type=float, default=0.9, help="weight of the soft-target loss")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--beta", type=float, default=None,
                   help="weight of the feature loss (mode full); default: value chosen by "
                        "04b_calibrate_feature_weight.py, or 50 if that has not been run")
    p.add_argument("--epochs", type=int, default=STUDENT_HPARAMS["epochs"])
    p.add_argument("--teacher", default=str(CKPT_DIR / "teacher" / "resnet50_cifar100.pth"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def load_teacher(path, device, num_classes=100):
    """Rebuild the teacher from its checkpoint (no ImageNet download needed) and freeze it."""
    state = load_checkpoint(path)
    teacher = build_teacher(num_classes=num_classes, pretrained=False)
    teacher.load_state_dict(state["state_dict"])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, state


def train_kd(arch, args, device, teacher):
    quick = "_quick" if args.quick else ""
    tag = "kd" if args.mode == "full" else "kd_soft"
    set_seed(SEED)
    hp = dict(STUDENT_HPARAMS, epochs=2 if args.quick else args.epochs)
    train, val, test, classes = get_loaders("cifar100", device, batch_size=hp["batch_size"],
                                            train_subset=5000 if args.quick else None)
    model = build_student(arch, num_classes=len(classes))
    kd_cfg = KDConfig(alpha=args.alpha, temperature=args.temperature,
                      beta=args.beta if args.mode == "full" else 0.0)
    kd_loss = DistillationLoss(kd_cfg, student_channels=model.hint_channels,
                               teacher_channels=teacher.hint_channels)

    banner(f"Knowledge distillation ({args.mode}): ResNet-50 -> {DISPLAY_NAMES[arch]}")
    print(f"student params {count_params(model):,} | teacher params {count_params(teacher):,}")
    print(f"KD config: alpha {kd_cfg.alpha} | T {kd_cfg.temperature} | beta {kd_cfg.beta} | "
          f"hint: student {model.hint_channels}ch -> teacher {teacher.hint_layer} "
          f"{teacher.hint_channels}ch" + (" (regressor 1x1 conv + BN)" if kd_cfg.beta > 0 else " (unused)"))
    print(f"epochs {hp['epochs']} | lr {hp['lr']} | wd {hp['weight_decay']} | batch {hp['batch_size']}")

    config = {"arch": arch, "num_classes": len(classes), "dataset": "cifar100", "mode": tag,
              "kd": kd_cfg.to_dict(), "teacher": Path(args.teacher).name, "seed": SEED, **hp}
    folder = CKPT_DIR / "kd" if args.mode == "full" else CKPT_DIR / "kd" / "ablation"
    ckpt = folder / f"{arch}_{tag}{quick}.pth"
    out_dir = RESULTS_DIR / "students"
    t0 = time.time()
    fit(model, train, val, epochs=hp["epochs"], lr=hp["lr"], weight_decay=hp["weight_decay"],
        warmup_epochs=hp["warmup_epochs"], device=device, ckpt_path=ckpt, config=config,
        history_csv=out_dir / f"{arch}_{tag}_history{quick}.csv", teacher=teacher, kd_loss=kd_loss,
        resume=args.resume)
    minutes = (time.time() - t0) / 60

    # the deployable student is just the student network: the regressor is discarded
    state = load_checkpoint(ckpt)
    model.load_state_dict(state["state_dict"])
    val_m, test_m = evaluate(model, val, device), evaluate(model, test, device)
    state.update(test_acc=test_m["top1"], test_top5=test_m["top5"])
    torch.save(state, ckpt)
    res = {"arch": arch, "mode": tag, "params": count_params(model), "best_epoch": state["epoch"],
           "val_top1": val_m["top1"], "test_top1": test_m["top1"], "test_top5": test_m["top5"],
           "train_minutes": round(minutes, 2), "kd": kd_cfg.to_dict(),
           "checkpoint": str(ckpt.relative_to(CKPT_DIR).as_posix())}
    save_json(res, out_dir / f"{arch}_{tag}{quick}.json")
    print(f"\n[{arch}] {tag}: best epoch {state['epoch']} | val {val_m['top1']:.2f}% | "
          f"test top-1 {test_m['top1']:.2f}% | top-5 {test_m['top5']:.2f}% | {minutes:.1f} min")
    return res


def main():
    args = parse_args()
    if args.beta is None:
        calib = RESULTS_DIR / "kd_beta_calibration.json"
        args.beta = float(json.loads(calib.read_text())["chosen_beta"]) if calib.exists() else 50.0
    device = get_device()
    teacher_path = Path(args.teacher)
    if args.quick and not teacher_path.exists():
        teacher_path = teacher_path.with_name("resnet50_cifar100_quick.pth")
    teacher, tstate = load_teacher(teacher_path, device)
    archs = list(STUDENTS) if args.arch == "all" else [args.arch]
    for arch in archs:
        tag = "kd" if args.mode == "full" else "kd_soft"
        with Tee(LOG_DIR / f"04_{tag}_{arch}{'_quick' if args.quick else ''}.log"):
            print(f"teacher checkpoint {teacher_path.name}: test top-1 {tstate.get('test_acc', float('nan')):.2f}%")
            train_kd(arch, args, device, teacher)


if __name__ == "__main__":
    main()
