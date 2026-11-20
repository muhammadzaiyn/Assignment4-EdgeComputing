"""Choose the feature-distillation weight beta from measured loss magnitudes.

The three KD loss terms have very different natural scales: the soft-target term T^2*KL and the
cross-entropy are a few units at initialisation, while the hint MSE depends on how large the
teacher's layer3 activations are. If beta is too small the feature term does nothing, if it is
too large it swamps the classification signal.

This script measures, for every (untrained) student with a freshly initialised regressor and the
trained teacher, the average value of each term over 20 training batches, and reports the beta
that makes  beta * MSE  equal to the classification part  (1-alpha)*CE + alpha*T^2*KL  at the
start of training. The median over the three students, rounded to a convenient value, is used
for all students (one shared setting keeps the comparison simple).

Output: results/kd_beta_calibration.json
Run:    python scripts/04b_calibrate_feature_weight.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statistics

import torch

from src.common import CKPT_DIR, LOG_DIR, RESULTS_DIR, SEED, Tee, banner, get_device, save_json, set_seed
from src.data import get_loaders
from src.kd import HintRegressor, feature_loss, soft_target_loss
from src.models import STUDENTS, build_student, build_teacher
from src.train_utils import load_checkpoint

ALPHA, T = 0.9, 4.0


@torch.no_grad()
def main():
    with Tee(LOG_DIR / "04b_calibrate_feature_weight.log"):
        set_seed(SEED)
        dev = get_device()
        teacher = build_teacher(100, pretrained=False)
        teacher.load_state_dict(load_checkpoint(CKPT_DIR / "teacher" / "resnet50_cifar100.pth")["state_dict"])
        teacher.to(dev).eval()
        train, _, _, _ = get_loaders("cifar100", dev, batch_size=128)
        banner("Loss magnitudes at initialisation (20 batches)")
        out = {}
        for arch in STUDENTS:
            s = build_student(arch).to(dev).train()
            reg = HintRegressor(s.hint_channels, teacher.hint_channels).to(dev).train()
            ce, kl, mse = [], [], []
            for i, (x, y) in enumerate(train):
                if i == 20:
                    break
                t_logits, feats = teacher(x, return_features=True)
                s_logits, f_s = s(x, return_features=True)
                ce.append(torch.nn.functional.cross_entropy(s_logits, y).item())
                kl.append(soft_target_loss(s_logits, t_logits, T).item())
                mse.append(feature_loss(reg(f_s), feats[teacher.hint_layer]).item())
            cls = (1 - ALPHA) * statistics.mean(ce) + ALPHA * statistics.mean(kl)
            beta = cls / statistics.mean(mse)
            out[arch] = {"ce": statistics.mean(ce), "soft_kl_T2": statistics.mean(kl),
                         "feat_mse": statistics.mean(mse), "classification_part": cls, "balanced_beta": beta}
            print(f"{arch:20s} CE {out[arch]['ce']:.3f} | T^2*KL {out[arch]['soft_kl_T2']:.3f} | "
                  f"feature MSE {out[arch]['feat_mse']:.4f} -> balanced beta {beta:.1f}")
        med = statistics.median(v["balanced_beta"] for v in out.values())
        # round to 1 significant figure (e.g. 23.4 -> 20, 61 -> 60)
        import math
        mag = 10 ** math.floor(math.log10(med))
        chosen = round(med / mag) * mag
        print(f"\nmedian balanced beta {med:.1f} -> chosen beta = {chosen:g}")
        save_json({"alpha": ALPHA, "temperature": T, "per_student": out, "median_beta": med, "chosen_beta": chosen},
                  RESULTS_DIR / "kd_beta_calibration.json")


if __name__ == "__main__":
    main()
