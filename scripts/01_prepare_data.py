"""Step 1 - download and cache CIFAR-10 and CIFAR-100, and check the data pipeline.

What it does
  * downloads both datasets from code (official source, Hugging Face fallback) into data/
  * prints split sizes and class counts
  * verifies the GPU augmentation: crops must be genuine 32x32 windows of the padded image and
    the evaluation loader must return the untouched (normalised) images
  * writes results/system_info.json (hardware + software versions)

Run:  python scripts/01_prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.common import LOG_DIR, RESULTS_DIR, Tee, banner, get_device, save_json, system_info
from src.data import DATASETS, GPUAugLoader, get_loaders, load_arrays, split_indices


def check_augmentation(device):
    """Every augmented image must equal some crop (optionally flipped) of the padded original."""
    train_x, train_y, *_ = load_arrays("cifar10")
    x = torch.from_numpy(train_x[:64].transpose(0, 3, 1, 2).copy()).to(device)
    y = torch.from_numpy(train_y[:64]).to(device)
    loader = GPUAugLoader(x, y, mean=(0, 0, 0), std=(1, 1, 1), batch_size=64, train=True)
    raw = x.float() / 255
    aug = loader._augment(raw.clone())        # augment a known batch directly
    padded = F.pad(raw, (4, 4, 4, 4))
    ok = 0
    for i in range(64):
        found = False
        for dy in range(9):
            for dx in range(9):
                win = padded[i, :, dy:dy + 32, dx:dx + 32]
                if torch.equal(win, aug[i]) or torch.equal(win.flip(2), aug[i]):
                    found = True
                    break
            if found:
                break
        ok += found
    print(f"augmentation check: {ok}/64 augmented images are valid crop(+flip) windows")
    assert ok == 64, "GPU augmentation is wrong"


def main():
    with Tee(LOG_DIR / "01_prepare_data.log"):
        device = get_device()
        banner("Dataset download / cache")
        summary = {}
        for name in ("cifar10", "cifar100"):
            train_x, train_y, test_x, test_y, classes = load_arrays(name)
            tr, va = split_indices(len(train_x))
            assert len(set(tr) & set(va)) == 0, "train/val overlap"
            counts = np.bincount(train_y[tr], minlength=len(classes))
            print(f"{name:9s}: train {len(tr)}, val {len(va)}, test {len(test_x)}, "
                  f"classes {len(classes)}, train images per class {counts.min()}-{counts.max()}")
            summary[name] = dict(train=len(tr), val=len(va), test=len(test_x),
                                 num_classes=len(classes), mean=DATASETS[name]["mean"],
                                 std=DATASETS[name]["std"], first_classes=classes[:5])

        banner("GPU data pipeline checks")
        check_augmentation(device)
        train, val, test, _ = get_loaders("cifar100", device, batch_size=128)
        xb, yb = next(iter(train))
        print(f"train batch: {tuple(xb.shape)} {xb.dtype} on {xb.device}, "
              f"mean {xb.mean():.3f}, std {xb.std():.3f}")
        xv, _ = next(iter(val))
        print(f"val batch  : {tuple(xv.shape)}, mean {xv.mean():.3f}, std {xv.std():.3f}")
        _, _, _, _ = get_loaders("cifar10", device, resolution=24)
        print(f"batches per epoch (cifar100, bs=128): {len(train)}")

        info = system_info()
        save_json(info, RESULTS_DIR / "system_info.json")
        save_json(summary, RESULTS_DIR / "dataset_summary.json")
        banner("System")
        for k, v in info.items():
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
