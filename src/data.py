"""CIFAR-10 / CIFAR-100 download, fixed train/val/test splits and a GPU-resident data loader.

Datasets
--------
* CIFAR-100 is used for Question 1 (teacher ResNet-50 and the three distilled students).
* CIFAR-10  is used for Question 2 (neural architecture search).

Both are always downloaded from code: first from the official source (University of Toronto)
through torchvision, and if that fails or is too slow, from the Hugging Face hub mirror. The raw
uint8 images are then cached once as `data/<name>.npz` so every later run starts instantly.

Splits
------
The official 50,000 training images are split once with a fixed seed into 45,000 for training
and 5,000 for validation. Validation is used for model selection (best epoch, NAS ranking);
the official 10,000 test images are only used for the final reported numbers.

Why a GPU-resident loader?
--------------------------
CIFAR is tiny (50,000 x 32 x 32 x 3 bytes = 150 MB), so the whole training set fits on the GPU.
Doing the standard augmentation (random crop with 4-pixel padding + horizontal flip) directly on
GPU tensors removes the CPU DataLoader bottleneck, which is severe on Windows where worker
processes are slow to start. It also keeps the CPU free for the latency measurements of the NAS.
The augmentation is mathematically the same as torchvision's RandomCrop(32, 4) + RandomHorizontalFlip.
"""
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import DATA_DIR, SEED

# ----------------------------------------------------------------------------- dataset constants
DATASETS = {
    "cifar10": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "folder": "cifar-10-batches-py",
        "hf_name": "uoft-cs/cifar10", "hf_label": "label",
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "folder": "cifar-100-python",
        "hf_name": "uoft-cs/cifar100", "hf_label": "fine_label",
        "num_classes": 100,
        "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762),
    },
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

NUM_TRAIN, NUM_VAL = 45_000, 5_000      # split of the official 50k training images


# ----------------------------------------------------------------------------- download helpers
def _download(url: str, dest: Path, check_after_s: int = 180, min_mb_per_min: float = 3.0) -> bool:
    """Download `url` to `dest` with progress output.

    Returns False (instead of hanging for an hour) if the server is too slow after
    `check_after_s` seconds, so the caller can switch to the Hugging Face mirror.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    start, done, last_print = time.time(), 0, 0.0
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            elapsed = time.time() - start
            if elapsed - last_print > 10:
                last_print = elapsed
                print(f"  downloaded {done / 1e6:6.1f} / {total / 1e6:.1f} MB in {elapsed:4.0f} s")
            if elapsed > check_after_s and done / 1e6 / (elapsed / 60) < min_mb_per_min:
                print("  server too slow, giving up on it")
                return False
    tmp.replace(dest)
    return True


def _read_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _from_toronto(name: str):
    """Official source: download the python tarball, extract it, read the pickled batches."""
    spec = DATASETS[name]
    tar_path = DATA_DIR / Path(spec["url"]).name
    folder = DATA_DIR / spec["folder"]
    if not folder.exists():
        if not tar_path.exists():
            print(f"Downloading {name} from {spec['url']}")
            if not _download(spec["url"], tar_path):
                return None
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR, filter="data")

    def to_hwc(raw):  # stored as (N, 3072) in CHW order
        return np.asarray(raw, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    if name == "cifar10":
        parts = [_read_pickle(folder / f"data_batch_{i}") for i in range(1, 6)]
        train_x = np.concatenate([to_hwc(p["data"]) for p in parts])
        train_y = np.concatenate([np.asarray(p["labels"]) for p in parts])
        test = _read_pickle(folder / "test_batch")
        test_x, test_y = to_hwc(test["data"]), np.asarray(test["labels"])
        classes = CIFAR10_CLASSES
    else:
        train, test = _read_pickle(folder / "train"), _read_pickle(folder / "test")
        train_x, train_y = to_hwc(train["data"]), np.asarray(train["fine_labels"])
        test_x, test_y = to_hwc(test["data"]), np.asarray(test["fine_labels"])
        classes = _read_pickle(folder / "meta")["fine_label_names"]
    return train_x, train_y.astype(np.int64), test_x, test_y.astype(np.int64), classes


def _from_huggingface(name: str):
    """Fallback source: the Hugging Face copy (needs `pip install datasets`)."""
    from datasets import load_dataset
    spec = DATASETS[name]
    print(f"Downloading {name} from Hugging Face ({spec['hf_name']})")
    ds = load_dataset(spec["hf_name"], cache_dir=str(DATA_DIR / "hf_cache"))
    img_col = "img"

    def arrays(split):
        x = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in split[img_col]])
        return x, np.asarray(split[spec["hf_label"]], dtype=np.int64)

    train_x, train_y = arrays(ds["train"])
    test_x, test_y = arrays(ds["test"])
    classes = ds["train"].features[spec["hf_label"]].names
    return train_x, train_y, test_x, test_y, classes


def load_arrays(name: str):
    """Return (train_x, train_y, test_x, test_y, class_names) as uint8 HWC images + int64 labels.

    Cached in `data/<name>.npz` after the first call.
    """
    assert name in DATASETS, f"unknown dataset {name}"
    cache = DATA_DIR / f"{name}.npz"
    if cache.exists():
        d = np.load(cache)
        classes = list(d["classes"]) if "classes" in d.files else CIFAR10_CLASSES
        return d["train_x"], d["train_y"], d["test_x"], d["test_y"], [str(c) for c in classes]

    result, source = None, "torchvision / cs.toronto.edu"
    try:
        result = _from_toronto(name)
    except Exception as e:  # network error, corrupt archive, ...
        print(f"  official source failed: {e}")
    if result is None:
        result, source = _from_huggingface(name), "huggingface"
    train_x, train_y, test_x, test_y, classes = result

    # sanity checks: shapes and label range must match the official dataset description
    assert train_x.shape == (50_000, 32, 32, 3) and test_x.shape == (10_000, 32, 32, 3)
    assert train_y.max() + 1 == DATASETS[name]["num_classes"] == len(classes)
    np.savez(cache, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
             classes=np.array(classes), source=source)
    print(f"{name} cached in {cache} (source: {source})")
    return train_x, train_y, test_x, test_y, list(classes)


def class_names(name: str):
    return load_arrays(name)[4]


def split_indices(n_total: int = 50_000, seed: int = SEED):
    """Fixed random 45k/5k split of the official training set (identical in every script)."""
    perm = np.random.RandomState(seed).permutation(n_total)
    return perm[:NUM_TRAIN], perm[NUM_TRAIN:NUM_TRAIN + NUM_VAL]


# ----------------------------------------------------------------------------- GPU data pipeline
class GPUAugLoader:
    """Mini-batch iterator over a uint8 image tensor that lives on the training device.

    Parameters
    ----------
    images      : uint8 tensor (N, 3, 32, 32) on the device
    labels      : int64 tensor (N,) on the device
    mean, std   : per-channel normalisation statistics of the dataset
    batch_size  : images per batch
    train       : True -> shuffle + random crop (pad 4) + horizontal flip; False -> plain
    resolution  : optional output size (used by the NAS for variable input resolution);
                  images are resized with bilinear interpolation after augmentation
    drop_last   : drop the last incomplete batch (keeps BatchNorm statistics stable in training)
    """

    def __init__(self, images, labels, mean, std, batch_size=128, train=True,
                 resolution=None, drop_last=None, seed=SEED):
        self.x, self.y = images, labels
        self.batch_size, self.train = batch_size, train
        self.resolution = resolution
        self.drop_last = train if drop_last is None else drop_last
        dev = images.device
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
        self.gen = torch.Generator(device=dev).manual_seed(seed)

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _augment(self, x):
        """Random 32x32 crop from the zero-padded 40x40 image + random horizontal flip."""
        b = x.shape[0]
        pad = 4
        x = F.pad(x, (pad, pad, pad, pad))                       # zero padding, like torchvision
        off_y = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=self.gen)
        off_x = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=self.gen)
        base = torch.arange(32, device=x.device)
        width = x.shape[3]                                        # 40 after padding
        rows = (off_y[:, None] + base)[:, None, :, None].expand(b, 3, 32, width)
        cols = (off_x[:, None] + base)[:, None, None, :].expand(b, 3, 32, 32)
        # gather the 32 selected rows, then the 32 selected columns -> per-sample crop window
        x = x.gather(2, rows)
        x = x.gather(3, cols)
        flip = torch.rand(b, device=x.device, generator=self.gen) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def __iter__(self):
        n = len(self.x)
        if self.train:
            order = torch.randperm(n, device=self.x.device, generator=self.gen)
        else:
            order = torch.arange(n, device=self.x.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.x[idx].float().div_(255.0)
            if self.train:
                x = self._augment(x)
            x = (x - self.mean) / self.std
            if self.resolution is not None and self.resolution != x.shape[-1]:
                x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear",
                                  align_corners=False, antialias=self.resolution < x.shape[-1])
            yield x.contiguous(memory_format=torch.channels_last), self.y[idx]


def get_loaders(name: str, device, batch_size=128, eval_batch_size=500, resolution=None,
                train_subset=None, seed=SEED):
    """Build train / val / test GPU loaders for `name` ('cifar10' or 'cifar100').

    `train_subset` (int) optionally keeps only the first k shuffled training images, which the
    NAS uses for cheap proxy training of candidate architectures.
    """
    train_x, train_y, test_x, test_y, classes = load_arrays(name)
    tr_idx, va_idx = split_indices(len(train_x), seed)
    if train_subset is not None:
        tr_idx = tr_idx[:train_subset]
    spec = DATASETS[name]

    def to_dev(x, y):
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
        return x, torch.from_numpy(y).to(device)

    common = dict(mean=spec["mean"], std=spec["std"], resolution=resolution, seed=seed)
    train = GPUAugLoader(*to_dev(train_x[tr_idx], train_y[tr_idx]), batch_size=batch_size,
                         train=True, **common)
    val = GPUAugLoader(*to_dev(train_x[va_idx], train_y[va_idx]), batch_size=eval_batch_size,
                       train=False, **common)
    test = GPUAugLoader(*to_dev(test_x, test_y), batch_size=eval_batch_size, train=False, **common)
    return train, val, test, classes


# ----------------------------------------------------------------------------- single-image preprocessing
def preprocess_image(img, name: str, resolution: int = 32) -> torch.Tensor:
    """Turn a PIL image of any size into a normalised (1, 3, R, R) float tensor.

    Used by the inference pipelines for user-supplied pictures. CIFAR models expect small,
    centred objects, so the image is simply resized (no crop) to the model's input resolution.
    """
    from PIL import Image
    spec = DATASETS[name]
    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    x = (x - torch.tensor(spec["mean"]).view(1, 3, 1, 1)) / torch.tensor(spec["std"]).view(1, 3, 1, 1)
    if resolution != 32:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False,
                          antialias=resolution < 32)
    return x


def normalise_uint8_batch(x_uint8_nhwc: np.ndarray, name: str, resolution: int = 32) -> torch.Tensor:
    """Normalise a batch of raw uint8 HWC test images exactly like the evaluation loader does."""
    spec = DATASETS[name]
    x = torch.from_numpy(np.ascontiguousarray(x_uint8_nhwc.transpose(0, 3, 1, 2))).float() / 255.0
    x = (x - torch.tensor(spec["mean"]).view(1, 3, 1, 1)) / torch.tensor(spec["std"]).view(1, 3, 1, 1)
    if resolution != 32:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False,
                          antialias=resolution < 32)
    return x
