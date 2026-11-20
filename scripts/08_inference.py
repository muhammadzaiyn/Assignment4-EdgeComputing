"""Unified PyTorch inference pipeline for every trained student and the NAS model.  (deliverable 9)

One command-line tool covers all seven deployable models (3 baseline students, 3 distilled
students, NAS-best). The correct architecture, input resolution, normalisation and class list
are taken from each checkpoint, so the user only chooses a model name.

Examples
  # classify your own picture (any size; it is resized to the model's input resolution)
  python scripts/08_inference.py --model mobilenetv2_kd --image path/to/picture.jpg

  # the same picture through every model, side by side
  python scripts/08_inference.py --model all --image path/to/picture.jpg

  # N random test images of the model's dataset, prediction vs ground truth
  python scripts/08_inference.py --model squeezenet_kd --num-samples 8

  # full test-set accuracy of one or all models
  python scripts/08_inference.py --model all --evaluate

Outputs: printed tables; with --evaluate also results/inference_eval_pytorch.csv
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.common import LOG_DIR, RESULTS_DIR, SEED, Tee, banner, write_csv
from src.data import load_arrays, normalise_uint8_batch, preprocess_image
from src.infer_utils import ALL_MODELS, load_model, predict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", choices=["all", *ALL_MODELS])
    p.add_argument("--image", help="path to an image file to classify")
    p.add_argument("--num-samples", type=int, default=0, help="classify N random test images")
    p.add_argument("--evaluate", action="store_true", help="accuracy on the full test set")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def timed_predict(model, x):
    """Probabilities and wall-clock latency (ms) of one forward pass (after one warm-up)."""
    predict(model, x[:1])
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs = predict(model, x)
    return probs, (time.perf_counter() - t0) * 1000


def run_image(names, args):
    from PIL import Image
    img = Image.open(args.image)
    banner(f"Image: {args.image}  ({img.size[0]}x{img.size[1]})")
    print(f"{'model':34s} {'top-1 prediction':22s} {'conf':>7s}   {'latency':>9s}   other top-k")
    for name in names:
        model, info = load_model(name, args.device)
        x = preprocess_image(img, info.dataset, info.resolution)
        probs, ms = timed_predict(model, x)
        conf, idx = probs[0].topk(args.topk)
        others = ", ".join(f"{info.classes[i]} {c * 100:.1f}%" for c, i in zip(conf[1:], idx[1:]))
        print(f"{info.display:34s} {info.classes[idx[0]]:22s} {conf[0] * 100:6.2f}%   {ms:7.2f} ms   {others}")


def run_samples(names, args):
    rng = np.random.RandomState(args.seed)
    for name in names:
        model, info = load_model(name, args.device)
        _, _, test_x, test_y, _ = load_arrays(info.dataset)
        idx = rng.choice(len(test_x), args.num_samples, replace=False)
        x = normalise_uint8_batch(test_x[idx], info.dataset, info.resolution)
        probs, ms = timed_predict(model, x)
        conf, pred = probs.max(1)
        banner(f"{info.display} - {args.num_samples} random {info.dataset} test images "
               f"(batch latency {ms:.2f} ms on {args.device})")
        print(f"{'image':>6s}  {'ground truth':18s} {'prediction':18s} {'conf':>7s}  ok")
        for k, i in enumerate(idx):
            ok = "yes" if pred[k] == test_y[i] else "NO"
            print(f"{i:6d}  {info.classes[test_y[i]]:18s} {info.classes[pred[k]]:18s} "
                  f"{conf[k] * 100:6.2f}%  {ok}")
        print(f"correct: {(pred.numpy() == test_y[idx]).sum()}/{len(idx)}")


def run_evaluate(names, args):
    banner("Full test-set accuracy (PyTorch)")
    rows = []
    for name in names:
        model, info = load_model(name, args.device)
        _, _, test_x, test_y, _ = load_arrays(info.dataset)
        preds, t0 = [], time.perf_counter()
        for s in range(0, len(test_x), 500):
            x = normalise_uint8_batch(test_x[s:s + 500], info.dataset, info.resolution)
            preds.append(predict(model, x).argmax(1).numpy())
        sec = time.perf_counter() - t0
        acc = float((np.concatenate(preds) == test_y).mean() * 100)
        rows.append({"model": name, "dataset": info.dataset, "test_top1": acc, "images": len(test_y),
                     "seconds": round(sec, 2), "device": args.device})
        print(f"{info.display:34s} {info.dataset:9s} top-1 {acc:6.2f}%   ({len(test_y)} images, {sec:.1f} s)")
    write_csv(rows, RESULTS_DIR / "inference_eval_pytorch.csv")


def main():
    args = parse_args()
    names = ALL_MODELS if args.model == "all" else [args.model]
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not (args.image or args.num_samples or args.evaluate):
        args.num_samples = 5                                   # sensible default demo
    with Tee(LOG_DIR / "08_inference.log", mode="a"):
        if args.image:
            run_image(names, args)
        if args.num_samples:
            run_samples(names, args)
        if args.evaluate:
            run_evaluate(names, args)


if __name__ == "__main__":
    main()
