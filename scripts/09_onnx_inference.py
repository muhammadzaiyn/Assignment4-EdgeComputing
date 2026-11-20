"""CPU inference with ONNX Runtime for the exported student and NAS models.   (deliverable 10)

Runs the .onnx files from onnx_models/ with the CPUExecutionProvider only - no PyTorch is
needed at inference time, only numpy, Pillow and onnxruntime, which is what an edge device
would ship. Preprocessing (resize, scale to [0,1], per-channel normalisation) is done in numpy.

Examples
  python scripts/09_onnx_inference.py --model mobilenetv2_kd --image picture.jpg
  python scripts/09_onnx_inference.py --model all --num-samples 5
  python scripts/09_onnx_inference.py --model all --evaluate            (full test set, CPU)
  python scripts/09_onnx_inference.py --model nas_best --threads 4 --evaluate

Outputs: printed tables; with --evaluate also results/inference_eval_onnx.csv
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common import LOG_DIR, ONNX_DIR, RESULTS_DIR, SEED, Tee, banner, load_json, write_csv
from src.data import DATASETS, load_arrays
from src.infer_utils import ALL_MODELS, ort_session, softmax_np


def model_meta(name):
    """Dataset and input size of an exported model (read from the export report, no torch needed)."""
    dataset = "cifar10" if name == "nas_best" else "cifar100"
    res = 32
    if name == "nas_best":
        res = load_json(RESULTS_DIR / "nas" / "best_architecture.json")["config"]["resolution"]
    return dataset, res


def preprocess_np(images_uint8_nhwc, dataset, res):
    """uint8 HWC images -> normalised float32 NCHW at the model's resolution (numpy + Pillow)."""
    from PIL import Image
    if images_uint8_nhwc.shape[1] != res:
        images_uint8_nhwc = np.stack([np.asarray(Image.fromarray(im).resize((res, res), Image.BILINEAR))
                                      for im in images_uint8_nhwc])
    x = images_uint8_nhwc.astype(np.float32) / 255.0
    mean = np.array(DATASETS[dataset]["mean"], dtype=np.float32)
    std = np.array(DATASETS[dataset]["std"], dtype=np.float32)
    return np.ascontiguousarray(((x - mean) / std).transpose(0, 3, 1, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", choices=["all", *ALL_MODELS])
    p.add_argument("--image")
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads")
    args = p.parse_args()
    names = ALL_MODELS if args.model == "all" else [args.model]
    if not (args.image or args.num_samples or args.evaluate):
        args.num_samples = 5

    with Tee(LOG_DIR / "09_onnx_inference.log", mode="a"):
        rows = []
        rng = np.random.RandomState(SEED)
        for name in names:
            path = ONNX_DIR / f"{name}.onnx"
            sess = ort_session(path, threads=args.threads)
            dataset, res = model_meta(name)
            _, _, test_x, test_y, classes = load_arrays(dataset)
            banner(f"ONNX Runtime ({sess.get_providers()[0]}, {args.threads} thread) - {path.name}")

            if args.image:
                from PIL import Image
                img = np.asarray(Image.open(args.image).convert("RGB").resize((32, 32), Image.BILINEAR))[None]
                x = preprocess_np(img, dataset, res)
                t0 = time.perf_counter()
                probs = softmax_np(sess.run(None, {"input": x})[0])[0]
                ms = (time.perf_counter() - t0) * 1000
                top = probs.argsort()[::-1][:3]
                print(f"{args.image}: {classes[top[0]]} ({probs[top[0]] * 100:.2f}%) | "
                      f"then {classes[top[1]]} {probs[top[1]] * 100:.1f}%, {classes[top[2]]} "
                      f"{probs[top[2]] * 100:.1f}% | {ms:.2f} ms")

            if args.num_samples:
                idx = rng.choice(len(test_x), args.num_samples, replace=False)
                x = preprocess_np(test_x[idx], dataset, res)
                for k, i in enumerate(idx):
                    t0 = time.perf_counter()
                    probs = softmax_np(sess.run(None, {"input": x[k:k + 1]})[0])[0]
                    ms = (time.perf_counter() - t0) * 1000
                    pred = int(probs.argmax())
                    print(f"  image {i:5d}: truth {classes[test_y[i]]:15s} pred {classes[pred]:15s} "
                          f"{probs[pred] * 100:6.2f}%  {ms:6.2f} ms  {'ok' if pred == test_y[i] else 'WRONG'}")

            if args.evaluate:
                preds, t0 = [], time.perf_counter()
                for s in range(0, len(test_x), 256):
                    preds.append(sess.run(None, {"input": preprocess_np(test_x[s:s + 256], dataset, res)})[0].argmax(1))
                sec = time.perf_counter() - t0
                acc = float((np.concatenate(preds) == test_y).mean() * 100)
                rows.append({"model": name, "dataset": dataset, "test_top1": acc, "threads": args.threads,
                             "seconds": round(sec, 2), "images_per_s": round(len(test_y) / sec, 1)})
                print(f"  test top-1 {acc:.2f}% on {len(test_y)} images in {sec:.1f} s "
                      f"({len(test_y) / sec:.0f} img/s, batch 256)")
        if rows:
            write_csv(rows, RESULTS_DIR / "inference_eval_onnx.csv")


if __name__ == "__main__":
    main()
