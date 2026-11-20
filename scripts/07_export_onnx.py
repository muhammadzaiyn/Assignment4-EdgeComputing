"""Q1.7 / Q2.8 - Export the trained student models and the NAS model to ONNX.   (deliverable 10)

For every deployable model (3 baseline students, 3 distilled students, NAS-best):
  1. rebuild it from its checkpoint (FP32, eval mode, on the CPU)
  2. export with opset 17 and a dynamic batch dimension (input "input", output "logits");
     BatchNorm is folded into the convolutions by constant folding
  3. validate the file with onnx.checker
  4. compare ONNX Runtime (CPU) against PyTorch on 512 real test images:
     max |logit difference| and the fraction of identical top-1 predictions

Outputs
  onnx_models/<model>.onnx
  results/onnx_export.csv          size, parity numbers and opset for every model
  results/logs/07_export_onnx.log

Run:  python scripts/07_export_onnx.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import torch

from src.common import LOG_DIR, ONNX_DIR, RESULTS_DIR, Tee, banner, file_size_mb, write_csv
from src.data import load_arrays, normalise_uint8_batch
from src.infer_utils import ALL_MODELS, load_model, ort_session

OPSET = 17


def export(model, resolution, path):
    example = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(model, example, str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=ALL_MODELS)
    args = p.parse_args()

    with Tee(LOG_DIR / "07_export_onnx.log"):
        banner(f"ONNX export (opset {OPSET}) and ONNX Runtime parity check")
        rows = []
        for name in args.models:
            model, info = load_model(name, "cpu")
            path = ONNX_DIR / f"{name}.onnx"
            export(model, info.resolution, path)

            # parity check on 512 real test images of the model's own dataset
            _, _, test_x, _, _ = load_arrays(info.dataset)
            x = normalise_uint8_batch(test_x[:512], info.dataset, info.resolution)
            with torch.inference_mode():
                ref = model(x).numpy()
            sess = ort_session(path, threads=4)
            out = sess.run(None, {"input": x.numpy()})[0]
            max_diff = float(np.abs(ref - out).max())
            agree = float((ref.argmax(1) == out.argmax(1)).mean() * 100)
            g = onnx.load(str(path)).graph
            row = {"model": name, "onnx_file": path.name, "onnx_size_mb": round(file_size_mb(path), 4),
                   "pth_size_mb": round(file_size_mb(info.checkpoint), 4), "opset": OPSET,
                   "input_resolution": info.resolution, "graph_nodes": len(g.node),
                   "max_abs_logit_diff": max_diff, "top1_agreement_pct": agree,
                   "parity_ok": bool(max_diff < 1e-3 and agree >= 99.8)}
            rows.append(row)
            print(f"{name:28s} -> {path.name:34s} {row['onnx_size_mb']:7.3f} MB | nodes {row['graph_nodes']:4d} | "
                  f"max|diff| {max_diff:.2e} | top-1 agreement {agree:6.2f}% | "
                  f"{'OK' if row['parity_ok'] else 'CHECK'}")
        write_csv(rows, RESULTS_DIR / "onnx_export.csv")
        print(f"\nall parity checks passed: {all(r['parity_ok'] for r in rows)}")
        print(f"wrote {RESULTS_DIR / 'onnx_export.csv'}")


if __name__ == "__main__":
    main()
