"""Automated benchmark of every trained model.                            (deliverable 11)

No arguments are needed. For the teacher, the six students and the NAS model it measures

  accuracy   top-1 / top-5 on the full official test set (10,000 images), FP32 PyTorch, and
             top-1 of the exported ONNX model under ONNX Runtime on the CPU
  size       parameters, MACs, checkpoint size (.pth) and ONNX file size
  latency    batch-1 CPU latency, 1 thread, for PyTorch eager and ONNX Runtime:
             50 warm-up runs, then 300 timed runs on real test images; the whole measurement is
             repeated 3 times and the median of each statistic is kept (robust to one noisy run)
  throughput ONNX Runtime images/second at batch 128 using all physical cores
  memory     peak batch-1 inference memory (CUDA allocator, exact) and CPU RSS growth

and derives the knowledge-distillation comparison (baseline vs KD vs teacher), including how
often each student agrees with the teacher's top-1 prediction.

Outputs
  results/benchmark_summary.csv      one row per model, all metrics above
  results/kd_comparison.csv          per student: baseline, KD (and soft-only ablation) accuracy, gains
  results/predictions_cifar100.csv   per test image: label + prediction/confidence of every CIFAR-100 model
  results/predictions_cifar10.csv    per test image: label + NAS model prediction/confidence
  results/per_class_accuracy_cifar100.csv
  results/benchmark_settings.json    measurement settings + machine description
  results/logs/10_benchmark.log

Run:  python scripts/10_benchmark.py            (about 10 minutes; close other heavy programs first)
      python scripts/10_benchmark.py --runs 100 --repeats 1     (faster, noisier)
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil
import torch

from src.common import (CKPT_DIR, LOG_DIR, ONNX_DIR, RESULTS_DIR, Tee, banner, file_size_mb, load_json,
                        save_json, system_info, write_csv)
from src.data import load_arrays, normalise_uint8_batch
from src.infer_utils import ALL_MODELS, REFERENCE_MODELS, load_model, ort_session
from src.metrics import count_macs, count_params, measure_cpu_latency, measure_ort_latency, measure_peak_memory_cuda
from src.models import STUDENTS

STAT_KEYS = ("mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--models", nargs="*", default=REFERENCE_MODELS + ALL_MODELS)
    return p.parse_args()


def median_stats(list_of_stats):
    return {k: statistics.median(s[k] for s in list_of_stats) for k in STAT_KEYS}


@torch.inference_mode()
def full_test_logits(model, dataset, res, device):
    """FP32 logits for the whole test set (no autocast, so these are the deployment numbers)."""
    _, _, test_x, test_y, _ = load_arrays(dataset)
    model = model.to(device)
    out = []
    for s in range(0, len(test_x), 500):
        x = normalise_uint8_batch(test_x[s:s + 500], dataset, res).to(device)
        out.append(model(x).float().cpu())
    model.cpu()
    return torch.cat(out).numpy(), test_y


def onnx_accuracy(path, dataset, res):
    sess = ort_session(path, threads=psutil.cpu_count(logical=False))
    _, _, test_x, test_y, _ = load_arrays(dataset)
    preds = []
    for s in range(0, len(test_x), 500):
        x = normalise_uint8_batch(test_x[s:s + 500], dataset, res).numpy()
        preds.append(sess.run(None, {"input": x})[0].argmax(1))
    return float((np.concatenate(preds) == test_y).mean() * 100)


def ort_throughput(path, x_np, batch=128, seconds=3.0):
    """Images/second with batch 128 and all physical cores."""
    sess = ort_session(path, threads=psutil.cpu_count(logical=False))
    xb = np.repeat(x_np, int(np.ceil(batch / len(x_np))), axis=0)[:batch]
    for _ in range(3):
        sess.run(None, {"input": xb})
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        sess.run(None, {"input": xb})
        n += batch
    return n / (time.perf_counter() - t0)


def main():
    args = parse_args()
    with Tee(LOG_DIR / "10_benchmark.log"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        info = system_info()
        load = psutil.cpu_percent(interval=2.0)
        banner("Benchmark settings")
        print(f"CPU {info['cpu']} | {info['cpu_physical_cores']} cores / {info['cpu_logical_cores']} threads | "
              f"torch {info['torch']} | onnxruntime {info['onnxruntime']}")
        print(f"latency: batch 1, 1 thread, {args.warmup} warm-up + {args.runs} timed runs, "
              f"median over {args.repeats} repeats | throughput: ORT batch 128, "
              f"{info['cpu_physical_cores']} threads")
        print(f"CPU load before start: {load:.0f}%" + ("  (machine looks busy, numbers may be noisy)" if load > 20 else ""))
        save_json({"warmup": args.warmup, "runs": args.runs, "repeats": args.repeats, "batch_latency": 1,
                   "threads_latency": 1, "throughput_batch": 128, "throughput_threads": info["cpu_physical_cores"],
                   "cpu_load_before_pct": load, "system": info}, RESULTS_DIR / "benchmark_settings.json")

        rows, logits = [], {}
        loaded = {}
        for name in args.models:
            model, mi = load_model(name, "cpu")
            loaded[name] = (model, mi)

        # ------------------------------------------------------------ accuracy + static metrics
        banner("Accuracy and size")
        for name, (model, mi) in loaded.items():
            lg, y = full_test_logits(model, mi.dataset, mi.resolution, dev)
            logits[name] = (lg, y)
            top5 = np.argsort(-lg, axis=1)[:, :5]
            ckpt_state = torch.load(mi.checkpoint, map_location="cpu", weights_only=False)
            row = {"model": name, "display": mi.display, "dataset": mi.dataset, "input_res": mi.resolution,
                   "test_top1": float((lg.argmax(1) == y).mean() * 100),
                   "test_top5": float((top5 == y[:, None]).any(1).mean() * 100),
                   "params_m": count_params(model) / 1e6, "macs_m": count_macs(model, mi.resolution) / 1e6,
                   "pth_size_mb": file_size_mb(mi.checkpoint),
                   "onnx_size_mb": file_size_mb(mi.onnx_path) if mi.onnx_path.exists() else None,
                   "best_epoch": ckpt_state.get("epoch")}
            row["onnx_top1"] = onnx_accuracy(mi.onnx_path, mi.dataset, mi.resolution) if mi.onnx_path.exists() else None
            mem = measure_peak_memory_cuda(model, mi.resolution)
            row["peak_mem_mb"] = mem["peak_mem_mb"]
            row["activation_mem_mb"] = mem["activation_mem_mb"]
            rows.append(row)
            print(f"{mi.display:34s} top-1 {row['test_top1']:6.2f}% (ONNX {row['onnx_top1'] if row['onnx_top1'] is None else f'{row['onnx_top1']:.2f}%'}) | "
                  f"top-5 {row['test_top5']:6.2f}% | params {row['params_m']:6.3f} M | MACs {row['macs_m']:8.1f} M | "
                  f"pth {row['pth_size_mb']:7.2f} MB | peak mem {row['peak_mem_mb']:.1f} MB")

        # ------------------------------------------------------------ latency (repeated, fixed order)
        banner("CPU latency (batch 1, 1 thread)")
        samples = {}
        for name, (model, mi) in loaded.items():
            _, _, test_x, _, _ = load_arrays(mi.dataset)
            samples[name] = normalise_uint8_batch(test_x[:64], mi.dataset, mi.resolution)
        pt_runs = {n: [] for n in loaded}
        ort_runs = {n: [] for n in loaded}
        sessions = {n: ort_session(mi.onnx_path, threads=1) for n, (_, mi) in loaded.items() if mi.onnx_path.exists()}
        for rep in range(args.repeats):
            for name, (model, mi) in loaded.items():
                pt_runs[name].append(measure_cpu_latency(model, mi.resolution, threads=1, warmup=args.warmup,
                                                         runs=args.runs, inputs=samples[name]))
                if name in sessions:
                    ort_runs[name].append(measure_ort_latency(sessions[name], samples[name].numpy(),
                                                              warmup=args.warmup, runs=args.runs))
            print(f"  repeat {rep + 1}/{args.repeats} done")

        for row in rows:
            name = row["model"]
            pt = median_stats(pt_runs[name])
            row.update({f"pt_{k}": v for k, v in pt.items()})
            if name in sessions:
                ort = median_stats(ort_runs[name])
                row.update({f"ort_{k}": v for k, v in ort.items()})
                row["ort_throughput_img_s"] = ort_throughput(loaded[name][1].onnx_path, samples[name].numpy())
            print(f"{row['display']:34s} PyTorch p50 {pt['p50_ms']:7.3f} ms (p99 {pt['p99_ms']:7.3f}) | "
                  + (f"ORT p50 {row['ort_p50_ms']:7.3f} ms (p99 {row['ort_p99_ms']:7.3f}) | "
                     f"ORT throughput {row['ort_throughput_img_s']:8.0f} img/s" if name in sessions else "no ONNX (reference)"))

        # ------------------------------------------------------------ relative numbers vs teacher
        teacher = next((r for r in rows if r["model"] == "teacher"), None)
        for row in rows:
            if teacher and row["dataset"] == "cifar100":
                row["param_compression_vs_teacher"] = teacher["params_m"] / row["params_m"]
                row["pt_speedup_vs_teacher"] = teacher["pt_p50_ms"] / row["pt_p50_ms"]
                row["acc_gap_to_teacher"] = row["test_top1"] - teacher["test_top1"]
        cols = ["model", "display", "dataset", "input_res", "test_top1", "test_top5", "onnx_top1", "params_m",
                "macs_m", "pth_size_mb", "onnx_size_mb", "peak_mem_mb", "activation_mem_mb",
                *[f"pt_{k}" for k in STAT_KEYS], *[f"ort_{k}" for k in STAT_KEYS], "ort_throughput_img_s",
                "param_compression_vs_teacher", "pt_speedup_vs_teacher", "acc_gap_to_teacher", "best_epoch"]
        rows = [{c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c)) for c in cols} for r in rows]
        write_csv(rows, RESULTS_DIR / "benchmark_summary.csv", fieldnames=cols)

        # ------------------------------------------------------------ KD comparison + predictions
        banner("Knowledge distillation: baseline vs distilled students")
        kd_rows = []
        t_pred = logits["teacher"][0].argmax(1) if "teacher" in logits else None
        by = {r["model"]: r for r in rows}
        for arch in STUDENTS:
            b, k = by.get(f"{arch}_baseline"), by.get(f"{arch}_kd")
            if not (b and k):
                continue
            # ablation checkpoint (soft targets only): re-evaluated here in FP32 like every other model
            soft = None
            soft_ckpt = CKPT_DIR / "kd" / "ablation" / f"{arch}_kd_soft.pth"
            if soft_ckpt.exists():
                from src.models import build_student
                m = build_student(arch)
                m.load_state_dict(torch.load(soft_ckpt, map_location="cpu", weights_only=False)["state_dict"])
                lg_s, y_s = full_test_logits(m.eval(), "cifar100", 32, dev)
                soft = float((lg_s.argmax(1) == y_s).mean() * 100)
            row = {"arch": arch, "params_m": b["params_m"], "baseline_top1": b["test_top1"],
                   "kd_soft_only_top1": soft, "kd_full_top1": k["test_top1"],
                   "kd_gain_points": k["test_top1"] - b["test_top1"],
                   "baseline_top5": b["test_top5"], "kd_full_top5": k["test_top5"],
                   "teacher_top1": teacher["test_top1"] if teacher else None}
            if t_pred is not None:
                row["baseline_teacher_agreement_pct"] = float((logits[f"{arch}_baseline"][0].argmax(1) == t_pred).mean() * 100)
                row["kd_teacher_agreement_pct"] = float((logits[f"{arch}_kd"][0].argmax(1) == t_pred).mean() * 100)
            kd_rows.append(row)
            print(f"{arch:20s} baseline {row['baseline_top1']:6.2f}% | "
                  + (f"KD soft-only {soft:6.2f}% | " if soft is not None else "")
                  + f"KD full {row['kd_full_top1']:6.2f}% | gain {row['kd_gain_points']:+.2f} pts | "
                  f"teacher agreement {row.get('baseline_teacher_agreement_pct', 0):.1f}% -> "
                  f"{row.get('kd_teacher_agreement_pct', 0):.1f}%")
        write_csv(kd_rows, RESULTS_DIR / "kd_comparison.csv")

        # per-image predictions and per-class accuracy
        for dataset in ("cifar100", "cifar10"):
            names = [n for n in logits if loaded[n][1].dataset == dataset]
            if not names:
                continue
            y = logits[names[0]][1]
            classes = loaded[names[0]][1].classes
            pred_rows = []
            probs = {}
            for n in names:
                lg = logits[n][0]
                e = np.exp(lg - lg.max(1, keepdims=True))
                probs[n] = e / e.sum(1, keepdims=True)
            for i in range(len(y)):
                r = {"image_id": i, "true_label": int(y[i]), "true_name": classes[y[i]]}
                for n in names:
                    p = int(probs[n][i].argmax())
                    r[f"{n}_pred"] = p
                    r[f"{n}_conf"] = round(float(probs[n][i, p]), 4)
                    r[f"{n}_correct"] = int(p == y[i])
                pred_rows.append(r)
            write_csv(pred_rows, RESULTS_DIR / f"predictions_{dataset}.csv")
            if dataset == "cifar100":
                pc = []
                for c, cname in enumerate(classes):
                    m = y == c
                    pc.append({"class_id": c, "class": cname,
                               **{n: round(float((probs[n][m].argmax(1) == c).mean() * 100), 2) for n in names}})
                write_csv(pc, RESULTS_DIR / "per_class_accuracy_cifar100.csv")

        banner("Summary table")
        print(f"{'model':34s} {'top-1':>7s} {'params':>8s} {'MACs':>8s} {'size':>8s} {'PT p50':>8s} {'ORT p50':>8s}")
        for r in rows:
            print(f"{r['display']:34s} {r['test_top1']:6.2f}% {r['params_m']:7.3f}M {r['macs_m']:7.1f}M "
                  f"{r['pth_size_mb']:6.2f}MB {r['pt_p50_ms']:6.3f}ms "
                  + (f"{r['ort_p50_ms']:6.3f}ms" if r.get('ort_p50_ms') is not None else "     -"))
        print(f"\nwrote {RESULTS_DIR / 'benchmark_summary.csv'} and {RESULTS_DIR / 'kd_comparison.csv'}")


if __name__ == "__main__":
    main()
