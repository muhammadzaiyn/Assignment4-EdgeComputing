"""Generate every figure of the technical report from the saved result files.

Nothing here is typed by hand: each plot reads a CSV/JSON written by the experiment scripts.
Figures go to results/figures/ (PNG, 200 dpi). Console-capture images of the real run logs are
rendered to report/screenshots/console_*.png (clearly labelled as captured console output).

Run:  python scripts/11_make_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import FIG_DIR, LOG_DIR, REPORT_DIR, RESULTS_DIR, load_json

# fixed categorical order (colour-blind-checked reference palette), never cycled
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
     "grey": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e"}
MODE_COLORS = {"baseline": C["blue"], "kd_soft": C["aqua"], "kd": C["orange"]}
MODE_LABELS = {"baseline": "Baseline (no KD)", "kd_soft": "KD: soft targets only", "kd": "KD: soft + feature"}
ARCHS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
ARCH_NAMES = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet 1.1", "efficientnet_lite0": "EfficientNet-Lite0"}
SHOTS = REPORT_DIR / "screenshots"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ad",
    "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def exists(*paths):
    return all(Path(p).exists() for p in paths)


# ----------------------------------------------------------------------------- Q1 figures
def fig_teacher_curve():
    p = RESULTS_DIR / "teacher_history.csv"
    if not exists(p):
        return
    h = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.epoch, h.train_loss, color=C["blue"], label="train loss")
    ax[0].plot(h.epoch, h.val_loss, color=C["orange"], label="validation loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="ResNet-50 teacher: loss")
    ax[0].legend(frameon=False)
    ax[1].plot(h.epoch, h.train_acc, color=C["blue"], label="train top-1")
    ax[1].plot(h.epoch, h.val_acc, color=C["orange"], label="validation top-1")
    best = h.val_acc.idxmax()
    ax[1].scatter([h.epoch[best]], [h.val_acc[best]], s=50, color=C["orange"], zorder=3, edgecolor="white", linewidth=2)
    ax[1].annotate(f"best {h.val_acc[best]:.2f}%", (h.epoch[best], h.val_acc[best]), textcoords="offset points",
                   xytext=(-70, -22), color=C["ink2"])
    ax[1].set(xlabel="epoch", ylabel="accuracy (%)", title="ResNet-50 teacher: accuracy")
    ax[1].legend(frameon=False, loc="lower right")
    save(fig, "q1_teacher_training.png")


def fig_student_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    any_data = False
    for ax, arch in zip(axes, ARCHS):
        for mode in ("baseline", "kd_soft", "kd"):
            p = RESULTS_DIR / "students" / f"{arch}_{mode}_history.csv"
            if exists(p):
                h = pd.read_csv(p)
                ax.plot(h.epoch, h.val_acc, color=MODE_COLORS[mode], label=MODE_LABELS[mode],
                        linewidth=2 if mode != "kd_soft" else 1.6)
                any_data = True
        ax.set(title=ARCH_NAMES[arch], xlabel="epoch")
    axes[0].set_ylabel("validation top-1 (%)")
    axes[0].legend(frameon=False, loc="lower right")
    if any_data:
        save(fig, "q1_student_val_curves.png")


def fig_kd_bars():
    p = RESULTS_DIR / "kd_comparison.csv"
    if not exists(p):
        return
    d = pd.read_csv(p)
    has_soft = "kd_soft_only_top1" in d and d.kd_soft_only_top1.notna().any()
    modes = [("baseline_top1", "baseline")] + ([("kd_soft_only_top1", "kd_soft")] if has_soft else []) + [("kd_full_top1", "kd")]
    x = np.arange(len(d))
    w = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (col, mode) in enumerate(modes):
        bars = ax.bar(x + (i - (len(modes) - 1) / 2) * w, d[col], w * 0.92, color=MODE_COLORS[mode],
                      label=MODE_LABELS[mode])
        for b, v in zip(bars, d[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=C["ink2"])
    t = d.teacher_top1.iloc[0]
    ax.axhline(t, color=C["ink2"], linestyle="--", linewidth=1.4)
    ax.text(len(d) - 0.5, t + 0.5, f"ResNet-50 teacher {t:.1f}%", ha="right", va="bottom", color=C["ink2"])
    ax.set_xticks(x, [ARCH_NAMES[a] for a in d.arch])
    lo = min(d[c].min() for c, _ in modes)
    ax.set_ylim(max(0, lo - 12), t + 6)
    ax.set(ylabel="CIFAR-100 test top-1 (%)", title="Effect of knowledge distillation on each student")
    ax.legend(frameon=False, loc="upper left", ncol=len(modes))
    save(fig, "q1_kd_vs_baseline.png")


def fig_kd_loss_components():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    ok = False
    for ax, arch in zip(axes, ARCHS):
        p = RESULTS_DIR / "students" / f"{arch}_kd_history.csv"
        if not exists(p):
            continue
        h = pd.read_csv(p)
        ok = True
        for col, color, lab in (("hard_ce", C["blue"], "hard CE"), ("soft_kl", C["orange"], "T² · KL (soft)"),
                                ("feat_mse", C["aqua"], "feature MSE (unweighted; x beta in the loss)")):
            if col in h:
                ax.plot(h.epoch, h[col], color=color, label=lab)
        ax.set(title=ARCH_NAMES[arch], xlabel="epoch", yscale="log")
    axes[0].set_ylabel("loss term (train, log scale)")
    axes[0].legend(frameon=False)
    if ok:
        save(fig, "q1_kd_loss_terms.png")


def fig_q1_tradeoff():
    p = RESULTS_DIR / "benchmark_summary.csv"
    if not exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.dataset == "cifar100"]
    fig, ax = plt.subplots(figsize=(9, 5))
    arch_color = {"mobilenetv2": C["blue"], "squeezenet": C["aqua"], "efficientnet_lite0": C["violet"], "teacher": C["ink2"]}
    for _, r in d.iterrows():
        arch = "teacher" if r.model == "teacher" else r.model.rsplit("_", 1)[0]
        kd = r.model.endswith("_kd")
        ax.scatter(r.pt_p50_ms, r.test_top1, s=40 + 30 * r.params_m, color=arch_color[arch],
                   marker="o" if (kd or arch == "teacher") else "s", edgecolor="white", linewidth=2,
                   alpha=1 if kd or arch == "teacher" else 0.55, zorder=3)
        # MobileNetV2 and EfficientNet-Lite0 sit close together: label one on the left, one on the right
        left = arch in ("mobilenetv2", "teacher")
        ax.annotate(r.display, (r.pt_p50_ms, r.test_top1), textcoords="offset points",
                    xytext=(-10 if left else 10, -3), ha="right" if left else "left", fontsize=9, color=C["ink2"])
    for arch in ARCHS:
        b = d[d.model == f"{arch}_baseline"]
        k = d[d.model == f"{arch}_kd"]
        if len(b) and len(k):
            ax.annotate("", xy=(k.pt_p50_ms.iloc[0], k.test_top1.iloc[0]), xytext=(b.pt_p50_ms.iloc[0], b.test_top1.iloc[0]),
                        arrowprops=dict(arrowstyle="->", color=arch_color[arch], lw=1.5))
    ax.set_xscale("log")
    ax.set_xlim(1.2, 60)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    ax.set(xlabel="PyTorch CPU latency, batch 1, 1 thread (ms, log scale)", ylabel="CIFAR-100 test top-1 (%)",
           title="Accuracy vs CPU latency (marker area ~ parameters; arrow = baseline -> KD)")
    save(fig, "q1_accuracy_vs_latency.png")


def fig_latency_bars():
    p = RESULTS_DIR / "benchmark_summary.csv"
    if not exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.ort_p50_ms.notna()]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(d) + 1.4))
    ax.barh(y + 0.2, d.pt_p50_ms, 0.38, color=C["blue"], label="PyTorch eager (FP32)")
    ax.barh(y - 0.2, d.ort_p50_ms, 0.38, color=C["orange"], label="ONNX Runtime (FP32)")
    for i, (a, b) in enumerate(zip(d.pt_p50_ms, d.ort_p50_ms)):
        ax.text(a, i + 0.2, f" {a:.2f}", va="center", fontsize=9, color=C["ink2"])
        ax.text(b, i - 0.2, f" {b:.2f}", va="center", fontsize=9, color=C["ink2"])
    ax.set_yticks(y, d.display)
    ax.invert_yaxis()
    ax.set(xlabel="median latency, batch 1, 1 CPU thread (ms)", title="CPU inference latency: PyTorch vs ONNX Runtime")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "latency_pytorch_vs_onnx.png")


def fig_agreement():
    p = RESULTS_DIR / "kd_comparison.csv"
    if not exists(p):
        return
    d = pd.read_csv(p)
    if "kd_teacher_agreement_pct" not in d:
        return
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, d.baseline_teacher_agreement_pct, 0.38, color=C["blue"], label=MODE_LABELS["baseline"])
    ax.bar(x + 0.2, d.kd_teacher_agreement_pct, 0.38, color=C["orange"], label=MODE_LABELS["kd"])
    for i, (a, b) in enumerate(zip(d.baseline_teacher_agreement_pct, d.kd_teacher_agreement_pct)):
        ax.text(i - 0.2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9, color=C["ink2"])
        ax.text(i + 0.2, b + 0.5, f"{b:.1f}", ha="center", fontsize=9, color=C["ink2"])
    ax.set_xticks(x, [ARCH_NAMES[a] for a in d.arch])
    ax.set_ylim(min(d.baseline_teacher_agreement_pct.min(), d.kd_teacher_agreement_pct.min()) - 10, 100)
    ax.set(ylabel="same top-1 as teacher (%)", title="How often the student agrees with the teacher (test set)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "q1_teacher_agreement.png")


# ----------------------------------------------------------------------------- Q2 figures
def nas_history():
    p = RESULTS_DIR / "nas" / "search_history.csv"
    return pd.read_csv(p) if exists(p) else None


def fig_nas_progress():
    h = nas_history()
    if h is None:
        return
    # top panel: accuracy of trained trials; bottom strip: status of every trial (incl. rejected ones)
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [4, 0.7]})
    styles = {"complete": (C["blue"], "o", "trained to completion"),
              "pruned": (C["yellow"], "v", "pruned early (best epoch shown)"),
              "duplicate": (C["violet"], "D", "duplicate proposal"),
              "rejected": (C["red"], "x", "rejected by constraints (not trained)")}
    for status, (color, marker, lab) in styles.items():
        s = h[h.status == status]
        if not len(s):
            continue
        if status != "rejected":
            ax.scatter(s.trial, s.val_acc, color=color, marker=marker, s=40, zorder=3, edgecolor="white", linewidth=0.8)
        strip.scatter(s.trial, np.zeros(len(s)), color=color, marker=marker, s=34,
                      linewidths=1.6 if marker == "x" else 0.6, edgecolor=None if marker == "x" else "white",
                      label=f"{lab} ({len(s)})")
    comp = h[h.status == "complete"].sort_values("trial")
    if len(comp):
        ax.step(list(comp.trial) + [h.trial.max()], list(comp.val_acc.cummax()) + [comp.val_acc.max()],
                where="post", color=C["orange"], label="best so far")
    best = load_json(RESULTS_DIR / "nas" / "best_architecture.json")
    sel = h[h.trial == best["selected_trial"]]
    ax.scatter(sel.trial, sel.val_acc, s=170, facecolor="none", edgecolor=C["ink"], linewidth=2, zorder=4,
               label=f"selected (trial {best['selected_trial']})")
    trained = h[h.status != "rejected"].val_acc.dropna()
    ax.set_ylim(trained.min() - 2, trained.max() + 2)
    ax.set(ylabel="proxy validation top-1 (%)", title="NAS search progress (Optuna TPE, CIFAR-10)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    strip.set_yticks([])
    strip.set_ylim(-1, 1)
    strip.grid(False)
    strip.set_xlabel("trial")
    strip.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.9), ncol=4)
    save(fig, "q2_nas_search_progress.png")


def fig_nas_cost_space():
    h = nas_history()
    if h is None:
        return
    best = load_json(RESULTS_DIR / "nas" / "best_architecture.json")
    cons = best["constraints"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    feas = h[h.feasible == True]
    infe = h[h.feasible != True]
    for a, xcol, ycol, xl, yl, xlim_c, ylim_c in (
            (ax[0], "cpu_latency_ms", "size_mb", "CPU latency, batch 1, 1 thread (ms)", "FP32 size on disk (MB)",
             cons["max_latency_ms"], cons["max_size_mb"]),
            (ax[1], "cpu_latency_ms", "peak_mem_mb", "CPU latency, batch 1, 1 thread (ms)", "peak inference memory (MB)",
             cons["max_latency_ms"], cons["max_peak_mem_mb"])):
        a.scatter(infe[xcol], infe[ycol], color=C["red"], marker="x", s=30, label=f"rejected ({len(infe)})", linewidths=1.5)
        a.scatter(feas[xcol], feas[ycol], color=C["blue"], s=36, edgecolor="white", linewidth=1, label=f"feasible ({len(feas)})", zorder=3)
        sel = h[h.trial == best["selected_trial"]]
        a.scatter(sel[xcol], sel[ycol], s=170, facecolor="none", edgecolor=C["ink"], linewidth=2, zorder=4, label="selected")
        a.axvline(xlim_c, color=C["ink2"], linestyle="--", linewidth=1.2)
        a.axhline(ylim_c, color=C["ink2"], linestyle="--", linewidth=1.2)
        a.fill_between([0, xlim_c], 0, ylim_c, color=C["aqua"], alpha=0.08, zorder=0)
        a.set(xlabel=xl, ylabel=yl, xscale="log", yscale="log")
        for axis in (a.xaxis, a.yaxis):            # plain numbers instead of 2x10^0 style ticks
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    ax[0].set_title("Measured cost of every candidate vs constraints")
    ax[1].set_title("Latency vs peak memory")
    ax[0].legend(frameon=False, loc="upper left")
    save(fig, "q2_nas_constraints.png")


def fig_nas_pareto():
    h = nas_history()
    if h is None:
        return
    best = load_json(RESULTS_DIR / "nas" / "best_architecture.json")
    comp = h[(h.status == "complete") & (h.feasible == True)]
    front = pd.read_csv(RESULTS_DIR / "nas" / "pareto_front.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sc = ax.scatter(comp.cpu_latency_ms, comp.val_acc, c=comp.size_mb, cmap="Blues", vmin=0, s=50,
                    edgecolor=C["ink2"], linewidth=0.6, zorder=3)
    fig.colorbar(sc, ax=ax, label="size (MB)")
    ax.step(front.cpu_latency_ms, front.val_acc, where="post", color=C["orange"], label="Pareto front")
    sel = comp[comp.trial == best["selected_trial"]]
    ax.scatter(sel.cpu_latency_ms, sel.val_acc, s=180, facecolor="none", edgecolor=C["ink"], linewidth=2, zorder=4,
               label=f"selected (trial {best['selected_trial']})")
    ax.axvline(best["constraints"]["max_latency_ms"], color=C["ink2"], linestyle="--", linewidth=1.2)
    ax.set(xlabel="CPU latency, batch 1, 1 thread (ms)", ylabel="proxy validation top-1 (%)",
           title="Feasible trained candidates: accuracy vs latency")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "q2_nas_pareto.png")


def fig_nas_factors():
    h = nas_history()
    if h is None:
        return
    comp = h[h.status == "complete"]
    if not len(comp):
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for a, col, title in ((ax[0], "activation", "activation"), (ax[1], "resolution", "input resolution"),
                          (ax[2], "num_stages", "number of stages")):
        groups = sorted(comp[col].unique(), key=str)
        data = [comp[comp[col] == g].val_acc.values for g in groups]
        a.boxplot(data, tick_labels=[str(g) for g in groups], widths=0.5,
                  medianprops=dict(color=C["orange"], linewidth=2), boxprops=dict(color=C["ink2"]),
                  whiskerprops=dict(color=C["ink2"]), capprops=dict(color=C["ink2"]), flierprops=dict(markeredgecolor=C["grey"]))
        for i, vals in enumerate(data):
            a.scatter(np.full(len(vals), i + 1) + np.random.RandomState(0).uniform(-0.12, 0.12, len(vals)), vals,
                      s=14, color=C["blue"], alpha=0.6, zorder=3)
        a.set(title=f"by {title}", xlabel=title)
    ax[0].set_ylabel("proxy validation top-1 (%)")
    save(fig, "q2_nas_factors.png")


def fig_nas_final_curve():
    p = RESULTS_DIR / "nas" / "final_history.csv"
    if not exists(p):
        return
    h = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.epoch, h.train_loss, color=C["blue"], label="train loss")
    ax[0].plot(h.epoch, h.val_loss, color=C["orange"], label="validation loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="NAS-best trained from scratch: loss")
    ax[0].legend(frameon=False)
    ax[1].plot(h.epoch, h.train_acc, color=C["blue"], label="train top-1")
    ax[1].plot(h.epoch, h.val_acc, color=C["orange"], label="validation top-1")
    ax[1].set(xlabel="epoch", ylabel="accuracy (%)", title="NAS-best: accuracy")
    ax[1].legend(frameon=False, loc="lower right")
    save(fig, "q2_nas_final_training.png")


# ----------------------------------------------------------------------------- console captures
def render_console(log_name, out_name, title, max_lines=48, filt=None, tail=True):
    """Render the real text of a run log as a dark terminal-style PNG."""
    p = LOG_DIR / log_name
    if not p.exists():
        return
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if filt:
        lines = [l for l in lines if filt(l)]
    lines = lines[-max_lines:] if tail else lines[:max_lines]
    lines = [l[:150] for l in lines]
    h = 0.19 * len(lines) + 0.7
    fig = plt.figure(figsize=(14, h))
    fig.patch.set_facecolor("#1a1a19")
    fig.text(0.01, 1 - 0.25 / h, f"{title}   (console output captured from results/logs/{log_name})",
             color="#9cc2f5", family="monospace", fontsize=9, va="top")
    fig.text(0.01, 1 - 0.55 / h, "\n".join(lines), color="#f0efec", family="monospace", fontsize=8.2,
             va="top", linespacing=1.35)
    SHOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(SHOTS / out_name, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote report/screenshots/{out_name}")


def console_figures():
    keep = lambda l: not l.startswith("    it ")
    render_console("02_train_teacher.log", "console_teacher.png", "Teacher training (ResNet-50, CIFAR-100)", 30, keep)
    render_console("03_baseline_mobilenetv2.log", "console_baseline_mobilenetv2.png", "Baseline student training", 24, keep)
    render_console("04_kd_mobilenetv2.log", "console_kd_mobilenetv2.png", "Knowledge distillation (MobileNetV2)", 26, keep)
    render_console("05_nas_search.log", "console_nas_search_start.png", "NAS search (first trials)", 42, tail=False)
    render_console("05_nas_search.log", "console_nas_search_summary.png", "NAS search summary", 30)
    render_console("06_nas_train_best.log", "console_nas_final.png", "Training the selected architecture", 22)
    render_console("07_export_onnx.log", "console_onnx_export.png", "ONNX export + parity", 20)
    render_console("08_inference.log", "console_inference_pytorch.png", "Unified PyTorch inference", 44)
    render_console("09_onnx_inference.log", "console_inference_onnx.png", "ONNX Runtime CPU inference", 40)
    render_console("10_benchmark.log", "console_benchmark.png", "Automated benchmark", 34)


def main():
    print("Q1 figures")
    fig_teacher_curve()
    fig_student_curves()
    fig_kd_bars()
    fig_kd_loss_components()
    fig_q1_tradeoff()
    fig_agreement()
    fig_latency_bars()
    print("Q2 figures")
    fig_nas_progress()
    fig_nas_cost_space()
    fig_nas_pareto()
    fig_nas_factors()
    fig_nas_final_curve()
    print("console captures")
    console_figures()


if __name__ == "__main__":
    main()
