"""Q2 - Hardware-aware neural architecture search on CIFAR-10.         (deliverables 5, 6, 7)

Search strategy: Bayesian optimisation with Optuna's Tree-structured Parzen Estimator (TPE).

For every trial
  1. TPE proposes an architecture config from the search space (src/nas/search_space.py).
  2. The generator builds the network (src/nas/generator.py).
  3. Hardware profiling MEASURES its CPU latency, storage size and peak inference memory
     (src/nas/hardware.py).
  4. If any deployment constraint is violated the candidate is REJECTED without training
     (objective 0, flagged infeasible). Optuna's constraint-aware TPE (`constraints_func`) learns
     to avoid infeasible regions of the space.
  5. Feasible candidates go through the automated training/validation pipeline: a short proxy
     training on the 45k CIFAR-10 training split, scored by top-1 accuracy on the 5k validation
     split. A median pruner stops candidates that are clearly worse than earlier ones.

Selection: the best architecture is the feasible, fully-trained candidate with the highest
validation accuracy; if another feasible candidate is within `--tie-margin` points of it, the one
with the lower measured CPU latency wins (resource-constrained selection).

Outputs
  results/nas/search_history.csv / .json   every trial: config, measured cost, accuracy, status
  results/nas/best_architecture.json       selected config + its measurements + constraints
  results/nas/pareto_front.csv             feasible accuracy-vs-latency Pareto set
  results/nas/search_space.json            the full search space definition
  results/nas/nas_study.db                 Optuna study (SQLite) - allows resuming the search
  results/logs/05_nas_search.log

After the search the best architecture is trained from scratch automatically
(scripts/06_nas_train_best.py) unless --no-train-best is given.

Run:  python scripts/05_nas_search.py                  (full search, ~25-35 min)
      python scripts/05_nas_search.py --quick          (5 trials, 1 epoch)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
import torch

from src.common import LOG_DIR, RESULTS_DIR, SEED, Tee, banner, get_device, save_json, set_seed, write_csv
from src.data import get_loaders
from src.nas import DeploymentConstraints, build_network, config_id, config_summary, profile_architecture, suggest_config
from src.nas.search_space import search_space_description, search_space_size
from src.train_utils import fit

NAS_DIR = RESULTS_DIR / "nas"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=80, help="number of Optuna trials")
    p.add_argument("--timeout-min", type=float, default=50, help="stop the search after this many minutes")
    p.add_argument("--proxy-epochs", type=int, default=6, help="training epochs per candidate")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--max-latency-ms", type=float, default=2.0)
    p.add_argument("--max-size-mb", type=float, default=1.0)
    p.add_argument("--max-peak-mem-mb", type=float, default=8.0)
    p.add_argument("--tie-margin", type=float, default=0.3, help="accuracy points treated as a tie")
    p.add_argument("--no-train-best", action="store_true", help="do not start the final training")
    p.add_argument("--fresh", action="store_true", help="delete an existing study and start over")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


class NASObjective:
    """Callable objective for Optuna: profile -> reject or train -> validation accuracy."""

    def __init__(self, args, constraints, device, train_loader, val_loader):
        self.args, self.constraints, self.device = args, constraints, device
        self.train_loader, self.val_loader = train_loader, val_loader
        self.cache = {}           # config_id -> accuracy (identical configs are not retrained)

    def __call__(self, trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        cid = config_id(cfg)
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("config_id", cid)
        trial.set_user_attr("summary", config_summary(cfg))

        # ---- 1) hardware-aware profiling (measured, before any training)
        hw = profile_architecture(cfg)
        for k, v in hw.items():
            trial.set_user_attr(k, v)
        feasible, reasons = self.constraints.check(hw)
        violations = self.constraints.violations(hw)
        trial.set_user_attr("constraints", violations)
        if hasattr(trial, "set_constraint"):          # Optuna >= 5: constraint-aware TPE input
            for key, value in zip(("latency", "size", "peak_memory"), violations):
                trial.set_constraint(key, value)
        trial.set_user_attr("feasible", feasible)
        trial.set_user_attr("rejection_reasons", reasons)
        head = (f"trial {trial.number:3d} | {config_summary(cfg)}\n"
                f"          params {hw['params']:,} | size {hw['size_mb']:.3f} MB | "
                f"CPU {hw['cpu_latency_ms']:.2f} ms | peak mem {hw['peak_mem_mb']:.2f} MB | "
                f"MACs {hw['macs_m']:.1f} M")
        if not feasible:
            trial.set_user_attr("status", "rejected")
            print(head + f"\n          REJECTED: {'; '.join(reasons)}")
            return 0.0

        if cid in self.cache:                                    # duplicate proposal
            trial.set_user_attr("status", "duplicate")
            print(head + f"\n          duplicate of an earlier trial -> val {self.cache[cid]:.2f}%")
            return self.cache[cid]

        # ---- 2) automated proxy training + validation
        set_seed(SEED)
        model = build_network(cfg, num_classes=10)
        self.train_loader.resolution = self.val_loader.resolution = cfg["resolution"]
        t0 = time.time()

        def report(epoch, val_acc):                              # median-pruner hook
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            best, hist = fit(model, self.train_loader, self.val_loader, epochs=self.args.proxy_epochs,
                             lr=self.args.lr, weight_decay=5e-4, device=self.device, ckpt_path=None,
                             history_csv=None, config=cfg, warmup_epochs=1, trial_callback=report,
                             save_best=False)
        except optuna.TrialPruned:
            trial.set_user_attr("status", "pruned")
            trial.set_user_attr("train_seconds", time.time() - t0)
            print(head + "\n          pruned by the median rule (worse than earlier candidates)")
            raise
        trial.set_user_attr("status", "complete")
        trial.set_user_attr("train_seconds", time.time() - t0)
        self.cache[cid] = best
        print(head + f"\n          trained {self.args.proxy_epochs} ep in {time.time() - t0:.0f}s -> "
              f"val top-1 {best:.2f}%")
        return best


def constraints_func(trial):
    """Constraint values for constraint-aware TPE (<= 0 is feasible)."""
    return trial.user_attrs.get("constraints", [0.0, 0.0, 0.0])


def history_rows(study):
    """Flatten the study into one row per trial for CSV / JSON."""
    rows = []
    for t in study.trials:
        u = t.user_attrs
        if "config" not in u:
            continue
        cfg = u["config"]
        status = u.get("status", t.state.name.lower())
        rows.append({
            "trial": t.number, "status": status, "feasible": u.get("feasible"),
            "val_acc": t.value if status in ("complete", "duplicate") else (
                max(t.intermediate_values.values()) if t.intermediate_values else None),
            "epochs_trained": len(t.intermediate_values),
            "params": u.get("params"), "size_mb": u.get("size_mb"),
            "cpu_latency_ms": u.get("cpu_latency_ms"), "cpu_latency_p90_ms": u.get("cpu_latency_p90_ms"),
            "peak_mem_mb": u.get("peak_mem_mb"), "activation_mem_mb": u.get("activation_mem_mb"),
            "macs_m": u.get("macs_m"),
            "resolution": cfg["resolution"], "activation": cfg["activation"],
            "num_stages": len(cfg["stages"]), "total_blocks": sum(s["depth"] for s in cfg["stages"]),
            "blocks": "/".join(s["block"] for s in cfg["stages"]),
            "kernels": "/".join(str(s["kernel"]) for s in cfg["stages"]),
            "widths": "/".join(str(s["width"]) for s in cfg["stages"]),
            "train_seconds": u.get("train_seconds"), "rejection_reasons": "; ".join(u.get("rejection_reasons", [])),
            "config_id": u.get("config_id"), "summary": u.get("summary"), "config": json.dumps(cfg),
        })
    return rows


def select_best(rows, tie_margin):
    """Resource-constrained selection among feasible, fully trained candidates."""
    done = [r for r in rows if r["status"] == "complete" and r["feasible"]]
    if not done:
        raise RuntimeError("no feasible candidate finished training - relax the constraints")
    top = max(r["val_acc"] for r in done)
    near = [r for r in done if r["val_acc"] >= top - tie_margin]
    best = min(near, key=lambda r: r["cpu_latency_ms"])
    return best, top, near


def pareto_front(rows):
    """Feasible complete candidates not dominated in (higher accuracy, lower latency)."""
    done = sorted([r for r in rows if r["status"] == "complete" and r["feasible"]],
                  key=lambda r: (r["cpu_latency_ms"], -r["val_acc"]))
    front, best_acc = [], -1
    for r in done:
        if r["val_acc"] > best_acc:
            front.append(r)
            best_acc = r["val_acc"]
    return front


def main():
    args = parse_args()
    if args.quick:
        args.trials, args.proxy_epochs, args.timeout_min = 5, 1, 5
    suffix = "_quick" if args.quick else ""
    NAS_DIR.mkdir(parents=True, exist_ok=True)
    with Tee(LOG_DIR / f"05_nas_search{suffix}.log"):
        set_seed(SEED)
        device = get_device()
        constraints = DeploymentConstraints(args.max_latency_ms, args.max_size_mb, args.max_peak_mem_mb)
        banner("Hardware-aware NAS (Optuna TPE) on CIFAR-10")
        print(f"search space: {search_space_size():.3e} architectures")
        print(f"constraints : CPU latency <= {constraints.max_latency_ms} ms (batch 1, 1 thread) | "
              f"size <= {constraints.max_size_mb} MB | peak inference memory <= {constraints.max_peak_mem_mb} MB")
        print(f"proxy training: {args.proxy_epochs} epochs on 45k images, batch {args.batch_size}, "
              f"lr {args.lr}; ranking on the 5k validation split")
        print(f"budget: {args.trials} trials or {args.timeout_min} min\n")
        save_json(search_space_description(), NAS_DIR / "search_space.json")

        train, val, _, _ = get_loaders("cifar10", device, batch_size=args.batch_size,
                                       train_subset=5000 if args.quick else None)
        db = NAS_DIR / f"nas_study{suffix}.db"
        if args.fresh and db.exists():
            db.unlink()
        # constraint-aware TPE: Optuna >= 5 reads the values given to trial.set_constraint();
        # older versions take them through constraints_func
        tpe_kwargs = {} if hasattr(optuna.Trial, "set_constraint") else {"constraints_func": constraints_func}
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=12, **tpe_kwargs)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2)
        study = optuna.create_study(study_name="hw_aware_nas", storage=f"sqlite:///{db.as_posix()}",
                                    direction="maximize", sampler=sampler, pruner=pruner, load_if_exists=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        objective = NASObjective(args, constraints, device, train, val)

        def write_history(study, _trial):
            rows = history_rows(study)
            write_csv(rows, NAS_DIR / f"search_history{suffix}.csv")
            save_json(rows, NAS_DIR / f"search_history{suffix}.json")

        t0 = time.time()
        remaining = max(0, args.trials - len(study.trials))
        study.optimize(objective, n_trials=remaining, timeout=args.timeout_min * 60,
                       callbacks=[write_history], gc_after_trial=True)
        minutes = (time.time() - t0) / 60

        rows = history_rows(study)
        write_csv(rows, NAS_DIR / f"search_history{suffix}.csv")
        save_json(rows, NAS_DIR / f"search_history{suffix}.json")
        best, top, near = select_best(rows, args.tie_margin)
        front = pareto_front(rows)
        write_csv(front, NAS_DIR / f"pareto_front{suffix}.csv")

        counts = {s: sum(r["status"] == s for r in rows) for s in ("complete", "pruned", "rejected", "duplicate")}
        result = {
            "selected_trial": best["trial"], "config": json.loads(best["config"]), "summary": best["summary"],
            "proxy_val_acc": best["val_acc"], "best_proxy_val_acc": top,
            "candidates_within_tie_margin": [r["trial"] for r in near],
            "measurements": {k: best[k] for k in ("params", "size_mb", "cpu_latency_ms", "cpu_latency_p90_ms",
                                                  "peak_mem_mb", "activation_mem_mb", "macs_m")},
            "constraints": constraints.to_dict(),
            "selection_rule": f"highest proxy val accuracy among feasible trials; ties within "
                              f"{args.tie_margin} points broken by lower measured CPU latency",
            "search": {"strategy": "Optuna TPE (constraint-aware) + median pruner", "trials": len(rows),
                       **counts, "proxy_epochs": args.proxy_epochs, "minutes": round(minutes, 1),
                       "space_size": search_space_size(), "optuna": optuna.__version__},
        }
        save_json(result, NAS_DIR / f"best_architecture{suffix}.json")

        banner("NAS summary")
        print(f"trials {len(rows)} in {minutes:.1f} min | complete {counts['complete']} | "
              f"pruned {counts['pruned']} | rejected (constraints) {counts['rejected']} | "
              f"duplicates {counts['duplicate']}")
        print(f"best proxy accuracy {top:.2f}% | selected trial {best['trial']} "
              f"(val {best['val_acc']:.2f}%, {best['cpu_latency_ms']:.2f} ms, {best['size_mb']:.3f} MB, "
              f"{best['peak_mem_mb']:.2f} MB peak)")
        print(f"selected architecture: {best['summary']}")
        print(f"Pareto front (feasible, accuracy vs CPU latency): {len(front)} architectures")
        for r in front:
            print(f"   trial {r['trial']:3d}: {r['val_acc']:5.2f}% @ {r['cpu_latency_ms']:.2f} ms, {r['size_mb']:.3f} MB")
        print(f"saved {NAS_DIR / f'best_architecture{suffix}.json'}")

    if not args.no_train_best:
        # automatically train the selected architecture from scratch
        cmd = [sys.executable, str(Path(__file__).with_name("06_nas_train_best.py"))]
        if args.quick:
            cmd.append("--quick")
        print(f"\nstarting final training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
