"""Q2.7 - Train the best NAS architecture from scratch.                  (deliverable 8)

Reads the selected architecture from results/nas/best_architecture.json (written automatically
by 05_nas_search.py), rebuilds it with the generator from fresh random weights - nothing is
reused from the short proxy training - and trains it with a full schedule on CIFAR-10. The
deployment constraints are re-measured on the final network and stored with the checkpoint.

Outputs
  checkpoints/nas/nas_best.pth             best-validation weights + architecture config
  results/nas/final_history.csv            per-epoch log
  results/nas/final_results.json           val/test accuracy + measured latency/size/memory
  results/logs/06_nas_train_best.log

Run:  python scripts/06_nas_train_best.py               (100 epochs)
      python scripts/06_nas_train_best.py --quick
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.common import CKPT_DIR, LOG_DIR, RESULTS_DIR, SEED, Tee, banner, get_device, load_json, save_json, set_seed
from src.data import get_loaders
from src.nas import DeploymentConstraints, build_network, config_summary, profile_architecture
from src.train_utils import count_params, evaluate, fit, load_checkpoint

NAS_DIR = RESULTS_DIR / "nas"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    suffix = "_quick" if args.quick else ""
    with Tee(LOG_DIR / f"06_nas_train_best{suffix}.log"):
        set_seed(SEED)
        device = get_device()
        best = load_json(NAS_DIR / f"best_architecture{suffix}.json")
        cfg = best["config"]
        epochs = 2 if args.quick else args.epochs
        model = build_network(cfg, num_classes=10)

        banner("Training the selected NAS architecture from scratch (CIFAR-10)")
        print(f"architecture (trial {best['selected_trial']}): {config_summary(cfg)}")
        print(f"params {count_params(model):,} | input {cfg['resolution']}x{cfg['resolution']} | "
              f"proxy val acc during search {best['proxy_val_acc']:.2f}%")
        print(f"epochs {epochs} | lr {args.lr} | batch {args.batch_size} | wd {args.wd}")

        train, val, test, classes = get_loaders("cifar10", device, batch_size=args.batch_size,
                                                resolution=cfg["resolution"],
                                                train_subset=5000 if args.quick else None)
        config = {"arch": "nas_best", "nas_config": cfg, "num_classes": 10, "dataset": "cifar10",
                  "resolution": cfg["resolution"], "epochs": epochs, "lr": args.lr,
                  "batch_size": args.batch_size, "weight_decay": args.wd, "seed": SEED,
                  "selected_trial": best["selected_trial"]}
        ckpt = CKPT_DIR / "nas" / f"nas_best{suffix}.pth"
        t0 = time.time()
        fit(model, train, val, epochs=epochs, lr=args.lr, weight_decay=args.wd, device=device,
            ckpt_path=ckpt, config=config, history_csv=NAS_DIR / f"final_history{suffix}.csv",
            resume=args.resume)
        minutes = (time.time() - t0) / 60

        state = load_checkpoint(ckpt)
        model.load_state_dict(state["state_dict"])
        val_m, test_m = evaluate(model, val, device), evaluate(model, test, device)

        # re-measure the deployment cost of the final network (same procedure as in the search)
        hw = profile_architecture(cfg, latency_runs=300, warmup=50)
        constraints = DeploymentConstraints(**best["constraints"])
        feasible, reasons = constraints.check(hw)
        state.update(test_acc=test_m["top1"], hardware=hw)
        torch.save(state, ckpt)

        res = {"selected_trial": best["selected_trial"], "summary": config_summary(cfg), "config": cfg,
               "best_epoch": state["epoch"], "val_top1": val_m["top1"], "test_top1": test_m["top1"],
               "test_top5": test_m["top5"], "proxy_val_acc": best["proxy_val_acc"],
               "train_minutes": round(minutes, 2), "hardware": hw, "constraints": best["constraints"],
               "satisfies_constraints": feasible, "violations": reasons, "checkpoint": f"nas/{ckpt.name}"}
        save_json(res, NAS_DIR / f"final_results{suffix}.json")
        banner("Final NAS model")
        print(f"best epoch {state['epoch']} | val {val_m['top1']:.2f}% | test top-1 {test_m['top1']:.2f}% | "
              f"{minutes:.1f} min")
        print(f"measured: CPU {hw['cpu_latency_ms']:.3f} ms | size {hw['size_mb']:.3f} MB | "
              f"peak mem {hw['peak_mem_mb']:.2f} MB | params {hw['params']:,} | MACs {hw['macs_m']:.1f} M")
        print(f"constraints satisfied: {feasible} {reasons if reasons else ''}")
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
