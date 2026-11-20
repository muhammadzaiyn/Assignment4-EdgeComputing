"""Training / evaluation engine shared by the teacher, baseline, distillation and NAS runs.

One `fit()` function trains any classifier. When a teacher and a `DistillationLoss` are passed,
the same loop performs knowledge distillation; without them it is plain supervised training.
Sharing the loop guarantees that the baseline and the distilled students differ *only* in the
loss function (same optimiser, schedule, augmentation, epochs, seed), so any accuracy difference
can be attributed to distillation.

Training details
  * SGD with Nesterov momentum 0.9, weight decay on conv/linear weights only (not on BN/bias)
  * linear warm-up followed by cosine decay, updated every iteration
  * bfloat16 autocast on the GPU (no loss scaling needed with bf16) and channels_last memory layout
  * best checkpoint chosen on the validation split; test set evaluated once at the end
  * a "last" checkpoint is written every epoch so an interrupted run can be resumed (--resume)
"""
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import write_csv


# ----------------------------------------------------------------------------- optimiser / schedule
def make_optimizer(params_or_model, lr, weight_decay, momentum=0.9):
    """SGD-Nesterov; BatchNorm parameters and biases are excluded from weight decay."""
    named = params_or_model if isinstance(params_or_model, list) else list(params_or_model.named_parameters())
    decay, no_decay = [], []
    for name, p in named:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def warmup_cosine(optimizer, total_steps, warmup_steps):
    """Learning-rate multiplier: linear warm-up to 1, then cosine decay to 0."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device=None, amp=True):
    """Return dict(loss, top1, top5) of `model` on every batch of `loader`."""
    model.eval()
    device = device or next(model.parameters()).device
    n, loss_sum, c1, c5 = 0, 0.0, 0, 0
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += y.numel()
    return {"loss": loss_sum / n, "top1": 100.0 * c1 / n, "top5": 100.0 * c5 / n}


@torch.no_grad()
def predict_all(model, loader, device=None):
    """Logits for every image of a (non-shuffled) loader, returned on the CPU in float32."""
    model.eval()
    device = device or next(model.parameters()).device
    out = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu())
    return torch.cat(out)


# ----------------------------------------------------------------------------- training loop
def fit(model, train_loader, val_loader, *, epochs, lr, weight_decay, device,
        ckpt_path: Path, history_csv: Path, config: dict,
        teacher=None, kd_loss=None, warmup_epochs=1, label_smoothing=0.0,
        resume=False, log_every=0, trial_callback=None, save_best=True):
    """Train `model`; returns (best_val_top1, history list).

    Parameters
    ----------
    teacher, kd_loss : if given, knowledge distillation is performed. The teacher is frozen and
                       run in eval mode; `kd_loss` is a `DistillationLoss` whose regressor (if any)
                       is trained jointly with the student.
    ckpt_path        : best-validation checkpoint (dict with state_dict, config, metrics)
    history_csv      : per-epoch log written after every epoch
    trial_callback   : optional f(epoch, val_top1) used by the NAS to prune bad trials early
    save_best        : False for NAS proxy runs, which do not need checkpoints on disk
    """
    model.to(device).to(memory_format=torch.channels_last)
    distill = teacher is not None
    if distill:
        teacher.to(device).to(memory_format=torch.channels_last).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        kd_loss.to(device)

    # the hint regressor is trained together with the student, so it joins the optimiser
    named = list(model.named_parameters())
    if distill and kd_loss.regressor is not None:
        named += [("kd." + n, p) for n, p in kd_loss.regressor.named_parameters()]
    optimizer = make_optimizer(named, lr, weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, epochs * steps_per_epoch, warmup_epochs * steps_per_epoch)
    use_amp = device.type == "cuda"
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    start_epoch, best_acc, history = 0, -1.0, []
    last_path = Path(ckpt_path).with_name(Path(ckpt_path).stem + "_last.pth") if ckpt_path else None
    if resume and last_path is not None and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if distill and kd_loss.regressor is not None and state.get("regressor"):
            kd_loss.regressor.load_state_dict(state["regressor"])
        start_epoch, best_acc, history = state["epoch"] + 1, state["best_acc"], state["history"]
        print(f"resumed from {last_path.name} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        if distill and kd_loss.regressor is not None:
            kd_loss.regressor.train()
        t0 = time.time()
        n, correct, loss_sum = 0, 0, 0.0
        parts_sum = {}
        for it, (x, y) in enumerate(train_loader):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                if distill:
                    with torch.no_grad():
                        t_logits, t_feats = teacher(x, return_features=True)
                        f_t = t_feats[teacher.hint_layer]
                    s_logits, f_s = model(x, return_features=True)
                    loss, parts = kd_loss(s_logits, y, t_logits, f_s, f_t)
                else:
                    s_logits = model(x)
                    loss = ce(s_logits.float(), y)
                    parts = {"total": loss.detach()}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = y.numel()
            n += bs
            loss_sum += loss.item() * bs
            correct += (s_logits.argmax(1) == y).sum().item()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v.item() * bs
            if log_every and (it + 1) % log_every == 0:
                print(f"    it {it + 1}/{steps_per_epoch} loss {loss_sum / n:.4f}")

        train_time = time.time() - t0
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / n, "train_acc": 100.0 * correct / n,
               "val_loss": val["loss"], "val_acc": val["top1"], "val_top5": val["top5"],
               "epoch_time_s": train_time}
        if distill:
            row.update({k: v / n for k, v in parts_sum.items() if k != "total"})
        history.append(row)
        improved = val["top1"] > best_acc
        if improved:
            best_acc = val["top1"]
            if save_best and ckpt_path is not None:
                save_checkpoint(ckpt_path, model, config, epoch=epoch + 1, val_acc=best_acc)
        extra = "".join(f" {k} {row[k]:.3f}" for k in ("hard_ce", "soft_kl", "feat_mse") if k in row)
        print(f"epoch {epoch + 1:3d}/{epochs} | loss {row['train_loss']:.4f}{extra} | "
              f"train {row['train_acc']:5.2f}% | val {val['top1']:5.2f}% (top5 {val['top5']:5.2f}%) | "
              f"lr {row['lr']:.4f} | {train_time:5.1f}s{'  *best' if improved else ''}")

        if history_csv is not None:
            write_csv(history, history_csv)
        if save_best and last_path is not None:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "best_acc": best_acc,
                        "history": history,
                        "regressor": kd_loss.regressor.state_dict() if distill and kd_loss.regressor is not None else None},
                       last_path)
        if trial_callback is not None:
            trial_callback(epoch, val["top1"])
    return best_acc, history


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, model, config: dict, **metrics):
    """Save weights + everything needed to rebuild the model (no optimiser state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "config": config, **metrics}, path)


def load_checkpoint(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
