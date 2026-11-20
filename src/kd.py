"""Knowledge distillation framework (Question 1, requirement 3).

The framework combines three signals that the student learns from:

1. Hard labels  - ordinary cross-entropy with the ground-truth class.

2. Soft targets - the teacher's class probabilities softened with a temperature T
   (Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network", 2015):

        p_i^T = exp(z_i / T) / sum_j exp(z_j / T)

   A higher T flattens the distribution and exposes how the teacher ranks the wrong classes
   ("a cat looks more like a dog than like a truck"). The student matches these with the
   Kullback-Leibler divergence KL(p_teacher^T || p_student^T). Gradients of this term scale as
   1/T^2, so the loss is multiplied by T^2 to keep its magnitude independent of T.

3. Intermediate features - FitNets-style hints (Romero et al., "FitNets: Hints for Thin Deep
   Nets", ICLR 2015). A student feature map is projected by a small learnable regressor
   (1x1 conv + BN) to the teacher's channel count and pulled towards the teacher's feature map
   with a mean-squared error. This transfers *how* the teacher represents an image, not only
   its final answer. If the spatial sizes differ, the larger map is average-pooled down.

Combined objective (alpha balances hard vs soft labels, beta weights the feature term):

    L = (1 - alpha) * CE(y, z_s)  +  alpha * T^2 * KL(p_t^T || p_s^T)  +  beta * MSE(r(f_s), f_t)

Setting alpha = 0 and beta = 0 recovers plain supervised training, alpha > 0 and beta = 0 is
classic soft-target KD, and beta > 0 adds feature distillation on top.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def temperature_softmax(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Softmax with temperature: softmax(z / T). T = 1 is the ordinary softmax."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """Hinton soft-target loss: T^2 * KL(softmax(t/T) || softmax(s/T)), averaged over the batch.

    Computed in float32 for numerical safety (the logits may come from a bf16 autocast region).
    """
    log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
    p_t = temperature_softmax(teacher_logits.float(), T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class HintRegressor(nn.Module):
    """Maps a student feature map onto the teacher's feature space (FitNets regressor).

    1x1 convolution (student channels -> teacher channels) followed by BatchNorm. It is only used
    during training and is thrown away afterwards, so it adds nothing to the deployed student.
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
                                  nn.BatchNorm2d(teacher_channels))

    def forward(self, f_s: torch.Tensor) -> torch.Tensor:
        return self.proj(f_s)


def feature_loss(f_s_proj: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between projected student features and teacher features.

    If the two maps have different spatial sizes, the larger one is average-pooled to the
    smaller size so the comparison is still position by position.
    """
    hs, ht = f_s_proj.shape[-1], f_t.shape[-1]
    if hs > ht:
        f_s_proj = F.adaptive_avg_pool2d(f_s_proj, ht)
    elif ht > hs:
        f_t = F.adaptive_avg_pool2d(f_t, hs)
    return F.mse_loss(f_s_proj.float(), f_t.float())


@dataclass
class KDConfig:
    """Hyper-parameters of the distillation objective (stored inside every KD checkpoint)."""
    alpha: float = 0.9        # weight of the soft-target term (1 - alpha goes to hard labels)
    temperature: float = 4.0  # softmax temperature T
    beta: float = 0.0         # weight of the feature (hint) term; 0 disables feature KD

    def to_dict(self):
        return asdict(self)


class DistillationLoss(nn.Module):
    """Combined hard-label + soft-label + feature distillation loss.

    Call it with the student/teacher logits, the labels, and (optionally) the student and
    teacher hint feature maps. It returns the total loss and a dict with every component so the
    training log can show what the student is actually optimising.
    """

    def __init__(self, cfg: KDConfig, student_channels: int = None, teacher_channels: int = None):
        super().__init__()
        self.cfg = cfg
        self.regressor = None
        if cfg.beta > 0:
            assert student_channels and teacher_channels, "feature KD needs both channel counts"
            self.regressor = HintRegressor(student_channels, teacher_channels)

    def forward(self, s_logits, labels, t_logits=None, f_s=None, f_t=None):
        cfg = self.cfg
        hard = F.cross_entropy(s_logits.float(), labels)
        parts = {"hard_ce": hard.detach()}
        total = hard
        if t_logits is not None and cfg.alpha > 0:
            soft = soft_target_loss(s_logits, t_logits, cfg.temperature)
            total = (1.0 - cfg.alpha) * hard + cfg.alpha * soft
            parts["soft_kl"] = soft.detach()
        if self.regressor is not None and f_s is not None and f_t is not None:
            feat = feature_loss(self.regressor(f_s), f_t)
            total = total + cfg.beta * feat
            parts["feat_mse"] = feat.detach()
        parts["total"] = total.detach()
        return total, parts
