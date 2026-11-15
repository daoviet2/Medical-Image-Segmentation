import math

import torch
from torch import nn
from torch.nn import functional as F

from . import register_loss


def _binary_inputs(logits, targets):
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    if logits.ndim != 3 or logits.shape != targets.shape:
        raise ValueError(
            f"Expected matching binary masks, got {logits.shape}, {targets.shape}"
        )

    return logits.float(), targets.float()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        if not 0 <= alpha <= 1 or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("Expected 0 <= alpha <= 1 and finite gamma >= 0")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits, targets = _binary_inputs(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Tính 1 - p_t ổn định khi BCE gần 0.
        error_prob = (-torch.expm1(-bce)).clamp_min(torch.finfo(bce.dtype).tiny)
        return (alpha_t * error_prob.pow(self.gamma) * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        if not all(math.isfinite(x) and x > 0 for x in (alpha, beta, smooth)):
            raise ValueError("Expected finite positive alpha, beta and smooth")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        logits, targets = _binary_inputs(logits, targets)
        probs = logits.sigmoid()

        tp = (probs * targets).sum(dim=(1, 2))
        fp = (probs * (1 - targets)).sum(dim=(1, 2))
        fn = ((1 - probs) * targets).sum(dim=(1, 2))

        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1 - score).mean()


@register_loss("focal")
def _build_focal(params):
    return FocalLoss(
        alpha=float(params.get("alpha", 0.25)),
        gamma=float(params.get("gamma", 2.0)),
    )


@register_loss("tversky")
def _build_tversky(params):
    return TverskyLoss(
        alpha=float(params.get("alpha", 0.3)),
        beta=float(params.get("beta", 0.7)),
        smooth=float(params.get("smooth", 1.0)),
    )