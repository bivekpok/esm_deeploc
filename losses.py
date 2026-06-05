"""
Multilabel loss functions for DeepLoc training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    """
    Numerically stable focal loss for multilabel classification.
    Uses per-class alpha for handling class imbalance.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
    ):
        """
        Args:
            gamma: focusing parameter. 2.0 is the standard starting point.
            alpha: Tensor of shape [num_classes] or scalar.
                   Higher = more weight on positive examples for that class.
                   If None, no class-level reweighting beyond focal mechanism.
            pos_weight: Optional additional per-class positive weight tensor.
        """
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C] raw logits
        targets: [B, C] binary labels
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.pos_weight is not None:
            pos_weight_factor = self.pos_weight * targets + (1 - targets)
            loss = loss * pos_weight_factor

        return loss.mean()


# Per-class positive alpha for multilabel focal (higher = more weight on y=1).
# Used when config.focal_use_alpha=True and config.focal_alpha_mode="manual".
# Edit weights here; enable via config.py loss section.
MANUAL_FOCAL_ALPHA: dict[str, float] = {
    "Golgi apparatus": 0.75,
    "Lysosome/Vacuole": 0.75,
    "Peroxisome": 0.75,
    "Extracellular": 0.50,
    "Cell membrane": 0.50,
    "Mitochondrion": 0.50,
    "Plastid": 0.50,
    "Endoplasmic reticulum": 0.50,
    "Cytoplasm": 0.25,
    "Nucleus": 0.25,
}


def compute_focal_alpha_from_targets(targets: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency alpha per class, scaled to [0, 1] (legacy; can crush common classes)."""
    class_counts = targets.sum(dim=0).clamp(min=1.0)
    inv_freq = 1.0 / (class_counts / class_counts.sum())
    return inv_freq / inv_freq.max()


def compute_manual_focal_alpha(train_ds, device: torch.device) -> torch.Tensor:
    """Build alpha vector from ``MANUAL_FOCAL_ALPHA`` aligned to ``train_ds.classes``."""
    alpha = torch.full((len(train_ds.classes),), 0.5, dtype=torch.float32)
    for cls_name, val in MANUAL_FOCAL_ALPHA.items():
        idx = train_ds.class_to_idx.get(cls_name)
        if idx is not None:
            alpha[idx] = float(val)
    return alpha.to(device)


def resolve_focal_alpha(
    train_ds,
    device: torch.device,
    *,
    mode: str = "manual",
) -> torch.Tensor:
    """Return per-class alpha tensor for logging / criterion construction."""
    key = (mode or "manual").lower()
    if key == "manual":
        return compute_manual_focal_alpha(train_ds, device)
    if key == "inv_freq":
        return compute_focal_alpha_from_targets(train_ds.targets).to(device)
    raise ValueError(f"Unknown focal_alpha_mode={mode!r}; use 'manual' or 'inv_freq'.")


def build_multilabel_criterion(
    train_ds,
    device: torch.device,
    *,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_use_alpha: bool = True,
    focal_alpha_mode: str = "manual",
    focal_use_pos_weight: bool = False,
) -> nn.Module:
    """Factory: BCE (sqrt pos_weight) or multilabel focal loss."""
    loss_type = (loss_type or "bce").lower()
    if loss_type == "focal":
        alpha = None
        if focal_use_alpha:
            alpha = resolve_focal_alpha(
                train_ds, device, mode=focal_alpha_mode
            )
        pos_weight = None
        if focal_use_pos_weight:
            pos_weight = train_ds.pos_weight.to(device)
        return MultilabelFocalLoss(
            gamma=focal_gamma,
            alpha=alpha,
            pos_weight=pos_weight,
        ).to(device)

    if loss_type != "bce":
        raise ValueError(f"Unknown loss_type={loss_type!r}; use 'bce' or 'focal'.")

    pos_w = torch.sqrt(train_ds.pos_weight.to(device))
    return nn.BCEWithLogitsLoss(pos_weight=pos_w)
