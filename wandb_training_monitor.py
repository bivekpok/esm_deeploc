"""
Scalar (and optional histogram) W&B logging for trainable weights, biases, and grads.

Logs a small, fixed set of tensors so runs stay readable:
  - classifier MLP + pooling head
  - LoRA adapters (per-module norms + aggregates), or last ESMC block when full FT
  - ESMC token embedding (full FT only)
  - optimizer param-group learning rates + global grad norms

By default (``config.wandb_log_monitor_as_charts=False``) monitor scalars are
recorded in a W&B **Table** and the run **summary** — not as hundreds of line
charts. Training curves (loss / F1) still use ``wandb.log`` as charts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


_BLOCK_RE = re.compile(r"\.blocks\.(\d+)\.")


def _last_transformer_block_index(model: nn.Module) -> Optional[int]:
    idxs: List[int] = []
    for name, _ in model.named_parameters():
        m = _BLOCK_RE.search(name)
        if m is not None:
            idxs.append(int(m.group(1)))
    return max(idxs) if idxs else None


def _wandb_key(param_name: str) -> str:
    """Short, stable chart name (no fold prefix)."""
    key = param_name
    if key.startswith("classifier."):
        key = key[len("classifier.") :]
    key = key.replace("default.", "")
    return key.replace(".", "/")


def _is_backbone_slice(name: str, last_block: Optional[int]) -> bool:
    if last_block is None:
        return False
    if f".blocks.{last_block}." not in name:
        return False
    # High-signal linear layers in the last block only.
    return name.endswith(".weight") and any(
        s in name for s in ("attn.out_proj", "attn.layernorm_qkv", "ffn.1", "ffn.3")
    )


def select_monitored_parameters(
    model: nn.Module,
    *,
    use_lora: bool,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """Return ``(param_name, param)`` pairs to log this epoch."""
    last_block = _last_transformer_block_index(model)
    selected: List[Tuple[str, nn.Parameter]] = []
    lora_pairs: List[Tuple[str, nn.Parameter]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            selected.append((name, param))
            continue
        if "lora_" in name:
            lora_pairs.append((name, param))
            continue
        if use_lora:
            continue
        if name.startswith("esmc.embed") and name.endswith(".weight"):
            selected.append((name, param))
            continue
        if _is_backbone_slice(name, last_block):
            selected.append((name, param))

    if lora_pairs:
        # Per-adapter scalars for the last block; aggregates use all LoRA tensors.
        last_lora = [
            (n, p)
            for n, p in lora_pairs
            if last_block is not None and f".blocks.{last_block}." in n
        ]
        seen = {n for n, _ in selected}
        for n, p in last_lora[:32]:
            if n not in seen:
                selected.append((n, p))
                seen.add(n)
    return selected, lora_pairs


@torch.no_grad()
def _param_scalars(name: str, param: torch.Tensor) -> Dict[str, float]:
    w = param.detach().float()
    out: Dict[str, float] = {
        f"monitor/weights/norm/{_wandb_key(name)}": w.norm().item(),
        f"monitor/weights/abs_mean/{_wandb_key(name)}": w.abs().mean().item(),
        f"monitor/weights/std/{_wandb_key(name)}": w.std().item(),
    }
    if name.endswith(".bias"):
        out[f"monitor/bias/mean/{_wandb_key(name)}"] = w.mean().item()
    return out


@torch.no_grad()
def _grad_scalars(name: str, param: nn.Parameter) -> Dict[str, float]:
    if param.grad is None:
        return {}
    g = param.grad.detach().float()
    return {
        f"monitor/grad/norm/{_wandb_key(name)}": g.norm().item(),
        f"monitor/grad/abs_mean/{_wandb_key(name)}": g.abs().mean().item(),
    }


@torch.no_grad()
def collect_training_monitor_metrics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    use_lora: bool,
    total_grad_norm: Optional[float] = None,
    log_histograms: bool = False,
) -> Dict[str, object]:
    """Build monitor scalars (and optional histograms when enabled)."""
    metrics: Dict[str, object] = {}
    monitored, all_lora = select_monitored_parameters(model, use_lora=use_lora)

    lora_w_norms: List[float] = []
    lora_g_norms: List[float] = []
    head_w_norms: List[float] = []
    backbone_w_norms: List[float] = []

    for name, param in all_lora:
        lora_w_norms.append(param.detach().float().norm().item())
        if param.grad is not None:
            lora_g_norms.append(param.grad.detach().float().norm().item())

    for name, param in monitored:
        metrics.update(_param_scalars(name, param))
        metrics.update(_grad_scalars(name, param))

        wn = param.detach().float().norm().item()
        if name.startswith("classifier."):
            head_w_norms.append(wn)
        elif "lora_" not in name:
            backbone_w_norms.append(wn)

        if log_histograms and wandb is not None:
            metrics[f"monitor/hist/weights/{_wandb_key(name)}"] = wandb.Histogram(
                param.detach().float().cpu().numpy()
            )
            if param.grad is not None:
                metrics[f"monitor/hist/grad/{_wandb_key(name)}"] = wandb.Histogram(
                    param.grad.detach().float().cpu().numpy()
                )

    if lora_w_norms:
        metrics["monitor/lora/weight_norm_mean"] = float(sum(lora_w_norms) / len(lora_w_norms))
        metrics["monitor/lora/weight_norm_max"] = float(max(lora_w_norms))
    if lora_g_norms:
        metrics["monitor/lora/grad_norm_mean"] = float(sum(lora_g_norms) / len(lora_g_norms))
    if head_w_norms:
        metrics["monitor/head/weight_norm_mean"] = float(sum(head_w_norms) / len(head_w_norms))
    if backbone_w_norms:
        metrics["monitor/backbone/weight_norm_mean"] = float(
            sum(backbone_w_norms) / len(backbone_w_norms)
        )

    if total_grad_norm is not None:
        metrics["monitor/grad/norm_total"] = float(total_grad_norm)

    for i, group in enumerate(optimizer.param_groups):
        lr = group.get("lr")
        if lr is None:
            continue
        tag = "esmc" if i == 0 else "classifier"
        metrics[f"monitor/lr/{tag}"] = float(lr)

    esmc_g, clf_g = [], []
    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue
        gn = param.grad.detach().float().norm().item()
        if name.startswith("classifier."):
            clf_g.append(gn)
        elif "lora_" in name or name.startswith("esmc."):
            esmc_g.append(gn)
    if esmc_g:
        metrics["monitor/grad/norm_esmc_group"] = float(sum(esmc_g))
    if clf_g:
        metrics["monitor/grad/norm_classifier_group"] = float(sum(clf_g))

    return metrics


def prefix_metrics(metrics: Dict[str, object], fold_name: str) -> Dict[str, object]:
    return {f"{fold_name}/{k}": v for k, v in metrics.items()}


def should_log_monitor(epoch: int, every: int, histogram_every: int, log_histograms: bool) -> Tuple[bool, bool]:
    """Return ``(log_scalars, log_histograms_this_epoch)``."""
    every = max(1, every)
    log_scalars = (epoch + 1) % every == 0
    hist = log_histograms and histogram_every > 0 and (epoch + 1) % histogram_every == 0
    return log_scalars, hist


def setup_wandb_chart_metrics(fold_name: str) -> None:
    """Register step axis for training-curve metrics only (not monitor/*)."""
    if wandb is None or wandb.run is None:
        return
    wandb.define_metric("epoch")
    for key in (
        "train_loss",
        "val_loss",
        "val_macro_f1",
        "val_micro_f1",
        "epochs_no_improve",
        "best_epoch",
    ):
        wandb.define_metric(f"{fold_name}/{key}", step_metric="epoch")


class MonitorTableLogger:
    """Accumulate monitor scalars for a W&B Table (no line-chart spam)."""

    _COLUMNS = ("epoch", "metric", "value")

    def __init__(self) -> None:
        self._rows: List[List[object]] = []

    def append(self, epoch: int, metrics: Dict[str, object]) -> None:
        ep = int(epoch) + 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._rows.append([ep, str(key), float(value)])

    def update_summary(self, fold_name: str, metrics: Dict[str, object]) -> None:
        if wandb is None or wandb.run is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"{fold_name}/{key}"] = float(value)

    @property
    def rows(self) -> List[List[object]]:
        return self._rows

    def to_wandb_table(self):
        if wandb is None or not self._rows:
            return None
        return wandb.Table(columns=list(self._COLUMNS), data=self._rows)


def emit_monitor_metrics(
    monitor: Dict[str, object],
    fold_name: str,
    epoch: int,
    *,
    as_charts: bool,
    table_logger: Optional[MonitorTableLogger] = None,
) -> Dict[str, object]:
    """
    Return a ``wandb.log`` payload for monitor scalars.

    When ``as_charts`` is False, scalars are appended to ``table_logger`` and
    the run summary instead — nothing is returned for chart logging.
    Histogram objects are never returned as chart payloads.
    """
    scalar_monitor = {
        k: v for k, v in monitor.items() if isinstance(v, (int, float))
    }
    if as_charts:
        return prefix_metrics(scalar_monitor, fold_name)
    if table_logger is not None:
        table_logger.append(epoch, scalar_monitor)
        table_logger.update_summary(fold_name, scalar_monitor)
    return {}


def finalize_monitor_table(
    fold_name: str,
    table_logger: Optional[MonitorTableLogger],
    *,
    as_charts: bool,
) -> None:
    """Upload the accumulated monitor table once at end of fold."""
    if as_charts or table_logger is None or wandb is None or wandb.run is None:
        return
    table = table_logger.to_wandb_table()
    if table is not None:
        wandb.log({f"{fold_name}/monitor_scalars": table})
