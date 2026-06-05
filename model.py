"""
Model definitions for the DeepLoc 2.0 multilabel pipeline.

Backbones (chosen by ``config.use_lora_model``):
- Full ESMC (default): every parameter is trainable; AdamW differentiates the
  backbone LR (``config.lr_esmc``) from the classifier LR (``config.lr_classifier``).
- LoRA ESMC (``config.use_lora_model = True``): PEFT LoRA adapters on the
  backbone. If ``config.lora_last_n_layers`` is a positive int, only the **last
  N** ``transformer.blocks.*`` layers get adapters (explicit module paths); if
  ``None``, short ``target_modules`` names match **every** block. Only adapters
  + classifier head train by default.

Pooling heads (chosen by ``config.pooling_type``):
- ``"attention"``: ``LocalizationAttention`` -> N-term + C-term + global mean
  over context (3 * embed_dim feature vector). Returns a non-trivial
  entropy regularization signal.
- ``"average"``:   masked mean over the token sequence
  (embed_dim feature vector). Entropy loss is a zero tensor.
- ``"bom"``:       Bag-of-(k)Mer pooling - within each k-mer window we mean
  the per-token embeddings, then summarize across windows with mean & max
  (and a small MLP projection). Returns a single ``bom_output_dim`` vector
  (defaults to embed_dim). Entropy loss is a zero tensor.
- ``"bom_attn"``:  Same k-mer windows as ``bom``, then self-attention among
  windows (Q/K/V, softmax over windows), mean-pool over valid windows. See
  ``BoMAttentionPooling``. Entropy loss is a zero tensor.

DDP notes:
- ``ESMCClassifier.forward`` always returns ``(logits, entropy_loss)``; for
  the non-attention pooling heads we still return a zero scalar so callers do
  not need a special-case path.
- ``synchronize_esmc_classifier_batched_path`` probes the tokenizer + single-
  forward path on every rank and folds the result via ``all_reduce(MIN)``
  before DDP wraps the model, so every rank executes the same autograd graph.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from config import config


# ---------------------------------------------------------------------------
# Pooling heads
# ---------------------------------------------------------------------------

class LocalizationAttention(nn.Module):
    """N-term / C-term / global mean over a self-attention context.

    Returns ``(pooled[B, 3*E], entropy_loss)``. The slight scoring boost on the
    first 20 and last 20 valid positions biases the head toward N/C signal
    peptides without preventing global usage.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        for m in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(m.weight, gain=1 / (num_heads ** 0.25))
            nn.init.constant_(m.bias, 0.0)

    def output_dim(self) -> int:
        return self.embed_dim * 3

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        device = next(self.parameters()).device
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        B, L, _ = x.shape
        x = self.layer_norm(x)

        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.25)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, -1e4)
            seq_lengths = mask.sum(dim=1)
            for i in range(B):
                actual_len = int(seq_lengths[i].item())
                if actual_len >= 20:
                    scores[i, :, :, :20] += 1.0
                    scores[i, :, :, actual_len - 20 : actual_len] += 0.8
                else:
                    scores[i, :, :, :actual_len] += 1.0 + 0.8
        else:
            if L >= 20:
                scores[:, :, :, :20] += 1.0
                scores[:, :, :, -20:] += 0.8

        attn_weights = F.softmax(scores, dim=-1)
        entropy_loss = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1).mean()
        context = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, L, -1)

        if mask is not None:
            seq_lengths = mask.sum(dim=1)
            n_term_list, c_term_list = [], []
            for i in range(B):
                actual_len = int(seq_lengths[i].item())
                n_size = min(20, actual_len)
                n_term_list.append(context[i, :n_size].mean(dim=0))
                c_size = min(20, actual_len)
                start = max(0, actual_len - c_size)
                c_term_list.append(context[i, start:actual_len].mean(dim=0))
            n_term = torch.stack(n_term_list)
            c_term = torch.stack(c_term_list)
        else:
            n_term = context[:, :20].mean(dim=1)
            c_term = context[:, -20:].mean(dim=1)

        global_pool = context.mean(dim=1)
        pooled = torch.cat([n_term, c_term, global_pool], dim=1)
        return pooled, entropy_loss


class AveragePooling(nn.Module):
    """Mask-aware mean pooling. Returns ``(pooled[B, E], zero_loss)``."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim)

    def output_dim(self) -> int:
        return self.embed_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.layer_norm(x)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            m = mask.to(dtype=x.dtype).unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        zero_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        return pooled, zero_loss


class BoMPooling(nn.Module):
    """
    Bag-of-(k)Mer pooling.

    1. Slide a length-``k`` window with stride ``s`` over the (LayerNorm'd) token
       embeddings; for every window output the per-window mean over valid tokens
       (windows that contain only padding are dropped from both summary stats).
    2. Project the window vectors through a small MLP (``inner_dim``).
    3. Summarize across windows with ``mean`` and (optionally) ``max`` and
       concatenate to give the pooled vector.

    Output dimensionality is ``bom_output_dim`` (defaults to ``embed_dim``).
    """

    def __init__(
        self,
        embed_dim: int,
        k_mer_size: int = 5,
        stride: int = 1,
        inner_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        summary: str = "mean_max",
        dropout: float = 0.1,
    ):
        super().__init__()
        if k_mer_size <= 0:
            raise ValueError("k_mer_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if summary not in ("mean", "max", "mean_max"):
            raise ValueError("summary must be one of 'mean'|'max'|'mean_max'.")

        self.embed_dim = embed_dim
        self.k = k_mer_size
        self.stride = stride
        self.summary = summary
        self.inner_dim = inner_dim or (embed_dim // 2)
        self._output_dim = output_dim or embed_dim

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.window_mlp = nn.Sequential(
            nn.Linear(embed_dim, self.inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        summary_mult = 2 if summary == "mean_max" else 1
        self.out_proj = nn.Linear(self.inner_dim * summary_mult, self._output_dim)

    def output_dim(self) -> int:
        return self._output_dim

    def _window_means(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(windows[B, W, E], valid[B, W])`` where each window slot
        is the masked mean of up to ``k`` consecutive tokens."""
        B, L, E = x.shape
        k, s = self.k, self.stride

        if L < k:
            if mask is None:
                pooled = x.mean(dim=1, keepdim=True)
                valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            else:
                m = mask.to(dtype=x.dtype).unsqueeze(-1)
                denom = m.sum(dim=1).clamp(min=1.0)
                pooled = (x * m).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
                valid = (mask.sum(dim=1, keepdim=True) > 0).to(torch.bool)
            return pooled, valid

        # Token-level sum with mask folded in.
        if mask is None:
            x_eff = x
            m_eff = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)
        else:
            m_eff = mask.to(dtype=x.dtype).unsqueeze(-1)
            x_eff = x * m_eff

        # ``unfold`` slides a window along dim=1.
        x_un = x_eff.unfold(dimension=1, size=k, step=s)
        m_un = m_eff.unfold(dimension=1, size=k, step=s)
        win_sum = x_un.sum(dim=-1)
        win_count = m_un.sum(dim=-1).clamp(min=1.0)
        win_mean = win_sum / win_count

        valid = (m_un.sum(dim=-1).squeeze(-1) > 0)
        return win_mean, valid

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.layer_norm(x)
        windows, valid = self._window_means(x, mask)
        windows = self.window_mlp(windows)

        valid_f = valid.to(dtype=windows.dtype).unsqueeze(-1)
        masked_windows = windows * valid_f

        if self.summary == "mean":
            denom = valid_f.sum(dim=1).clamp(min=1.0)
            pooled = masked_windows.sum(dim=1) / denom
        elif self.summary == "max":
            neg_inf = torch.finfo(windows.dtype).min
            masked_for_max = masked_windows.masked_fill(~valid.unsqueeze(-1), neg_inf)
            pooled, _ = masked_for_max.max(dim=1)
            pooled = torch.where(
                valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled)
            )
        else:
            denom = valid_f.sum(dim=1).clamp(min=1.0)
            mean_pool = masked_windows.sum(dim=1) / denom
            neg_inf = torch.finfo(windows.dtype).min
            masked_for_max = masked_windows.masked_fill(~valid.unsqueeze(-1), neg_inf)
            max_pool, _ = masked_for_max.max(dim=1)
            max_pool = torch.where(
                valid.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool)
            )
            pooled = torch.cat([mean_pool, max_pool], dim=-1)

        pooled = self.out_proj(pooled)
        zero_loss = torch.zeros((), device=pooled.device, dtype=pooled.dtype)
        return pooled, zero_loss


class BoMAttentionPooling(nn.Module):
    """
    Bag-of-(k)Mer embeddings + **self-attention over windows** (per batch item).

    Matches the spirit of the user's ``BoMPooling`` snippet: for each sequence,
    build length-(k) sliding windows (masked mean of token embeddings), project
    to Q/K (``inner_dim``) and V (``value_dim``), then::

        Attn = softmax(Q K^T / sqrt(d)) V

    Pool by **mean over valid windows only**. Output is ``out_proj`` →
    ``output_dim`` (defaults to ``value_dim``).
    """

    def __init__(
        self,
        embed_dim: int,
        k_mer_size: int = 7,
        stride: int = 1,
        inner_dim: int = 256,
        value_dim: int = 1024,
        output_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if k_mer_size <= 0 or stride <= 0:
            raise ValueError("k_mer_size and stride must be positive.")
        self.embed_dim = embed_dim
        self.k = k_mer_size
        self.stride = stride
        self.inner_dim = inner_dim
        self.value_dim = value_dim
        self._output_dim = int(output_dim or value_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, inner_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, inner_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None
        if self._output_dim != value_dim:
            self.out_proj = nn.Linear(value_dim, self._output_dim)
        else:
            self.out_proj = nn.Identity()

    def output_dim(self) -> int:
        return self._output_dim

    def _window_means(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same contract as ``BoMPooling._window_means``."""
        B, L, E = x.shape
        k, s = self.k, self.stride

        if L < k:
            if mask is None:
                pooled = x.mean(dim=1, keepdim=True)
                valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            else:
                m = mask.to(dtype=x.dtype).unsqueeze(-1)
                denom = m.sum(dim=1).clamp(min=1.0)
                pooled = (x * m).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
                valid = (mask.sum(dim=1, keepdim=True) > 0).to(torch.bool)
            return pooled, valid

        if mask is None:
            x_eff = x
            m_eff = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)
        else:
            m_eff = mask.to(dtype=x.dtype).unsqueeze(-1)
            x_eff = x * m_eff

        x_un = x_eff.unfold(dimension=1, size=k, step=s)
        m_un = m_eff.unfold(dimension=1, size=k, step=s)
        win_sum = x_un.sum(dim=-1)
        win_count = m_un.sum(dim=-1).clamp(min=1.0)
        win_mean = win_sum / win_count
        valid = (m_un.sum(dim=-1).squeeze(-1) > 0)
        return win_mean, valid

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.layer_norm(x)
        windows, valid = self._window_means(x, mask)
        B, W, _E = windows.shape

        q = self.q_proj(windows)
        k = self.k_proj(windows)
        v = self.v_proj(windows)

        scale = self.inner_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        neg_inf = torch.finfo(scores.dtype).min
        key_ok = valid.unsqueeze(1).expand(B, W, W)
        scores = scores.masked_fill(~key_ok, neg_inf)
        attn = F.softmax(scores, dim=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v)

        valid_f = valid.to(dtype=ctx.dtype).unsqueeze(-1)
        denom = valid_f.sum(dim=1).clamp(min=1.0)
        pooled = (ctx * valid_f).sum(dim=1) / denom
        pooled = self.out_proj(pooled)

        zero_loss = torch.zeros((), device=pooled.device, dtype=pooled.dtype)
        return pooled, zero_loss


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

def _build_pooling(pooling_type: str, embed_dim: int) -> nn.Module:
    t = (pooling_type or "attention").strip().lower()
    if t == "attention":
        return LocalizationAttention(embed_dim=embed_dim)
    if t == "average":
        return AveragePooling(embed_dim=embed_dim)
    if t == "bom":
        return BoMPooling(
            embed_dim=embed_dim,
            k_mer_size=int(getattr(config, "k_mer_size", 5)),
            stride=int(getattr(config, "bom_stride", 1)),
            inner_dim=getattr(config, "bom_inner_dim", None),
            output_dim=getattr(config, "bom_output_dim", None),
            summary=str(getattr(config, "bom_summary", "mean_max")),
        )
    if t == "bom_attn":
        k_win = int(
            getattr(config, "bom_attn_k_mer_size", getattr(config, "k_mer_size", 7))
        )
        _bs = getattr(config, "bom_attn_stride", None)
        stride = int(
            config.bom_stride if _bs is None else _bs
        )
        return BoMAttentionPooling(
            embed_dim=embed_dim,
            k_mer_size=k_win,
            stride=stride,
            inner_dim=int(getattr(config, "bom_attn_inner_dim", 256)),
            value_dim=int(getattr(config, "bom_attn_value_dim", 1024)),
            output_dim=getattr(config, "bom_attn_output_dim", None),
            attn_dropout=float(getattr(config, "bom_attn_dropout", 0.0)),
        )
    raise ValueError(
        f"Unknown pooling_type='{pooling_type}'. "
        f"Expected 'attention'|'average'|'bom'|'bom_attn'."
    )


class ProteinClassifier(nn.Module):
    """Pooling head + MLP classifier. ``pooling_type`` selects the head."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 960,
        pooling_type: str = "attention",
        dropout: float = 0.4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.pooling_type = pooling_type
        self.pooling = _build_pooling(pooling_type, embed_dim)
        in_dim = self.pooling.output_dim()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, entropy_loss = self.pooling(x, mask)
        logits = self.classifier(pooled)
        return logits, entropy_loss


# ---------------------------------------------------------------------------
# ESMC wrapper: batched forward (preferred) with per-sequence SDK fallback
# ---------------------------------------------------------------------------

def _resolve_esmc_inner(esmc: nn.Module) -> nn.Module:
    inner = esmc
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model
    return inner


def _resolve_tokenizer(esmc: nn.Module):
    cur = esmc
    while cur is not None:
        tok = getattr(cur, "tokenizer", None)
        if tok is not None:
            return tok
        for attr in ("base_model", "model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                cur = nxt
                break
        else:
            break
    return None


def _select_block_layers(hs: torch.Tensor) -> torch.Tensor:
    """Pick the slice of ESMC block hidden states to average over.

    ``hs`` is a stack [n_layers+1, ...] where index 0 is the pre-block
    embedding and indices 1..n_layers correspond to block outputs 0..n_layers-1.
    The returned tensor preserves the leading ``layers`` dim for later mean.

    Aggregation modes (see ``config.layer_aggregation``):
      * ``"all"``: every block output (current default; n_layers entries).
      * ``"last_n"``: outputs of the last ``layer_agg_n`` blocks.
      * ``"band"``: outputs of blocks ``[a, b]`` inclusive (0-indexed).
    """
    n_blocks = hs.shape[0] - 1  # exclude pre-block embedding
    mode = (getattr(config, "layer_aggregation", "all") or "all").lower()
    if mode == "last_n":
        n = max(1, min(int(getattr(config, "layer_agg_n", 5)), n_blocks))
        # block outputs are at indices 1..n_blocks; last n -> [n_blocks-n+1 .. n_blocks]
        return hs[n_blocks - n + 1 : n_blocks + 1]
    if mode == "band":
        a, b = getattr(config, "layer_agg_band", (0, n_blocks - 1))
        a = max(0, int(a))
        b = min(int(b), n_blocks - 1)
        if a > b:
            a, b = b, a
        return hs[a + 1 : b + 2]
    return hs[1:]


def _batched_layer_embeddings(
    esmc: nn.Module, sequences: Sequence[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize + a single inner forward call. Returns embeddings + (B, L) bool mask."""
    tok = _resolve_tokenizer(esmc)
    if tok is None:
        raise RuntimeError("No tokenizer on ESMC")
    tokenized = [torch.tensor(tok.encode(s), dtype=torch.long) for s in sequences]
    tokens_t = torch.nn.utils.rnn.pad_sequence(
        tokenized, batch_first=True, padding_value=1
    ).to(device)
    inner = _resolve_esmc_inner(esmc)
    out = inner.forward(sequence_tokens=tokens_t, sequence_id=None)
    hs = out.hidden_states
    if hs.dim() == 4 and hs.size(1) == 1:
        hs = hs.squeeze(1)
    layer_stack = _select_block_layers(hs)
    embeddings = layer_stack.float().mean(dim=0)
    mask = tokens_t != 1
    return embeddings, mask


def _sdk_per_sequence_embeddings(
    esmc: nn.Module, sequences: Sequence[str], lengths: Sequence[int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slow SDK path: ``esmc.encode`` + ``esmc.logits`` per sequence, then pad."""
    embeddings = []
    inner = _resolve_esmc_inner(esmc)
    encode_fn = getattr(esmc, "encode", None) or getattr(inner, "encode", None)
    logits_fn = getattr(esmc, "logits", None) or getattr(inner, "logits", None)
    if encode_fn is None or logits_fn is None:
        raise RuntimeError(
            "SDK fallback unavailable: ESMC backbone has no .encode/.logits."
        )
    for seq in sequences:
        protein = ESMProtein(sequence=seq)
        protein_tensor = encode_fn(protein).to(device)
        outputs = logits_fn(
            protein_tensor,
            LogitsConfig(return_embeddings=False, return_hidden_states=True),
        )
        layer_stack = _select_block_layers(outputs.hidden_states).squeeze(1)
        embed = torch.mean(layer_stack.float(), dim=0)
        embeddings.append(embed)
    max_len = max(lengths)
    B = len(embeddings)
    emb = torch.zeros(B, max_len, 960, device=device, dtype=embeddings[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for i, (e, sl) in enumerate(zip(embeddings, lengths)):
        emb[i, :sl] = e[:sl]
        mask[i, :sl] = True
    return emb, mask


def probe_batched_esm_embeddings(esmc: nn.Module, device: torch.device) -> bool:
    try:
        with torch.no_grad():
            _batched_layer_embeddings(esmc, ["ACDEFGHIKLMNP"], device)
        return True
    except Exception:
        return False


def synchronize_esmc_classifier_batched_path(
    model: "ESMCClassifier", device: torch.device, world_size: int
) -> None:
    """DDP needs identical autograd graphs per rank: probe + ``all_reduce(MIN)``
    so any rank's failure disables batched mode for all ranks before DDP."""
    ok_local = 1 if probe_batched_esm_embeddings(model.esmc, device) else 0
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        t = torch.tensor(ok_local, device=device, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        ok_local = int(t.item())
    model.use_batched = bool(ok_local)


class ESMCClassifier(nn.Module):
    """ESMC backbone (full or LoRA) + ``ProteinClassifier`` head."""

    def __init__(self, esmc_model: nn.Module, classifier: ProteinClassifier):
        super().__init__()
        self.esmc = esmc_model
        self.classifier = classifier
        self.use_batched = True

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        sequences = batch["sequences"]
        lengths = batch["lengths"].tolist()

        if self.use_batched:
            emb, mask = _batched_layer_embeddings(self.esmc, sequences, device)
        else:
            emb, mask = _sdk_per_sequence_embeddings(self.esmc, sequences, lengths, device)

        logits, entropy_loss = self.classifier(emb, mask)
        return logits, entropy_loss


# ---------------------------------------------------------------------------
# LoRA wrapping
# ---------------------------------------------------------------------------

# Linear submodules inside each ESMC ``UnifiedTransformerBlock`` (matches
# ``MultiHeadAttention`` + ``ffn`` Sequential layout in ``esm.layers``).
_LORA_BLOCK_LINEAR_SUFFIXES: Tuple[str, ...] = (
    "attn.layernorm_qkv.1",
    "attn.out_proj",
    "ffn.1",
    "ffn.3",
)


def _find_transformer_blocks_prefix(backbone: nn.Module) -> Tuple[str, int]:
    """
    Locate the largest ``nn.ModuleList`` whose registered name ends with
    ``blocks`` (e.g. ``transformer.blocks`` on ``ESMC``). Returns
    ``(dotted_prefix, num_blocks)``.
    """
    best_n = 0
    best_prefix: Optional[str] = None
    for name, module in backbone.named_modules():
        if name.endswith("blocks") and isinstance(module, nn.ModuleList):
            n = len(module)
            if n > best_n:
                best_n = n
                best_prefix = name
    if best_prefix is None or best_n == 0:
        raise RuntimeError(
            "Could not find an nn.ModuleList named '*.blocks' on the backbone; "
            "cannot build explicit last-N-layer LoRA targets."
        )
    return best_prefix, best_n


def _lora_explicit_targets_last_n_layers(backbone: nn.Module) -> List[str]:
    """
    Build full dotted module paths for LoRA on the last ``lora_last_n_layers``
    blocks only (same idea as ``transformer.blocks.20..29`` for N=10).
    """
    last_n = getattr(config, "lora_last_n_layers", None)
    if last_n is None:
        return []
    last_n = int(last_n)
    if last_n <= 0:
        return []

    prefix, n_blocks = _find_transformer_blocks_prefix(backbone)
    override = getattr(config, "lora_total_blocks", None)
    if override is not None:
        n_blocks = int(override)

    if last_n > n_blocks:
        print(
            f"[LoRA] lora_last_n_layers={last_n} > n_blocks={n_blocks}; clamping to n_blocks.",
            flush=True,
        )
        last_n = n_blocks
    start = n_blocks - last_n
    targets: List[str] = []
    for i in range(start, n_blocks):
        for suffix in _LORA_BLOCK_LINEAR_SUFFIXES:
            targets.append(f"{prefix}.{i}.{suffix}")
    print(
        f"[LoRA] explicit targets: {len(targets)} modules "
        f"({last_n} blocks × {len(_LORA_BLOCK_LINEAR_SUFFIXES)} linears, "
        f"block indices {start}–{n_blocks - 1} under `{prefix}`)",
        flush=True,
    )
    return targets


def _wrap_with_lora(backbone: nn.Module) -> nn.Module:
    """Attach PEFT LoRA adapters to the ESMC backbone using config knobs.

    When ``config.lora_last_n_layers`` is a positive int, only those last
    transformer blocks receive adapters (explicit paths). Otherwise use short
    ``lora_target_modules`` names so every matching layer is adapted.

    PEFT is imported lazily so the file still imports without it when
    ``use_lora_model=False``.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise RuntimeError(
            "config.use_lora_model=True requires the `peft` package. "
            "Install with: pip install peft"
        ) from e

    explicit = _lora_explicit_targets_last_n_layers(backbone)
    if explicit:
        target_modules = explicit
    else:
        target_modules = list(
            getattr(
                config,
                "lora_target_modules",
                ("attn.layernorm_qkv.1", "attn.out_proj", "ffn.1", "ffn.3"),
            )
        )
        print(
            f"[LoRA] using global short target_modules ({len(target_modules)} patterns) "
            "on every matching submodule.",
            flush=True,
        )

    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=int(getattr(config, "lora_r", 8)),
        lora_alpha=int(getattr(config, "lora_alpha", 16)),
        lora_dropout=float(getattr(config, "lora_dropout", 0.05)),
        target_modules=target_modules,
    )
    return get_peft_model(backbone, peft_cfg)


# ---------------------------------------------------------------------------
# Build entry point
# ---------------------------------------------------------------------------

def _infer_backbone_embed_dim(backbone: nn.Module) -> int:
    """Read ESMC ``d_model`` from a loaded backbone (960 for 300M, 1152 for 600M)."""
    inner = _resolve_esmc_inner(backbone)
    stack = getattr(inner, "transformer", None)
    if stack is not None:
        d_model = getattr(stack, "d_model", None)
        if d_model is not None:
            return int(d_model)
    embed = getattr(inner, "embed", None)
    if embed is not None and hasattr(embed, "embedding_dim"):
        return int(embed.embedding_dim)
    return int(getattr(config, "embed_dim", 960))


def build_model(
    num_classes: int,
    model_name_or_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    pooling_type: Optional[str] = None,
    use_lora: Optional[bool] = None,
) -> ESMCClassifier:
    """
    Factory: load ESMC, optionally wrap in PEFT LoRA, build pooling head, and
    assemble the ``ESMCClassifier``. Defaults are taken from ``config``.
    """
    model_name_or_path = model_name_or_path or config.model_name_or_path
    pooling_type = pooling_type or getattr(config, "pooling_type", "attention")
    use_lora = config.use_lora_model if use_lora is None else use_lora

    backbone = ESMC.from_pretrained(model_name_or_path)

    # Pooling head width must match backbone hidden size (auto-detect; see config.embed_dim).
    embed_dim = _infer_backbone_embed_dim(backbone)
    config_embed_dim = int(getattr(config, "embed_dim", 960))
    if config_embed_dim != embed_dim:
        import warnings

        warnings.warn(
            f"config.embed_dim={config_embed_dim} != backbone d_model={embed_dim}; "
            "using backbone value for pooling/classifier head.",
            stacklevel=2,
        )
    config.embed_dim = embed_dim  # keep run_config.json / W&B in sync with backbone

    if use_lora:
        backbone = _wrap_with_lora(backbone)
    if device is not None:
        backbone = backbone.to(device)

    classifier = ProteinClassifier(
        num_classes=num_classes,
        embed_dim=embed_dim,
        # embed_dim=int(getattr(config, "embed_dim", 960)),  # old: static config only
        pooling_type=pooling_type,
        dropout=float(getattr(config, "classify_dropout", 0.4)),
        hidden_dim=int(getattr(config, "classifier_hidden_dim", 512)),
    )
    if device is not None:
        classifier = classifier.to(device)

    model = ESMCClassifier(backbone, classifier)
    if device is not None:
        model = model.to(device)
    return model
