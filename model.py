"""
model.py — a decoder-only transformer, written from first principles.

Every component here is implemented by hand. The only thing borrowed from torch
is tensor math and autograd — we are not importing anyone's attention block,
anyone's normalisation layer, anyone's pretrained weights.

Architecture choices and WHY (this is the part worth understanding):

  RMSNorm instead of LayerNorm
      LayerNorm subtracts the mean, then divides by the standard deviation.
      It turns out the mean-subtraction does almost nothing useful. RMSNorm
      just divides by the root-mean-square. Fewer ops, same quality.

  RoPE instead of learned position embeddings
      Rather than adding a "position vector" to each token, we ROTATE the
      query and key vectors by an angle proportional to their position. The
      dot product between two rotated vectors then depends only on their
      RELATIVE distance — which is the thing that actually matters. It also
      means we can extend context length later by changing one number.

  Grouped-Query Attention (GQA)
      Every query head gets its own view, but heads SHARE key/value pairs in
      groups. This shrinks the KV cache during generation by n_head/n_kv_head,
      which is the single biggest memory cost when running the model.

  SwiGLU feed-forward
      Instead of one projection + ReLU, we use two projections where one gates
      the other through a SiLU. Costs a third projection, buys a consistent
      quality win.

  Pre-norm residuals
      Normalise BEFORE each sub-layer, not after. Makes deep stacks trainable
      without warmup gymnastics.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ---------------------------------------------------------------- normalisation

class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2)) * weight"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in fp32 for stability even when the rest of the net is bf16
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# ------------------------------------------------------------- rotary embedding

def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables for rotary position embedding.

    Frequencies decay geometrically across the head dimension: the first pairs
    of channels rotate fast (encode fine local position), the last pairs rotate
    slowly (encode coarse global position).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                       # (seq_len, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate pairs of channels. x: (B, n_head, T, head_dim)"""
    # split each head_dim into interleaved halves
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


# ------------------------------------------------------------------- attention

class Attention(nn.Module):
    """Causal grouped-query self-attention with an optional KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head   # how many q heads share one kv head

        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- KV cache: during generation we only feed one new token at a time,
        # so we keep the keys/values of everything that came before.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # expand shared kv heads out to match the number of query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # When generating with a cache, T==1 and the query attends to everything
        # in the cache — so causal masking is only needed when T > 1.
        is_causal = kv_cache is None or T > 1

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y), new_cache


# ---------------------------------------------------------------- feed-forward

class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) )"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ----------------------------------------------------------------------- block

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.ff(self.ff_norm(x))
        return x, new_cache


# ----------------------------------------------------------------------- model

class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the residual-output projections by depth. Without this,
        # the residual stream grows with every layer and deep models diverge.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self._rope_cache = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, seq_len, offset, device, dtype):
        need = seq_len + offset
        if (self._rope_cache is None
                or self._rope_cache[0].shape[0] < need
                or self._rope_cache[0].device != device):
            cos, sin = build_rope_cache(
                max(need, self.cfg.block_size), self.cfg.head_dim,
                self.cfg.rope_theta, device, dtype)
            self._rope_cache = (cos, sin)
        cos, sin = self._rope_cache
        return cos[offset:offset + seq_len], sin[offset:offset + seq_len]

    def forward(self, idx, targets=None, kv_caches=None, pos_offset=0):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope(T, pos_offset, idx.device, x.dtype)

        new_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            cache_in = kv_caches[i] if kv_caches is not None else None
            x, cache_out = block(x, cos, sin, cache_in)
            if new_caches is not None:
                new_caches.append(cache_out)

        x = self.final_norm(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss

        # inference: only the last position matters, so skip the rest of the head
        logits = self.head(x[:, -1:, :])
        return logits, new_caches

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        """Weight-decay the matrices, don't weight-decay the vectors.

        Decaying norm gains and biases is a well-known way to make training
        slightly worse for no reason, so we split the parameters into two groups.
        """
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


if __name__ == "__main__":
    from config import PRESETS
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cfg = PRESETS["nano"]
    m = LanguageModel(cfg)
    print(f"params: {m.num_params()/1e6:.2f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, targets=x)
    print("logits:", logits.shape, "loss:", loss.item())
    print("expected loss at init ~", math.log(cfg.vocab_size))
