"""
config.py — every architectural knob in one place.

Nothing here is inherited from another project. These are OUR numbers.
Scale up by changing a preset, not by rewriting the model.
"""

from dataclasses import dataclass, asdict
import json


@dataclass
class ModelConfig:
    # --- vocabulary ---
    vocab_size: int = 8192          # set by our own tokenizer, not someone else's

    # --- shape ---
    n_layer: int = 8                # transformer blocks stacked
    n_head: int = 8                 # query heads
    n_kv_head: int = 4              # key/value heads (< n_head == grouped-query attention)
    d_model: int = 512              # residual stream width
    d_ff: int = None                # feed-forward width; None -> auto (8/3 * d_model, SwiGLU-adjusted)

    # --- context ---
    block_size: int = 512           # max sequence length the model can see at once
    rope_theta: float = 10000.0     # rotary base frequency; raise it to extend context later

    # --- regularisation ---
    dropout: float = 0.0            # 0.0 for large data, raise to 0.1 if you overfit a small corpus

    # --- numerics ---
    norm_eps: float = 1e-5
    tie_embeddings: bool = True     # share input embedding with output head; saves vocab_size*d_model params

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must divide evenly into n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        if self.d_ff is None:
            # SwiGLU uses two projections in, so we shrink the hidden dim to keep
            # parameter count comparable to a standard 4x MLP. 8/3 is the usual choice.
            hidden = int(8 * self.d_model / 3)
            # round to a multiple of 64 — much friendlier to GPU tensor cores
            self.d_ff = 64 * ((hidden + 63) // 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def n_params_estimate(self) -> int:
        """Rough parameter count, before you build anything."""
        c = self
        emb = c.vocab_size * c.d_model
        attn_per_layer = (
            c.d_model * c.d_model                                  # W_q
            + 2 * c.d_model * (c.n_kv_head * c.head_dim)           # W_k, W_v
            + c.d_model * c.d_model                                # W_o
        )
        ff_per_layer = 3 * c.d_model * c.d_ff                      # gate, up, down
        norms_per_layer = 2 * c.d_model
        per_layer = attn_per_layer + ff_per_layer + norms_per_layer
        total = emb + c.n_layer * per_layer + c.d_model            # + final norm
        if not c.tie_embeddings:
            total += c.vocab_size * c.d_model
        return total

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class TrainConfig:
    # --- data ---
    data_dir: str = "data"
    batch_size: int = 16
    grad_accum_steps: int = 4       # effective batch = batch_size * grad_accum_steps

    # --- optimisation ---
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- housekeeping ---
    eval_interval: int = 250
    eval_iters: int = 50
    ckpt_interval: int = 500
    ckpt_dir: str = "checkpoints"
    log_interval: int = 10
    seed: int = 1337
    compile_model: bool = False     # torch.compile — big speedup on Linux/CUDA, flaky on Windows
    device: str = "auto"
    dtype: str = "auto"             # auto -> bf16 on capable GPUs, fp16 on older, fp32 on CPU


# ----------------------------------------------------------------------------
# Presets. Start at NANO on whatever machine you have, work up.
# ----------------------------------------------------------------------------

PRESETS = {
    # ~1M params. Trains on a CPU in minutes. For proving the pipeline works.
    "nano": ModelConfig(
        vocab_size=4096, n_layer=4, n_head=4, n_kv_head=2,
        d_model=128, block_size=256,
    ),
    # ~25M params. A few hours on a mid-range GPU. First model that says real words.
    "micro": ModelConfig(
        vocab_size=8192, n_layer=8, n_head=8, n_kv_head=4,
        d_model=512, block_size=512,
    ),
    # ~120M params. GPT-2-small territory. Overnight on a decent GPU.
    "small": ModelConfig(
        vocab_size=16384, n_layer=12, n_head=12, n_kv_head=4,
        d_model=768, block_size=1024,
    ),
    # ~450M params. This is where a DGX Spark starts to make sense.
    "base": ModelConfig(
        vocab_size=32768, n_layer=24, n_head=16, n_kv_head=4,
        d_model=1024, block_size=2048,
    ),
}


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        n = cfg.n_params_estimate()
        print(f"{name:8s}  {n/1e6:8.2f}M params   "
              f"d_model={cfg.d_model:5d}  layers={cfg.n_layer:3d}  ctx={cfg.block_size}")
