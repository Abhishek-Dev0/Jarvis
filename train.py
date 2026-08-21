"""
train.py — the training loop.

Handles the things that separate "it runs" from "it actually learns":
  - warmup + cosine decay learning rate schedule
  - gradient accumulation (large effective batch on small hardware)
  - gradient clipping (stops one bad batch from wrecking the run)
  - mixed precision (bf16 where supported, fp16 + loss scaling otherwise)
  - periodic eval on held-out data so you can SEE overfitting happen
  - resumable checkpoints

Run:
    python -m core.train --preset micro --max-steps 5000
    python -m core.train --resume checkpoints/latest.pt
"""

from __future__ import annotations
import argparse
import math
import os
import time
import numpy as np
import torch

from .config import ModelConfig, TrainConfig, PRESETS
from .model import LanguageModel
from .data import BinaryDataset, read_meta


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str):
    if requested != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}[requested]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    if step > cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, datasets, cfg: TrainConfig, device, ctx):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, device)
            with ctx:
                _, loss = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="micro", choices=list(PRESETS))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    tcfg = TrainConfig(data_dir=args.data_dir, ckpt_dir=args.ckpt_dir)
    if args.max_steps: tcfg.max_steps = args.max_steps
    if args.batch_size: tcfg.batch_size = args.batch_size
    if args.grad_accum: tcfg.grad_accum_steps = args.grad_accum
    if args.lr: tcfg.lr = args.lr
    tcfg.compile_model = args.compile

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = pick_dtype(tcfg.dtype, device)
    ctx = (torch.autocast(device_type=device_type, dtype=dtype)
           if device_type == "cuda" else torch.autocast(device_type="cpu", enabled=False))

    print(f"device={device}  dtype={dtype}")

    # ---- config: from checkpoint if resuming, else from preset + data meta
    meta = read_meta(tcfg.data_dir)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        mcfg = ModelConfig(**ck["model_config"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"resuming from {args.resume} at step {start_step}")
    else:
        mcfg = PRESETS[args.preset]
        if "vocab_size" in meta:
            mcfg.vocab_size = int(meta["vocab_size"])
            print(f"vocab_size <- {mcfg.vocab_size} (from data/meta.txt)")
        ck = None
        start_step = 0
        best_val = float("inf")

    np_dtype = np.uint16 if mcfg.vocab_size < 2**16 else np.uint32
    datasets = {
        "train": BinaryDataset(os.path.join(tcfg.data_dir, "train.bin"), mcfg.block_size, np_dtype),
        "val":   BinaryDataset(os.path.join(tcfg.data_dir, "val.bin"),   mcfg.block_size, np_dtype),
    }

    model = LanguageModel(mcfg).to(device)
    if ck: model.load_state_dict(ck["model"])
    print(f"parameters: {model.num_params()/1e6:.2f}M "
          f"({model.num_params(non_embedding=True)/1e6:.2f}M non-embedding)")

    opt = model.configure_optimizer(tcfg.weight_decay, tcfg.lr,
                                    (tcfg.beta1, tcfg.beta2), device_type)
    if ck: opt.load_state_dict(ck["optimizer"])

    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == torch.float16))

    raw_model = model
    if tcfg.compile_model:
        print("compiling model (first step will be slow) ...")
        model = torch.compile(model)

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum_steps * mcfg.block_size
    print(f"tokens/step: {tokens_per_step:,}   "
          f"total: {tokens_per_step * tcfg.max_steps / 1e6:.1f}M")
    os.makedirs(tcfg.ckpt_dir, exist_ok=True)

    def save(step, val_loss, name):
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": opt.state_dict(),
            "model_config": raw_model.cfg.__dict__,
            "step": step,
            "val_loss": val_loss,
            "best_val": best_val,
        }, os.path.join(tcfg.ckpt_dir, name))

    t0 = time.time()
    model.train()
    for step in range(start_step, tcfg.max_steps):
        lr = lr_at(step, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(tcfg.grad_accum_steps):
            x, y = datasets["train"].get_batch(tcfg.batch_size, device)
            with ctx:
                _, loss = model(x, targets=y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            tps = tokens_per_step * (step - start_step + 1) / max(dt, 1e-9)
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"gnorm {norm:.2f} | {tps:,.0f} tok/s")

        if step > 0 and step % tcfg.eval_interval == 0:
            m = evaluate(model, datasets, tcfg, device, ctx)
            print(f"  eval @ {step}: train {m['train']:.4f}  val {m['val']:.4f}  "
                  f"ppl {math.exp(min(m['val'],20)):.1f}")
            if m["val"] < best_val:
                best_val = m["val"]
                save(step, m["val"], "best.pt")
                print(f"  new best -> checkpoints/best.pt")

        if step > 0 and step % tcfg.ckpt_interval == 0:
            save(step, None, "latest.pt")

    m = evaluate(model, datasets, tcfg, device, ctx)
    print(f"final: train {m['train']:.4f}  val {m['val']:.4f}")
    save(tcfg.max_steps - 1, m["val"], "latest.pt")
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
