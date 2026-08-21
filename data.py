"""
data.py — turn raw text into training batches.

Text is tokenised ONCE into a flat uint16/uint32 array on disk, then memory-mapped.
This matters: at training time you want to pull random windows out of a corpus that
may be larger than your RAM, thousands of times per second, without re-tokenising
anything. np.memmap gives you that for free.

Usage:
    python -m core.data prepare --input data/corpus.txt --tokenizer data/tokenizer.json
    python -m core.data prepare-tokenizer --input data/corpus.txt --vocab-size 8192
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import torch

from .tokenizer import Tokenizer


def _dtype_for(vocab_size: int):
    return np.uint16 if vocab_size < 2**16 else np.uint32


# ------------------------------------------------------------------- preparing

def train_tokenizer(input_path: str, out_path: str, vocab_size: int, max_chars: int = 5_000_000):
    """Train our BPE on (a sample of) the corpus.

    BPE training is O(corpus) per merge, so we cap the sample. A few million
    characters is plenty to learn a good merge table — more data helps the MODEL,
    not the tokeniser.
    """
    print(f"reading {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        text = f.read(max_chars)
    print(f"training BPE on {len(text):,} chars -> vocab {vocab_size}")
    tok = Tokenizer()
    tok.train(text, vocab_size=vocab_size, verbose=True)
    tok.save(out_path)
    print(f"saved tokenizer -> {out_path}")
    return tok


def prepare(input_path: str, tokenizer_path: str, out_dir: str,
            val_fraction: float = 0.005, chunk_chars: int = 1_000_000):
    """Tokenise the whole corpus to train.bin / val.bin."""
    tok = Tokenizer.load(tokenizer_path)
    dtype = _dtype_for(tok.vocab_size)
    os.makedirs(out_dir, exist_ok=True)

    print("tokenising corpus (streamed, so file size is not limited by RAM) ...")
    all_ids: list[np.ndarray] = []
    total = 0
    with open(input_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            ids = tok.encode(chunk, allow_special=True)
            all_ids.append(np.array(ids, dtype=dtype))
            total += len(ids)
            print(f"\r  {total:,} tokens", end="", flush=True)
    print()

    arr = np.concatenate(all_ids) if all_ids else np.array([], dtype=dtype)
    n_val = max(1, int(len(arr) * val_fraction))
    train_arr, val_arr = arr[:-n_val], arr[-n_val:]

    train_arr.tofile(os.path.join(out_dir, "train.bin"))
    val_arr.tofile(os.path.join(out_dir, "val.bin"))

    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(f"vocab_size={tok.vocab_size}\n")
        f.write(f"dtype={np.dtype(dtype).name}\n")
        f.write(f"train_tokens={len(train_arr)}\n")
        f.write(f"val_tokens={len(val_arr)}\n")

    print(f"train: {len(train_arr):,} tokens   val: {len(val_arr):,} tokens")
    print(f"wrote -> {out_dir}/train.bin, {out_dir}/val.bin")


# -------------------------------------------------------------------- loading

class BinaryDataset:
    """Memory-mapped view over a .bin file. Samples random windows."""

    def __init__(self, path: str, block_size: int, dtype=np.uint16):
        self.path = path
        self.block_size = block_size
        self.dtype = dtype
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run `python -m core.data prepare` first")
        self.data = np.memmap(path, dtype=dtype, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(
                f"{path} has only {len(self.data)} tokens, need > {block_size+1}. "
                "Use a bigger corpus or a smaller block_size.")

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size: int, device: str):
        # random starting offsets; +1 because targets are inputs shifted by one
        ix = np.random.randint(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = np.stack([self.data[i: i + self.block_size] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1: i + 1 + self.block_size] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if device.startswith("cuda"):
            # pinned + non_blocking lets the copy overlap with compute
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def read_meta(data_dir: str) -> dict:
    meta = {}
    p = os.path.join(data_dir, "meta.txt")
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
    return meta


# ------------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser(description="corpus preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare-tokenizer")
    a.add_argument("--input", required=True)
    a.add_argument("--out", default="data/tokenizer.json")
    a.add_argument("--vocab-size", type=int, default=8192)
    a.add_argument("--max-chars", type=int, default=5_000_000)

    b = sub.add_parser("prepare")
    b.add_argument("--input", required=True)
    b.add_argument("--tokenizer", default="data/tokenizer.json")
    b.add_argument("--out-dir", default="data")
    b.add_argument("--val-fraction", type=float, default=0.005)

    args = ap.parse_args()
    if args.cmd == "prepare-tokenizer":
        train_tokenizer(args.input, args.out, args.vocab_size, args.max_chars)
    else:
        prepare(args.input, args.tokenizer, args.out_dir, args.val_fraction)


if __name__ == "__main__":
    main()
