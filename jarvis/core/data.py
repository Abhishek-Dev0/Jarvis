"""
data.py — turn raw text into training batches.

Text is tokenised ONCE into a flat uint16/uint32 array on disk, then memory-mapped.
This matters: at training time you want to pull random windows out of a corpus that
may be larger than your RAM, thousands of times per second, without re-tokenising
anything. np.memmap gives you that for free.

Usage:
    python -m core.data prepare --input data/corpus.txt --tokenizer data/tokenizer.json
    python -m core.data prepare-tokenizer --input data/corpus.txt --vocab-size 8192
    python -m core.data absorb-web --web-dir data/web --corpus data/corpus.txt
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Iterable

import numpy as np
import torch

from .tokenizer import Tokenizer


def format_chat_turns(turns: Iterable[tuple[str, str]]) -> str:
    """Format a list of (user, assistant) turns as a chat transcript.

    The output is deliberately minimal and strict: each turn becomes
    <|user|>...<|assistant|>...<|eos|> and empty turns are discarded.
    """
    parts: list[str] = []
    for user_text, assistant_text in turns:
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if not user and not assistant:
            continue
        if not user:
            user = ""
        if not assistant:
            assistant = ""
        parts.append(f"<|user|>{user}<|assistant|>{assistant}<|eos|>")
    return "".join(parts)


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


# ------------------------------------------------------------- chat datasets

_DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
              "resolve/main/databricks-dolly-15k.jsonl")


def fetch_dolly(output_path: str, max_combined_chars: int = 700) -> int:
    """Downloads Databricks Dolly 15k (CC BY-SA 3.0 — free for any purpose,
    including commercial, with attribution) and writes it out as the
    [[user, assistant], ...] JSON format-chat expects.

    This is the real chat fine-tune dataset — see the 2026-08-22 roadmap
    notes for why: a from-scratch nano/micro model has never seen a single
    conversation, only raw prose, so it cannot follow the <|user|>/
    <|assistant|> format at all until fine-tuned on examples of it. Dolly
    was picked over Alpaca-style datasets specifically because it's
    human-written, not distilled from another model's outputs, and its
    license is unambiguous (Alpaca's generation via OpenAI's API makes its
    own licensing murkier).

    max_combined_chars filters out examples whose instruction+context+
    response would blow well past a nano/micro model's block_size (256-512
    tokens) — keeps every kept example usable at pretraining's context
    length instead of silently truncating mid-example.
    """
    import requests
    r = requests.get(_DOLLY_URL, timeout=60)
    r.raise_for_status()

    pairs = []
    for line in r.text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        instruction = (rec.get("instruction") or "").strip()
        context = (rec.get("context") or "").strip()
        response = (rec.get("response") or "").strip()
        if not instruction or not response:
            continue
        user = f"{instruction}\n\n{context}" if context else instruction
        if len(user) + len(response) > max_combined_chars:
            continue
        pairs.append([user, response])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f)
    print(f"kept {len(pairs)} pairs (of {r.text.count(chr(10))} total records) -> {output_path}")
    return len(pairs)


# --------------------------------------------------------------- web absorption

def absorb_web(web_dir: str, corpus_path: str) -> int:
    """Fold pages fetched by WebGrowthSkill into the training corpus.

    Idempotent: each manifest entry is marked absorbed=True once appended, so
    re-running only picks up pages saved since the last absorb. The '# source:
    ...' header WebGrowthSkill writes is stripped first — the model should
    learn the prose, not our bookkeeping comments.
    """
    manifest_path = os.path.join(web_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"no manifest at {manifest_path} — nothing to absorb")
        return 0
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    pending = [(h, e) for h, e in manifest.items() if not e.get("absorbed")]
    if not pending:
        print("nothing new to absorb")
        return 0

    chunks = []
    for url_hash, entry in pending:
        fpath = os.path.join(web_dir, entry["file"])
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            lines = f.readlines()
        body = "".join(lines[4:]) if len(lines) > 4 else "".join(lines)
        body = body.strip()
        if body:
            chunks.append(body)
        entry["absorbed"] = True

    if not chunks:
        print("nothing new to absorb")
        return 0

    with open(corpus_path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write("\n\n" + chunk + "\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_chars = sum(len(c) for c in chunks)
    print(f"absorbed {len(chunks)} page(s), {total_chars:,} chars -> {corpus_path}")
    print("re-run prepare-tokenizer / prepare before your next training run")
    return len(chunks)


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

    c = sub.add_parser("absorb-web")
    c.add_argument("--web-dir", default="data/web")
    c.add_argument("--corpus", default="data/corpus.txt")

    d = sub.add_parser("format-chat")
    d.add_argument("--input", required=True, help="JSON file containing a list of [user, assistant] pairs")
    d.add_argument("--output", default="data/chat.txt")

    e = sub.add_parser("fetch-dolly", help="download Databricks Dolly 15k as [user, assistant] pairs JSON")
    e.add_argument("--output", default="data/dolly_pairs.json")
    e.add_argument("--max-combined-chars", type=int, default=700)

    args = ap.parse_args()
    if args.cmd == "prepare-tokenizer":
        train_tokenizer(args.input, args.out, args.vocab_size, args.max_chars)
    elif args.cmd == "prepare":
        prepare(args.input, args.tokenizer, args.out_dir, args.val_fraction)
    elif args.cmd == "absorb-web":
        absorb_web(args.web_dir, args.corpus)
    elif args.cmd == "fetch-dolly":
        fetch_dolly(args.output, args.max_combined_chars)
    elif args.cmd == "format-chat":
        with open(args.input, "r", encoding="utf-8") as f:
            turns = json.load(f)
        chat_text = format_chat_turns(turns)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(chat_text)
        print(f"wrote {len(chat_text):,} chat chars -> {args.output}")


if __name__ == "__main__":
    main()
