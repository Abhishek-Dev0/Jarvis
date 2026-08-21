"""
tokenizer.py — byte-level Byte-Pair Encoding, written from scratch.

No tiktoken. No HuggingFace. No sentencepiece.

Why byte-level: we start from the 256 raw bytes, so there is no such thing as an
unknown character. Japanese, Hindi, emoji, source code, binary garbage — it all
encodes. The merges we learn on top are OURS, derived from OUR corpus.

The algorithm:
  1. Start with a vocabulary of the 256 possible byte values.
  2. Count every adjacent pair of tokens in the corpus.
  3. Take the most frequent pair, mint a new token for it, replace every occurrence.
  4. Repeat until the vocabulary is the size we asked for.

Encoding then just replays those merges in the order they were learned.
"""

from __future__ import annotations
import json
import os
import regex as _regex_module  # only for the split pattern; see note below
from collections import Counter
from typing import Iterable


# GPT-style pre-tokenisation pattern. This stops merges from crossing word
# boundaries in stupid ways (e.g. gluing "dog" and "." into one token, or
# learning a token that spans a space plus punctuation plus a digit).
# If you'd rather have zero third-party imports at all, set USE_REGEX = False
# and it falls back to a hand-rolled splitter below. The regex version gives
# noticeably better compression.
SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
USE_REGEX = True


def _fallback_split(text: str) -> list[str]:
    """Pure-stdlib pre-tokeniser. Splits on whitespace boundaries, keeps the
    leading space attached to the word (the trick that makes ' the' one token)."""
    out, buf = [], ""
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        elif buf and (ch.isalnum() != buf[-1].isalnum()):
            out.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _get_stats(ids: list[int], counts: Counter | None = None) -> Counter:
    """Count how often each adjacent pair appears."""
    counts = Counter() if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """Byte-level BPE. Train it once on your corpus, save it, reuse forever.

    Special tokens live at the TOP of the vocabulary so that adding one later
    never renumbers anything the model already learned.
    """

    SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>"]

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # pair -> new token id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        self._pat = _regex_module.compile(SPLIT_PATTERN) if USE_REGEX else None

    # ------------------------------------------------------------------ train

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        assert vocab_size >= 256 + len(self.SPECIALS), "vocab_size too small"
        n_merges = vocab_size - 256 - len(self.SPECIALS)

        chunks = self._pretokenize(text)
        ids_list = [list(ch.encode("utf-8")) for ch in chunks]

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for k in range(n_merges):
            stats = Counter()
            for ids in ids_list:
                _get_stats(ids, stats)
            if not stats:
                if verbose:
                    print(f"  corpus exhausted after {k} merges")
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + k
            ids_list = [_merge(ids, pair, new_id) for ids in ids_list]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and (k + 1) % 250 == 0:
                token = self.vocab[new_id]
                shown = token.decode("utf-8", errors="replace")
                print(f"  merge {k+1:5d}/{n_merges}  count={stats[pair]:7d}  -> {shown!r}")

        # specials sit above every learned merge
        base = 256 + len(self.merges)
        for i, s in enumerate(self.SPECIALS):
            self.special_to_id[s] = base + i
            self.id_to_special[base + i] = s

        if verbose:
            raw = len(text.encode("utf-8"))
            enc = len(self.encode(text, allow_special=False))
            print(f"  vocab size : {self.vocab_size}")
            print(f"  compression: {raw/max(enc,1):.2f}x  ({raw} bytes -> {enc} tokens)")

    def _pretokenize(self, text: str) -> list[str]:
        if self._pat is not None:
            return self._pat.findall(text)
        return _fallback_split(text)

    # ----------------------------------------------------------------- encode

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as literal text."""
        out = []
        for chunk in self._pretokenize(text):
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                stats = _get_stats(ids)
                # apply the merge that was learned EARLIEST among those available
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = _merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode, optionally honouring <|...|> markers as single tokens."""
        if not allow_special or not self.special_to_id:
            return self.encode_ordinary(text)

        # split the text around any special token strings we know about
        import re as _re
        pattern = "(" + "|".join(_re.escape(s) for s in self.special_to_id) + ")"
        parts = _re.split(pattern, text)
        out = []
        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                out.append(self.special_to_id[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    # ----------------------------------------------------------------- decode

    def decode(self, ids: Iterable[int]) -> str:
        parts = []
        for i in ids:
            if i in self.id_to_special:
                parts.append(self.id_to_special[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            # unknown ids are silently dropped rather than crashing mid-generation
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- misc

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.SPECIALS)

    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "merges": [[list(p), i] for p, i in self.merges.items()],
            "specials": self.SPECIALS,
            "special_to_id": self.special_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        tok.merges = {tuple(p): i for p, i in payload["merges"]}
        tok.SPECIALS = payload["specials"]
        tok.special_to_id = {k: int(v) for k, v in payload["special_to_id"].items()}
        tok.id_to_special = {v: k for k, v in tok.special_to_id.items()}
        # rebuild the byte strings for every merged token, in merge order
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        return tok


if __name__ == "__main__":
    sample = (
        "The quick brown fox jumps over the lazy dog. " * 40
        + "こんにちは、世界。これはテストです。" * 20
        + "def hello(name):\n    return f'hi {name}'\n" * 20
    )
    t = Tokenizer()
    t.train(sample, vocab_size=1000, verbose=True)
    probe = "The quick brown fox says こんにちは"
    ids = t.encode(probe)
    print("\nround trip:", t.decode(ids) == probe)
    print("tokens:", ids[:20], "...")
