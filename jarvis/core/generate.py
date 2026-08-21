"""
generate.py — turning the model's probability distribution into text.

The model gives you a score for every token in the vocabulary. How you pick from
that distribution matters as much as the model itself:

  temperature      divides the logits. <1 sharpens (safer, repetitive),
                   >1 flattens (wilder). 0 = always take the top token.
  top_k            only consider the k highest-scoring tokens.
  top_p (nucleus)  only consider the smallest set of tokens whose probabilities
                   sum to p. Adapts its cutoff to how confident the model is.
  repetition_penalty
                   down-weight tokens already produced. Small models loop badly
                   without this.

Generation is incremental: we feed one token, keep the KV cache, feed the next.
That turns an O(n^2) process into O(n).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def prepare_prompt(tok, text: str, add_bos: bool = True, warn: bool = True) -> list[int]:
    """Encode a prompt safely.

    THE TRAILING SPACE TRAP — read this once, save yourself an afternoon.

    Byte-level BPE attaches the leading space to a word: " calibrates" is a
    SINGLE token. So a standalone space byte (id 32) almost never occurs in
    training data. If your prompt ends in a space, you hand the model a token it
    has essentially never seen, and the output degrades into garbage —
    immediately and completely.

    This is not a flaw in our implementation; GPT-2, GPT-3 and Llama all behave
    the same way. The fix is simply: never end a prompt with whitespace. Let the
    model generate the space as part of the next word.
    """
    stripped = text.rstrip()
    if warn and stripped != text:
        print("[tokenizer] note: stripped trailing whitespace from prompt "
              "(it maps to a rare standalone-space token and degrades output)")
    ids = tok.encode(stripped, allow_special=True)
    return ([tok.bos_id()] + ids) if add_bos else ids


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.1,
    eos_id: int | None = None,
    stream_callback=None,
):
    """idx: (1, T) prompt token ids. Yields nothing; returns the full sequence."""
    model.eval()
    device = idx.device
    cfg = model.cfg

    # prime the cache with the whole prompt in one pass
    caches = [(None, None)] * cfg.n_layer
    prompt = idx if idx.shape[1] <= cfg.block_size else idx[:, -cfg.block_size:]
    logits, caches = model(prompt, kv_caches=caches, pos_offset=0)
    pos = prompt.shape[1]
    generated = idx

    for _ in range(max_new_tokens):
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            for tok in set(generated[0].tolist()):
                if logits[0, tok] > 0:
                    logits[0, tok] /= repetition_penalty
                else:
                    logits[0, tok] *= repetition_penalty

        if temperature <= 0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = probs.cumsum(dim=-1)
                # keep tokens up to and including the one that crosses top_p
                remove = cumulative - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_tok], dim=1)

        if stream_callback is not None:
            stream_callback(next_tok.item())

        if eos_id is not None and next_tok.item() == eos_id:
            break

        # context window is full -> drop the cache and re-prime on a trimmed window
        if pos >= cfg.block_size:
            window = generated[:, -(cfg.block_size - 1):]
            caches = [(None, None)] * cfg.n_layer
            logits, caches = model(window, kv_caches=caches, pos_offset=0)
            pos = window.shape[1]
            continue

        logits, caches = model(next_tok, kv_caches=caches, pos_offset=pos)
        pos += 1

    return generated


def load_for_inference(ckpt_path: str, tokenizer_path: str, device: str = "auto"):
    """Convenience loader: checkpoint + tokenizer -> (model, tokenizer, device)."""
    from .config import ModelConfig
    from .model import LanguageModel
    from .tokenizer import Tokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = LanguageModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.load(tokenizer_path)
    return model, tok, device


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model, tok, device = load_for_inference(args.ckpt, args.tokenizer, args.device)
    ids = prepare_prompt(tok, args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    print(args.prompt, end="", flush=True)
    buf = []

    def on_token(t):
        buf.append(t)
        # decode incrementally; multi-byte characters need buffering
        text = tok.decode(buf)
        if "\ufffd" not in text:
            print(text, end="", flush=True)
            buf.clear()

    generate(model, x,
             max_new_tokens=args.max_new_tokens,
             temperature=args.temperature,
             top_k=args.top_k, top_p=args.top_p,
             eos_id=tok.eos_id(),
             stream_callback=on_token)
    print()
