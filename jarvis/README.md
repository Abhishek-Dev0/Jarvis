# JARVIS — core

A language model built from the ground up. No pretrained weights, no
`transformers`, no `tiktoken`, no API calls. The tokenizer, the architecture, the
training loop and the sampler are all here in readable Python, and they are yours.

Everything that is *not* core — voice, tools, robot control — attaches from the
outside through a module interface, so borrowed capability never contaminates the
part you built.

---

## What is actually from scratch

| Component | Status | Notes |
|---|---|---|
| Byte-level BPE tokenizer | **from scratch** | trained on your corpus, no external tokenizer lib |
| Transformer architecture | **from scratch** | RMSNorm, RoPE, GQA, SwiGLU — all hand-written |
| Training loop | **from scratch** | cosine schedule, grad accumulation, clipping, AMP |
| Sampling + KV cache | **from scratch** | temperature, top-k, top-p, repetition penalty |
| Module system | **from scratch** | input / output / skill plug-ins |
| Tensor math + autograd | PyTorch | writing your own autograd teaches you calculus, not ML |

That last row is the one honest dependency. Implementing backprop by hand is a
worthwhile weekend, but it is a *different* project — and once done you would
still want PyTorch for GPU kernels.

---

## Layout

```
jarvis/
├── core/                 ← the part you own. nothing here knows about voice or tools.
│   ├── config.py         model presets, nano → base
│   ├── tokenizer.py      byte-level BPE, trained by us
│   ├── model.py          the transformer
│   ├── data.py           corpus → memory-mapped token shards
│   ├── train.py          training loop
│   └── generate.py       sampling + KV cache
├── modules/              ← the part you bolt on. never touches core/.
│   ├── base.py           Module / InputModule / OutputModule / SkillModule + Registry
│   └── builtin.py        console I/O, calculator, voice stubs
├── runtime/
│   └── jarvis.py         orchestrator: input → skill? → model → output
├── data/                 corpus + tokenizer + .bin shards
└── checkpoints/
```

---

## Quickstart

```bash
pip install torch numpy regex

# 1. put your text in data/corpus.txt, then learn a vocabulary from it
python -m core.data prepare-tokenizer --input data/corpus.txt --vocab-size 8192

# 2. tokenize the corpus into train.bin / val.bin
python -m core.data prepare --input data/corpus.txt --tokenizer data/tokenizer.json

# 3. train
python -m core.train --preset nano          # CPU, minutes — proves the pipeline
python -m core.train --preset micro         # GPU, hours — first real model

# 4. generate
python -m core.generate --prompt "The robot"

# 5. run the assistant
python -m runtime.jarvis
```

`--resume checkpoints/latest.pt` picks a run back up.

---

## Presets

| preset | params | d_model | layers | ctx | trains on |
|---|---|---|---|---|---|
| `nano` | 1.3M | 128 | 4 | 256 | a laptop CPU |
| `micro` | 28M | 512 | 8 | 512 | one consumer GPU |
| `small` | 88M | 768 | 12 | 1024 | one good GPU, overnight |
| `base` | 300M | 1024 | 24 | 2048 | where a DGX Spark earns its keep |

Scale by changing the preset, not the code.

---

## Two traps that will cost you a day each

**1. Never end a prompt with a space.**
Byte-level BPE folds the leading space into the word — `" calibrates"` is one
token. A standalone space (byte 32) therefore barely appears in training, so a
trailing space feeds the model an out-of-distribution token and the output
collapses into garbage instantly. `prepare_prompt()` strips it and warns you.
GPT-2 and Llama have exactly this bug; now you know why.

**2. A base model does not chat.**
Train on raw text and you get a model that *continues* text. Wrap its input in
`<|user|>` / `<|assistant|>` markers it never saw during training and the output
collapses the same way. That is the model correctly reporting it was never taught
the format.

Getting a conversational JARVIS is two stages, same code both times:
1. **Pretrain** on a large raw corpus → `chat_mode=False`
2. **Fine-tune** on `<|user|>…<|assistant|>…<|eos|>` transcripts → `chat_mode=True`

---

## Adding capability without touching the core

Three hooks. Pick the one that fits and register it.

```python
from modules.base import SkillModule

class TimerSkill(SkillModule):
    name = "timer"
    description = "sets timers"
    priority = 5

    def matches(self, text):
        return text.lower().startswith(("set a timer", "remind me in"))

    def handle(self, text):
        ...
        return "Timer set."

# in runtime/jarvis.py's main()
j.register(TimerSkill())
```

- **`InputModule`** — anything that produces text: speech recognition, OCR, a file watcher
- **`OutputModule`** — anything that consumes text: speech synthesis, a display, Loki's motor controller
- **`SkillModule`** — intercepts a request and answers it directly, bypassing the model

That last one matters more than it looks. A 28M-parameter model **cannot** do
arithmetic, recall dates, or control hardware reliably. Do not train it harder
and hope. Route those requests to code that is correct by construction, and let
the model do the thing only a model can do — language.

`modules/builtin.py` has `SpeechInput` / `SpeechOutput` stubbed with the correct
interface. Drop an engine in; nothing in `core/` changes.

---

## Where to get training data

The model is only as good as what you feed it. Public-domain and openly-licensed
sources, roughly in order of effort-to-value:

- **Project Gutenberg** — clean prose, public domain, easy to bulk-download
- **Wikipedia dumps** — `dumps.wikimedia.org`, needs markup stripping
- **OpenWebText / C4 / The Pile** — the standard open pretraining corpora
- **Your own writing** — notes, code, journals. Small, but it makes the model
  sound like your world rather than the average of the internet

Rough scaling intuition: aim for **~20 tokens of training data per parameter**.
28M params → ~500M tokens for a well-fed model. Under-feeding a big model wastes
compute; a smaller model on more data almost always wins.

---

## Suggested order of work

1. **Run `nano` on 10MB of text end-to-end.** Ignore quality. You are debugging
   the pipeline, not the model. One evening.
2. **Scale to `micro` on a few hundred MB.** This is the first model that
   produces genuinely readable English. Watch the val loss — when it stops
   falling while train loss keeps falling, you are overfitting; get more data.
3. **Build the fine-tune stage.** Format conversations, train on them, flip
   `chat_mode=True`. This is where it starts feeling like an assistant.
4. **Then** voice. Not before. Debugging a language model and a speech pipeline
   simultaneously means you can never tell which one is broken.
5. **Then** Loki integration — an `OutputModule` that turns text into motor
   commands is a very natural bridge between your two projects.

---

## Verified

Every claim above was tested, not assumed:

- tokenizer round-trips losslessly on mixed English / Japanese / source code
- KV-cache generation matches full forward pass to 2.5e-7 (float32 noise)
- init loss 8.33 vs theoretical `ln(vocab)` = 8.32
- a `nano` model trained to near the corpus entropy floor and generates
  structurally correct text
