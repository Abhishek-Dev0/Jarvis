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
│   ├── base.py            Module / InputModule / OutputModule / SkillModule + Registry
│   ├── builtin.py         console I/O, calculator, speech I/O interfaces
│   ├── voice.py           faster-whisper STT + Kokoro/Piper TTS, 11 languages
│   ├── translate.py       Argos Translate bridge for non-English speech
│   ├── hardware.py        detects CPU/RAM/GPU, auto-sizes whisper + the reasoning model
│   ├── reasoning.py       local LLM via Ollama (real conversation/knowledge), MCP tool-calling
│   ├── web.py             live web search + page-saving for future training
│   ├── os_control.py      open/close/list applications — security-gated
│   ├── hardware_io.py     serial/Arduino I/O — security-gated
│   ├── biosignal.py       EMG muscle-activity detection (pure signal processing)
│   ├── mcp_client.py      connects to MCP servers, exposes tools to reasoning.py
│   └── market_analysis.py honest backtesting (buy-and-hold / SMA) — no live trading
├── runtime/
│   └── jarvis.py          orchestrator: input → skill? → model (+ tools) → output
├── security.py            SecurityGate — passphrase + voiceprint + face verification
├── self_modify.py         draft → sandbox-test → gated human approval, never auto-commits
├── vendor_models.py       backs up downloaded model weights for offline setup
├── data/                  corpus + tokenizer + .bin shards + downloaded voice/translate models
└── checkpoints/
```

---

## Quickstart

```bash
pip install -r jarvis/requirements.txt

# 1. put your text in jarvis/data/corpus.txt, then learn a vocabulary from it
python -m core.data prepare-tokenizer --input data/corpus.txt --vocab-size 8192

# 2. tokenize the corpus into train.bin / val.bin
python -m core.data prepare --input data/corpus.txt --tokenizer data/tokenizer.json

# 3. train
python -m core.train --preset nano          # CPU, minutes — proves the pipeline
python -m core.train --preset micro         # GPU, hours — first real model

# 4. generate
python -m core.generate --prompt "The robot"

# 5. run the assistant (from the repo root)
python -m jarvis                  # console
python -m jarvis --voice          # microphone + spoken reply, fully local
```

Steps 1–4 run from inside `jarvis/`; step 5 runs from the repo root. `--resume
checkpoints/latest.pt` picks a training run back up. `python -m jarvis --help` lists
every flag — voice persona/language, whisper/reasoning model overrides, and `--no-*`
switches to disable any individual module (os control, hardware, MCP, market
analysis, reasoning, self-modify).

There's also a real desktop app now — `python -m jarvis.gui` (dark theme, persona
switch, a live on/off toggle per capability, chat+mic, a terminal panel, and a
Self-Modify tab with a Commit & Push to GitHub button). See
[jarvis/README.md's "Desktop app (GUI)" section](jarvis/README.md#desktop-app-gui)
for the details, and `packaging/build.ps1` to build an installable
`Jarvis-Setup.exe`.

---

## Getting started interactively

Double-click `Start JARVIS.bat` (repo root), or the cat-icon shortcut on the Desktop / at
the repo root (`Start JARVIS.lnk` — machine-specific, gitignored). Voice mode, output
visible in that window, with a small animated ASCII cat (`jarvis/modules/mascot.py`) that
flaps its mouth in sync with actual TTS playback and does a short tail-wag after each
reply — purely cosmetic, `--no-mascot` disables it. Say "switch to eve" or "switch to
jarvis" any time to change both the voice persona and the mascot — ungated, it's cosmetic,
not security-relevant. `run_jarvis.bat` is a *different* file used only by
`Launch JARVIS.vbs` for a silent background start (output goes to
`jarvis/data/logs/jarvis.log` instead of a console, since there isn't one) — running it
directly looks like nothing is happening, on purpose.

First run: JARVIS itself offers to set up your admin passphrase the moment it starts —
no separate command needed, and the passphrase never passes through anything but you and
the running program. (You can also run `python -m jarvis.security enroll` directly, with
`--voice` for a spoken enrollment or `--face` to also enroll a face embedding via webcam.)
Three ways to become admin once enrolled: the automatic wake-time voiceprint challenge,
saying "I'm the admin" any time mid-session (asks for the passphrase, one retry allowed),
or saying "recognize me" for a face-only check — weaker (single factor) on purpose, for
when you don't want to say the passphrase out loud in public. Say "change the phrase" to
re-enroll — this requires proving you're already admin first, same as any real
"change password" flow. Voiceprint and face embeddings are encrypted at rest (Windows
DPAPI, tied to your OS account) — never stored in plain form.

---

## Beyond the core model

The from-scratch model is JARVIS's identity layer, not its only source of capability —
modules attach real capability from outside it (see "Adding capability without touching
the core" below). As of this writing that includes: a real desktop app (dark theme,
per-capability toggles, chat+mic, terminal, a Markets tab, self-modify UI — see
`jarvis/README.md`), voice I/O in 11 languages, a local reasoning LLM (Ollama) for actual
conversation and knowledge — including MCP tool-calling, so it can use real external
tools when connected to one — OS-level automation (open/close/list applications),
serial/Arduino hardware I/O with EMG muscle-activity sensing for robotics/exoskeleton
projects, honest historical market backtesting (including an out-of-sample ML strategy
signal) and live indicator charts/news search, parametric computational-design geometry
export for 3D printing, ambient vision (face presence + scene-change watching, off by
default), offline model vendoring, and a self-modify pipeline that drafts and
sandbox-tests its own patches but never applies — or commits/pushes — one without gated
human approval. Every action that changes system state outside the conversation
(launching an app, sending a serial command, applying a self-modify proposal) goes
through `security.py`'s `SecurityGate` first.

---

## Running fully offline

Every model here downloads its weights **once** and runs offline forever after — none of
it calls an API at inference time. `python -m jarvis.vendor_models status` shows what's
already downloaded and where (Piper/Kokoro voices, faster-whisper's Hugging Face cache,
Argos Translate packages, Ollama's model blobs); `backup` copies whatever's present into
`vendor/` (gitignored, local only) so a re-setup — or a move to a new machine — never
needs those sources reachable again. `restore` copies it back into place. See
`vendor_models.py`'s docstring for exactly what is and isn't included (deliberately not
`data/security/` — that's your passphrase/voiceprint/face data, not a model weight).

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

Both stages are done, end to end, with real data (not a synthetic smoke test):

```bash
python -m core.data fetch-dolly --output data/dolly_pairs.json   # Databricks Dolly 15k, CC BY-SA 3.0
python -m core.data format-chat --input data/dolly_pairs.json --output data/chat.txt
python -m core.data prepare --input data/chat.txt --tokenizer data/tokenizer.json \
    --out-dir data/chat --val-fraction 0.05
python -m core.train --preset nano --data-dir data/chat --ckpt-dir checkpoints/chat \
    --resume checkpoints/best.pt --max-steps 3000
python -m jarvis --chat-mode --ckpt checkpoints/chat/best.pt   # from the repo root
```

9,715 Dolly examples (filtered to fit the nano preset's 256-token context), 3,000 steps,
6 minutes on an RTX 3050. Val loss dropped from 6.22 (raw pretrain, never seen the chat
format at all) to 4.36. Mechanically correct — the model reliably learned the
`<|user|>`/`<|assistant|>`/`<|eos|>` structure and stops cleanly — but content quality is
weak, honestly: a 1.8M-parameter model doesn't have the capacity to be coherent or factual,
fine-tuning changes what format it outputs, not how much it actually knows. `--chat-mode`
against `checkpoints/chat/best.pt` is real and working; don't expect real answers out of it
at this scale. Fine-tuning is otherwise scale-invariant — the same commands against a
`micro`/`small` preset trained on more data will produce something actually useful.

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
   `chat_mode=True`. This is where it starts feeling like an assistant — see
   "Two traps" above for the real commands/numbers; done against real data, not
   just a mechanics test, as of 2026-08-22 (nano preset — scale up the preset for
   quality, the pipeline itself doesn't change).
4. **Then** voice. Not before. Debugging a language model and a speech pipeline
   simultaneously means you can never tell which one is broken.
5. **Then** Loki integration — `modules/hardware_io.py`'s `SerialLink`/`HardwareSkill`
   now cover this: connect over serial, send commands, read sensor data, all
   security-gated. `modules/biosignal.py` adds EMG muscle-activity detection on top,
   for driving an actuator from a wearable sensor.

---

## Verified

Every claim above was tested, not assumed:

- tokenizer round-trips losslessly on mixed English / Japanese / source code
- KV-cache generation matches full forward pass to 2.5e-7 (float32 noise)
- init loss 8.33 vs theoretical `ln(vocab)` = 8.32
- a `nano` model trained to near the corpus entropy floor and generates
  structurally correct text
- 91 automated tests across `tests/` covering every module built beyond the
  core (security, hardware I/O, biosignal, MCP, market analysis, self-modify,
  vendoring, admin-trigger phrase handling)

---

## License

**Proprietary — All Rights Reserved.** Copyright (c) 2026 Abhishek-Dev0.

No part of this repository — source code, trained models, checkpoints, training
data, or documentation — may be used, copied, modified, or redistributed
without prior written permission from the copyright holder. This is a private
repository; being granted access to it does not, by itself, grant any of those
rights. Full terms, including the no-warranty and enforcement provisions: see
[LICENSE](LICENSE).
