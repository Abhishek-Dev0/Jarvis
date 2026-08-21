"""
jarvis.py — the orchestrator.

Owns the loop: get input -> check skills -> otherwise ask the model -> emit output.
Also owns conversation state and prompt formatting.

Deliberately thin. All the intelligence is in core/, all the capability is in
modules/. This file is just the wiring, which means you can rewrite it without
touching anything that matters.
"""

from __future__ import annotations
import os
import sys
import torch

try:
    from jarvis.core.generate import generate, load_for_inference, prepare_prompt
    from jarvis.modules.base import Registry
    from jarvis.modules.builtin import ConsoleInput, ConsoleOutput, CalculatorSkill, SpeechInput, SpeechOutput
    from jarvis.modules.web import WebSearchSkill, WebGrowthSkill
    from jarvis.modules.reasoning import ReasoningSkill
    from jarvis.modules.os_control import OSControlSkill
    from jarvis.modules.hardware_io import HardwareSkill
    from jarvis.modules.mcp_client import MCPSkill
    from jarvis.modules.market_analysis import MarketAnalysisSkill
    from jarvis.security import SecurityGate
except ImportError:  # pragma: no cover - legacy direct execution
    from core.generate import generate, load_for_inference, prepare_prompt
    from modules.base import Registry
    from modules.builtin import ConsoleInput, ConsoleOutput, CalculatorSkill, SpeechInput, SpeechOutput
    from modules.web import WebSearchSkill, WebGrowthSkill
    from modules.reasoning import ReasoningSkill
    from modules.os_control import OSControlSkill
    from modules.hardware_io import HardwareSkill
    from modules.mcp_client import MCPSkill
    from modules.market_analysis import MarketAnalysisSkill
    from security import SecurityGate

# Phrases that mean "stop the process." Handled in run() itself (gated by
# SecurityGate) rather than by input modules swallowing them, so verification
# can't be bypassed by whichever module happens to read the phrase first.
# English is the always-available baseline; Jarvis.refresh_multilingual_phrases()
# extends both sets into every language in voice.SUPPORTED_LANGUAGES when a
# Translator is available (see main()), so "quit"/"yes" work no matter what
# language the conversation is actually happening in.
_SHUTDOWN_PHRASES_EN = {"quit", "exit", "shutdown", "shut down", "power off",
                         "turn off", "stop jarvis", "goodbye jarvis"}
_AFFIRMATIVE_PHRASES_EN = {"yes", "yeah", "yep", "yup", "correct", "affirmative"}


def _normalize_phrase(text: str) -> str:
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _is_shutdown_request(text: str, phrases: set[str]) -> bool:
    return _normalize_phrase(text) in phrases

# Same fix as ConsoleOutput.setup() (modules/builtin.py), applied here too:
# diagnostics printed before any module registers — e.g. load_model()'s
# "no checkpoint found" notice — would otherwise hit Windows' legacy console
# codepage and crash on the first em dash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# jarvis/ — the package root, however this was launched: `python -m jarvis`
# from the repo root, or the legacy `python -m runtime.jarvis` from inside
# jarvis/. Anchor default data paths here instead of leaving them relative to
# whatever the cwd happens to be, or the repo-root invocation silently can't
# find checkpoints/best.pt (it looks for ./checkpoints/best.pt in the repo
# root, not jarvis/checkpoints/best.pt).
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Jarvis:
    def __init__(self, ckpt="checkpoints/best.pt", tokenizer="data/tokenizer.json",
                 device="auto", max_history=6, gen_kwargs=None, chat_mode=False,
                 admin_name="admin"):
        self.registry = Registry()
        self.history: list[tuple[str, str]] = []
        self.max_history = max_history
        self.gen_kwargs = gen_kwargs or dict(
            max_new_tokens=200, temperature=0.8, top_k=40,
            top_p=0.95, repetition_penalty=1.1,
        )
        # chat_mode=False  -> raw text continuation (what a BASE model does)
        # chat_mode=True   -> wrap turns in <|user|>/<|assistant|> markers.
        #                     ONLY use this after fine-tuning on that format,
        #                     otherwise you feed the model unseen tokens and
        #                     the output collapses.
        self.chat_mode = chat_mode
        self.model = self.tok = None
        self.device = device
        self._ckpt, self._tokenizer_path = ckpt, tokenizer
        # No mic engine by default -> passphrase-only gate (voice factor
        # skipped even if a voiceprint is enrolled). main() replaces this
        # with a real SecurityGate(mic_engine=...) when --voice is active.
        self.security = SecurityGate()
        # None unless main() wires one up (see modules/translate.py) — with
        # no translator, non-English input/output just isn't bridged; the
        # core model only understands English regardless.
        self.translator = None
        # Session-level admin state (see security.SecurityGate.wake_challenge
        # and run()) — in memory only, resets every process restart. Whoever
        # this session belongs to, by name, for the "are you the admin, X?"
        # wake challenge.
        self.admin_name = admin_name
        self.is_admin = False
        self.shutdown_phrases = set(_SHUTDOWN_PHRASES_EN)
        self.affirmative_phrases = set(_AFFIRMATIVE_PHRASES_EN)

    def refresh_multilingual_phrases(self) -> None:
        """Extends shutdown/affirmative phrase matching into every supported
        language via self.translator. Safe to call with no translator (no-op)
        or call again later if the translator becomes available."""
        if self.translator is None:
            return
        try:
            from jarvis.modules.voice import SUPPORTED_LANGUAGES
        except ImportError:  # pragma: no cover - legacy direct execution
            from modules.voice import SUPPORTED_LANGUAGES
        for lang in SUPPORTED_LANGUAGES:
            if lang == "en":
                continue
            for phrase in _SHUTDOWN_PHRASES_EN:
                translated = self.translator.translate(phrase, "en", lang)
                if translated:
                    self.shutdown_phrases.add(_normalize_phrase(translated))
            for phrase in _AFFIRMATIVE_PHRASES_EN:
                translated = self.translator.translate(phrase, "en", lang)
                if translated:
                    self.affirmative_phrases.add(_normalize_phrase(translated))

    # ------------------------------------------------------------------ setup

    def load_model(self):
        if not os.path.exists(self._ckpt):
            print(f"[jarvis] no checkpoint at {self._ckpt} — "
                  "skills will work, generation will not.")
            print("[jarvis] train one with: python -m core.train --preset micro")
            return False
        self.model, self.tok, self.device = load_for_inference(
            self._ckpt, self._tokenizer_path, self.device)
        print(f"[jarvis] model loaded: {self.model.num_params()/1e6:.1f}M params "
              f"on {self.device}")
        return True

    def register(self, module):
        return self.registry.register(module)

    # --------------------------------------------------------------- prompting

    def build_prompt(self, user_text: str) -> str:
        """Format the conversation for the model.

        A base model trained on raw text does NOT chat — it continues text. If
        you wrap its input in <|user|>/<|assistant|> markers it has never seen,
        you get garbage. That is not a bug, it is the model correctly telling you
        it was never taught this format.

        The path to a real conversational JARVIS is two stages:
          1. pretrain on a large raw-text corpus   -> chat_mode=False
          2. fine-tune on <|user|>/<|assistant|> transcripts -> chat_mode=True
        Same code, same architecture. Only the data changes.
        """
        if not self.chat_mode:
            return user_text

        parts = []
        for u, a in self.history[-self.max_history:]:
            parts.append(f"<|user|>{u}<|assistant|>{a}<|eos|>")
        parts.append(f"<|user|>{user_text}<|assistant|>")
        return "".join(parts)

    # ---------------------------------------------------------------- response

    def respond(self, user_text: str, stream=True) -> str:
        skill = self.registry.find_skill(user_text)
        if skill is not None:
            reply = skill.handle(user_text)
            self.history.append((user_text, reply))
            return reply

        if self.model is None:
            return "I have no model loaded, so I can only handle skills right now."

        prompt = self.build_prompt(user_text)
        ids = prepare_prompt(self.tok, prompt, warn=False)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        pieces, buf = [], []

        def on_token(t):
            buf.append(t)
            text = self.tok.decode(buf)
            if "\ufffd" not in text:      # wait for complete UTF-8 sequences
                pieces.append(text)
                buf.clear()
                if stream:
                    for o in self.registry.outputs:
                        o.emit_stream(text)

        out = generate(self.model, x, eos_id=self.tok.eos_id(),
                       stream_callback=on_token, **self.gen_kwargs)
        if buf:
            pieces.append(self.tok.decode(buf))

        reply = "".join(pieces).strip()
        if stream:
            for o in self.registry.outputs:
                o.flush()
        self.history.append((user_text, reply))
        return reply

    # -------------------------------------------------------------------- loop

    def _set_output_lang(self, lang: str | None) -> None:
        for o in self.registry.outputs:
            if hasattr(o, "current_lang"):
                o.current_lang = lang

    def run(self):
        if not self.registry.inputs:
            raise RuntimeError("no input module registered")
        source = self.registry.inputs[0]
        print("\n" + "=" * 60)
        print("JARVIS — modules:")
        print(self.registry.summary())
        print("=" * 60)

        # Once, at session start: if the first thing heard matches the
        # enrolled voiceprint, offer the admin login. Silent no-op if no mic,
        # nothing enrolled, or the voice doesn't match — see
        # SecurityGate.wake_challenge's docstring for why that matters.
        if self.security.wake_challenge(self.admin_name, self.affirmative_phrases):
            self.is_admin = True
            self.security._say("Administrator access granted.")

        try:
            while True:
                text = source.listen()
                if text is None:
                    break
                if not text.strip():
                    continue

                # Whichever language Whisper detected this utterance in (None
                # from ConsoleInput, which has no such thing -> defaults to
                # English). The reply speaks back in the same language.
                lang = getattr(source, "last_language", None) or "en"
                self._set_output_lang(lang)
                translate = lang != "en" and self.translator is not None

                if _is_shutdown_request(text, self.shutdown_phrases):
                    # Already proved identity this session -> don't ask again.
                    if self.is_admin or self.security.authorize("shut down JARVIS"):
                        self.registry.emit_all("Shutting down.")
                        break
                    self.registry.emit_all("Verification failed — staying online.")
                    continue

                skill = self.registry.find_skill(text)
                if skill is not None:
                    reply = skill.handle(text)
                    self.history.append((text, reply))
                    if translate:
                        reply = self.translator.translate(reply, "en", lang)
                    self.registry.emit_all(reply)
                else:
                    if self.model is None:
                        self.registry.emit_all(
                            "No model loaded — train one first.")
                        continue
                    if translate:
                        # Translation needs the complete English reply before
                        # it can run, so this path can't stream token-by-token
                        # to the speaker the way the English path does.
                        english_in = self.translator.translate(text, lang, "en")
                        reply = self.respond(english_in, stream=False)
                        self.registry.emit_all(self.translator.translate(reply, "en", lang))
                    else:
                        print("jarvis> ", end="", flush=True)
                        self.respond(text, stream=True)
        finally:
            self.registry.teardown_all()
            print("\n[jarvis] shutdown")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_PKG_DIR, "checkpoints", "best.pt"))
    ap.add_argument("--tokenizer", default=os.path.join(_PKG_DIR, "data", "tokenizer.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--chat-mode", action="store_true",
                    help="wrap turns in <|user|>/<|assistant|> — only after fine-tuning")
    ap.add_argument("--voice", action="store_true",
                    help="microphone input + spoken output, fully local "
                         "(faster-whisper STT, Kokoro/Piper TTS) — no API key, no cost")
    ap.add_argument("--whisper-model", default=None,
                    help="faster-whisper model size — multilingual unless "
                         "suffixed .en (tiny/base/small/medium/large-v3, or "
                         "tiny.en/base.en/... for English-only + faster). "
                         "Default: auto-picked from detected RAM/VRAM, see modules/hardware.py")
    ap.add_argument("--persona", default="jarvis", choices=["jarvis", "eve"],
                    help="voice identity: jarvis (male) or eve (female)")
    ap.add_argument("--lang", default="en",
                    help="default spoken language (en/ja/es/fr/ru/ko/zh/hi/pt/ar/bn) — "
                         "used until Whisper detects something else in what you actually say")
    ap.add_argument("--admin-name", default="Abhishek",
                    help="name JARVIS uses in the wake-time \"are you the admin, X?\" challenge")
    ap.add_argument("--whisper-device", default=None,
                    help="cuda or cpu for faster-whisper — falls back to cpu automatically "
                         "if cuda isn't actually available. Default: auto-picked, see modules/hardware.py")
    ap.add_argument("--reasoning-model", default=None,
                    help="Ollama model for real conversation/knowledge. "
                         "Default: auto-picked biggest-that-fits from detected RAM/VRAM, "
                         "see modules/hardware.py's recommend_reasoning_model(). "
                         "Skipped automatically if Ollama isn't running.")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="disable the Ollama reasoning module — fall back to skills + the raw core model")
    ap.add_argument("--no-os-control", action="store_true",
                    help="disable OS-agentic control (open/close/list applications) — "
                         "every state-changing action is security-gated regardless, "
                         "this just removes the capability entirely")
    ap.add_argument("--no-hardware", action="store_true",
                    help="disable the serial/Arduino hardware skill (list/connect/send/read) — "
                         "connecting and sending are security-gated regardless; this just "
                         "removes the capability entirely. Silently unavailable anyway if "
                         "pyserial isn't installed.")
    ap.add_argument("--hardware-baud", type=int, default=115200,
                    help="baud rate for serial connections opened via the hardware skill")
    ap.add_argument("--no-mcp", action="store_true",
                    help="disable the MCP client (connects to servers listed in "
                         "data/mcp_servers.json, lets the reasoning model call their tools) — "
                         "tool calls are security-gated regardless; this just removes the "
                         "capability entirely")
    ap.add_argument("--mcp-config", default=os.path.join(_PKG_DIR, "data", "mcp_servers.json"),
                    help="path to the MCP server config (see modules/mcp_client.py for the shape)")
    ap.add_argument("--no-market-analysis", action="store_true",
                    help="disable the backtesting skill (\"backtest AAPL\", \"analyze crypto BTC-USD\") — "
                         "read-only historical analysis, no live trading account involved")
    args = ap.parse_args()

    j = Jarvis(ckpt=args.ckpt, tokenizer=args.tokenizer, device=args.device,
               chat_mode=args.chat_mode, admin_name=args.admin_name)
    j.gen_kwargs["temperature"] = args.temperature
    j.load_model()

    if args.voice:
        from jarvis.modules.voice import WhisperSTTEngine, PersonaTTSEngine
        whisper_model, whisper_device = args.whisper_model, args.whisper_device
        if whisper_model is None or whisper_device is None:
            from jarvis.modules import hardware
            auto_model, auto_device = hardware.recommend_whisper()
            whisper_model = whisper_model or auto_model
            whisper_device = whisper_device or auto_device
            print(f"[jarvis] auto-sized whisper for this machine: "
                  f"model={whisper_model} device={whisper_device}")
        print(f"[jarvis] loading voice engines (whisper={whisper_model}/{whisper_device}, "
              f"persona={args.persona}, default lang={args.lang})...")
        try:
            whisper_engine = WhisperSTTEngine(model_size=whisper_model,
                                               device=whisper_device,
                                               compute_type="float16" if whisper_device == "cuda" else "int8")
        except Exception as e:
            print(f"[jarvis] whisper on {whisper_device} failed ({e}), falling back to cpu")
            whisper_engine = WhisperSTTEngine(model_size=whisper_model, device="cpu")
        persona_engine = PersonaTTSEngine(persona=args.persona, default_lang=args.lang)
        j.register(SpeechInput(engine=whisper_engine))
        j.register(SpeechOutput(engine=persona_engine))
        # mic+tts here mean authorize() prompts by voice and listens for the
        # passphrase instead of calling getpass() — required since a
        # background launch (see launch_jarvis.vbs) has no stdin to read.
        j.security = SecurityGate(mic_engine=whisper_engine, tts_engine=persona_engine)
        try:
            from jarvis.modules.translate import Translator
            j.translator = Translator()
            j.refresh_multilingual_phrases()
        except Exception as e:
            print(f"[jarvis] translation unavailable ({e}) — non-English speech "
                  f"will be transcribed but not bridged to the (English-only) model")
    else:
        j.register(ConsoleInput())
    j.register(ConsoleOutput())
    j.register(CalculatorSkill())
    j.register(WebSearchSkill())
    j.register(WebGrowthSkill(data_dir=os.path.join(_PKG_DIR, "data", "web")))
    if not args.no_os_control:
        j.register(OSControlSkill(security_ref=lambda: j.security, is_admin_ref=lambda: j.is_admin))
    if not args.no_hardware:
        j.register(HardwareSkill(security_ref=lambda: j.security, is_admin_ref=lambda: j.is_admin,
                                  baud=args.hardware_baud))
    if not args.no_market_analysis:
        j.register(MarketAnalysisSkill())
    mcp_skill = None
    if not args.no_mcp:
        mcp_skill = MCPSkill(config_path=args.mcp_config,
                              security_ref=lambda: j.security, is_admin_ref=lambda: j.is_admin)
        j.register(mcp_skill)
    if not args.no_reasoning:
        reasoning_model = args.reasoning_model
        if reasoning_model is None:
            from jarvis.modules import hardware
            reasoning_model = hardware.recommend_reasoning_model()
            print(f"[jarvis] auto-sized reasoning model for this machine: {reasoning_model}")
        j.register(ReasoningSkill(model=reasoning_model, history_ref=lambda: j.history,
                                   mcp_ref=(lambda: mcp_skill) if mcp_skill is not None else None))
    j.run()


if __name__ == "__main__":
    main()
