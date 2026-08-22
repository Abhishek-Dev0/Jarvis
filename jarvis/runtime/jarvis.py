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
import time
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
    from jarvis.modules.health import HealthCheckSkill
    from jarvis.modules.memory import MemorySkill
    from jarvis.security import SecurityGate
    from jarvis import security
    from jarvis import self_modify
    from jarvis import telemetry
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
    from modules.health import HealthCheckSkill
    from modules.memory import MemorySkill
    from security import SecurityGate
    import security
    import self_modify
    import telemetry

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
# "I'm the admin" (mid-session, explicit) vs. wake_challenge's automatic
# first-utterance voice match — Abi asked for both paths on 2026-08-22:
# announce yourself any time, not just at session start. Apostrophe is
# stripped by _normalize_phrase, hence "im"/"i am" both listed.
_ADMIN_TRIGGER_PHRASES_EN = {"im the admin", "i am the admin"}
# "recognize me" — the face-only admin path, for not saying the passphrase
# out loud in public. See security.SecurityGate.face_login's docstring for
# why this is deliberately weaker (single-factor) than the passphrase path.
_FACE_LOGIN_PHRASES_EN = {"recognize me", "check my face", "face login", "admin face login"}
# Re-enrolling the passphrase requires PROVING you're already admin first
# (current session admin state, or the OLD passphrase via authorize()) —
# same reason every "change password" flow anywhere requires the current
# password: without that, anyone could say this phrase and lock the real
# admin out. Not an extra restriction Abi didn't ask for — it's what makes
# "I can just say change the phrase and it registers again" (his own
# 2026-08-22 request) safe to build at all.
_CHANGE_PASSPHRASE_PHRASES_EN = {"change the phrase", "change the passphrase",
                                  "change my passphrase", "reset the passphrase"}
# Cosmetic persona switch (voice + mascot) — ungated on purpose, anyone can
# ask for it, same as picking a UI theme. Not a security-relevant action.
_SWITCH_TO_EVE_PHRASES_EN = {"switch to eve", "become eve", "use eve", "eve persona"}
_SWITCH_TO_JARVIS_PHRASES_EN = {"switch to jarvis", "become jarvis", "use jarvis", "jarvis persona"}


def _normalize_phrase(text: str) -> str:
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _is_shutdown_request(text: str, phrases: set[str]) -> bool:
    return _normalize_phrase(text) in phrases


# Lines matched here get redacted before ever reaching the session log file
# — never the real console, which is local and already fine. Abi asked for
# this session-log capability specifically so he can copy/paste it into a
# chat with Claude for diagnosis (2026-08-22); a file meant to leave the
# machine that way must not carry a live credential. Currently one known
# pattern: security.enroll()'s voice-enrollment echo of the just-heard
# passphrase (useful on the real console — confirms what was actually
# transcribed — but exactly the kind of line that must never leave it).
# Any future code that prints something secret needs its own entry here.
import re as _re
_LOG_REDACTIONS = [
    (_re.compile(r'(\[security\] heard: ").*(" — say this back to verify later\.)'),
     r'\1[REDACTED]\2'),
]


class _TeeLog:
    """Duplicates every write to both the real console/stream and a session
    log file, applying _LOG_REDACTIONS to the file copy only. Forwards
    unknown attributes (reconfigure(), isatty(), etc.) to the wrapped
    stream so this is a transparent drop-in for sys.stdout/sys.stderr."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        redacted = data
        for pattern, repl in _LOG_REDACTIONS:
            redacted = pattern.sub(repl, redacted)
        try:
            self._log_file.write(redacted)
            self._log_file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _start_session_log() -> str:
    """Tees stdout/stderr into a timestamped file under data/logs/ for the
    rest of this process's life, so a full session transcript — not just
    caught exceptions (see self_modify.py's separate issues.jsonl) — is
    always available to copy/paste, without needing a screenshot.

    Pair with _stop_session_log() before the process exits — without it,
    Python's interpreter-shutdown sequence tries to flush/close sys.stdout
    itself, but by then the original real stream underneath this wrapper
    may already be torn down, which surfaced as a harmless but ugly
    "Exception ignored in: <_TeeLog ...>" / "Exception ignored in
    sys.unraisablehook" on every clean exit — found by actually running a
    full session end-to-end, not assumed away. Restoring the real streams
    and closing the log file explicitly, before that shutdown sequence
    ever runs, avoids it entirely.
    """
    import datetime
    logs_dir = os.path.join(_PKG_DIR, "data", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"session_{ts}.log")
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")
    sys.stdout = _TeeLog(sys.stdout, log_file)
    sys.stderr = _TeeLog(sys.stderr, log_file)
    return log_path


def _stop_session_log() -> None:
    if isinstance(sys.stdout, _TeeLog):
        real_stdout, log_file = sys.stdout._stream, sys.stdout._log_file
        sys.stdout = real_stdout
    else:
        log_file = None
    if isinstance(sys.stderr, _TeeLog):
        sys.stderr = sys.stderr._stream
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass

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
        self.admin_trigger_phrases = set(_ADMIN_TRIGGER_PHRASES_EN)
        self.face_login_phrases = set(_FACE_LOGIN_PHRASES_EN)
        self.change_passphrase_phrases = set(_CHANGE_PASSPHRASE_PHRASES_EN)
        self.switch_eve_phrases = set(_SWITCH_TO_EVE_PHRASES_EN)
        self.switch_jarvis_phrases = set(_SWITCH_TO_JARVIS_PHRASES_EN)
        # None unless main() wires one up (modules/mascot.py) — a cosmetic
        # terminal companion, not a capability. Callers check for None
        # before using it (see run()).
        self.mascot = None

    def refresh_multilingual_phrases(self) -> None:
        """Extends shutdown/affirmative/admin-trigger phrase matching into
        every supported language via self.translator. Safe to call with no
        translator (no-op) or call again later if the translator becomes
        available."""
        if self.translator is None:
            return
        try:
            from jarvis.modules.voice import SUPPORTED_LANGUAGES
        except ImportError:  # pragma: no cover - legacy direct execution
            from modules.voice import SUPPORTED_LANGUAGES
        phrase_sets = (
            (_SHUTDOWN_PHRASES_EN, self.shutdown_phrases),
            (_AFFIRMATIVE_PHRASES_EN, self.affirmative_phrases),
            (_ADMIN_TRIGGER_PHRASES_EN, self.admin_trigger_phrases),
            (_FACE_LOGIN_PHRASES_EN, self.face_login_phrases),
            (_CHANGE_PASSPHRASE_PHRASES_EN, self.change_passphrase_phrases),
            (_SWITCH_TO_EVE_PHRASES_EN, self.switch_eve_phrases),
            (_SWITCH_TO_JARVIS_PHRASES_EN, self.switch_jarvis_phrases),
        )
        for lang in SUPPORTED_LANGUAGES:
            if lang == "en":
                continue
            for source_phrases, target_set in phrase_sets:
                for phrase in source_phrases:
                    translated = self.translator.translate(phrase, "en", lang)
                    if translated:
                        target_set.add(_normalize_phrase(translated))

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

    def _yes_no(self, prompt: str) -> bool:
        """Asks a yes/no question through whatever I/O this session is
        already using (mic if --voice, console input() otherwise) and
        returns the answer. Used by _offer_enrollment/_handle_change_
        passphrase so setup never needs a separate command — it's just
        part of the running conversation."""
        self.registry.emit_all(prompt)
        voice_mode = self.security.mic_engine is not None and hasattr(self.security.mic_engine, "transcribe")
        if voice_mode:
            audio = self.security.mic_engine.record_until_silence()
            reply = self.security.mic_engine.transcribe(audio) or ""
        else:
            try:
                reply = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                reply = ""
        return _normalize_phrase(reply) in self.affirmative_phrases

    def _enroll_passphrase_live(self) -> None:
        """Captures a NEW passphrase through this session's own I/O (mic if
        --voice, hidden getpass() otherwise) and enrolls it — the secret
        only ever passes between the person at the keyboard/mic and this
        running process, never through anything else. Preserves any
        existing voiceprint/face enrollment (security.enroll() only
        overwrites what you pass it)."""
        voice_mode = self.security.mic_engine is not None and hasattr(self.security.mic_engine, "transcribe")
        if voice_mode:
            self.registry.emit_all("Speak your new passphrase now, then go quiet.")
            audio = self.security.mic_engine.record_until_silence()
            security.enroll(voice_audio=audio, transcribe_fn=self.security.mic_engine.transcribe)
        else:
            import getpass
            self.registry.emit_all("Setting a new passphrase — type it now (hidden).")
            passphrase = getpass.getpass("New passphrase: ")
            security.enroll(passphrase=passphrase)

    def _offer_enrollment(self) -> None:
        """Called once at the start of run() if nothing is enrolled yet.
        First-run setup done entirely within this running process — no
        separate `python -m jarvis.security enroll` command to remember.
        Declining is fine; every gated action just keeps denying (same as
        always) until this or the CLI enrollment is run."""
        if security.is_enrolled():
            return
        if not self._yes_no("No admin credentials are set up yet. Would you like to set one up now?"):
            self.registry.emit_all("Okay — say \"change the phrase\" any time to set one up later.")
            return
        self._enroll_passphrase_live()
        self.registry.emit_all("Passphrase set.")

        if self._yes_no("Would you also like to enroll face recognition, for a "
                         "passphrase-free admin check later?"):
            try:
                self.registry.emit_all("Look at the camera now...")
                security.enroll_face()
                self.registry.emit_all("Face enrolled.")
            except Exception as e:
                self_modify.log_exception("enroll_face", e)
                self.registry.emit_all(
                    f"Couldn't enroll a face right now ({e}). You can try again later "
                    f"with: python -m jarvis.security enroll --face")

    def _handle_change_passphrase(self) -> None:
        """"Change the phrase" — requires PROVING you're already admin
        first (current session state, or the OLD passphrase), same reason
        every "change password" flow anywhere requires the current
        password. Without that this would just be a way for anyone to
        lock the real admin out by saying five words."""
        if not (self.is_admin or self.security.authorize("change the passphrase")):
            self.registry.emit_all("Admin denied. You're a general user.")
            return
        self.is_admin = True
        self._enroll_passphrase_live()
        self.registry.emit_all("Passphrase changed.")

    def _switch_persona(self, persona: str) -> None:
        """Cosmetic — voice identity + mascot appearance, not gated (see
        the trigger phrase constants' comment for why). Updates whichever
        of the two is actually present; either can be missing (console
        mode has no TTS engine, --no-mascot or a headless run has no
        mascot) without the other failing."""
        for output in self.registry.outputs:
            engine = getattr(output, "engine", None)
            if engine is not None and hasattr(engine, "persona"):
                engine.persona = persona
        if self.mascot is not None:
            self.mascot.switch(persona)
        self.registry.emit_all(f"Switched to {persona}.")

    def run(self):
        if not self.registry.inputs:
            raise RuntimeError("no input module registered")
        source = self.registry.inputs[0]
        print("\n" + "=" * 60)
        print("JARVIS — modules:")
        print(self.registry.summary())
        print("=" * 60)

        # The "systems online" moment Abi asked for on 2026-08-22 — printed
        # either way (works in console mode, no TTS needed), spoken too
        # when a speech output is registered. Deliberately just a greeting,
        # not a security check: admin recognition (voice wake_challenge,
        # "I'm the admin", "recognize me") all still happen through the
        # normal gated paths below/in the main loop, nothing here bypasses
        # any of that.
        self.registry.emit_all("Systems online. How can I help you, sir?")

        # First-run setup, if nothing is enrolled yet — done live, through
        # this same session's own I/O, not a separate command. See
        # _offer_enrollment's docstring for why the secret never passes
        # through anything but this running process.
        try:
            self._offer_enrollment()
        except Exception as e:
            self_modify.log_exception("offer_enrollment", e)
            print(f"[jarvis] enrollment setup failed ({e}) — try again later with: "
                  f"python -m jarvis.security enroll")

        # Once, at session start: if the first thing heard matches the
        # enrolled voiceprint, offer the admin login. Silent no-op if no mic,
        # nothing enrolled, or the voice doesn't match — see
        # SecurityGate.wake_challenge's docstring for why that matters.
        # Wrapped: a transcription-engine crash here (e.g. the cuBLAS/CUDA
        # DLL mismatch this exact block surfaced during testing — see
        # modules/voice.py's _register_cuda_dll_dirs()) used to kill the
        # process before the main loop's own try/finally even started,
        # skipping teardown_all() entirely.
        try:
            if self.security.wake_challenge(self.admin_name, self.affirmative_phrases):
                self.is_admin = True
                self.security._say("Administrator access granted.")
        except Exception as e:
            self_modify.log_exception("wake_challenge", e)
            print(f"[jarvis] wake challenge failed ({e}) — continuing as a regular session")

        try:
            while True:
                try:
                    text = source.listen()
                except Exception as e:
                    self_modify.log_exception("source.listen", e)
                    print(f"[jarvis] input error ({e}) — retrying")
                    # A safety floor, not a real backoff: if listen() fails
                    # before it ever blocks on real I/O (e.g. a mic that's
                    # gone missing entirely, instead of the one-off cuBLAS
                    # crash this was written for), this stops the retry
                    # loop from spinning the CPU / flooding the issue log.
                    time.sleep(0.5)
                    continue
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

                if _normalize_phrase(text) in self.admin_trigger_phrases:
                    if self.is_admin:
                        self.registry.emit_all("Already recognized as admin.")
                    else:
                        # authorize() itself announces the outcome — "Admin
                        # control initiated. Please say the passphrase.",
                        # then either "Full capabilities online." or
                        # "Admin denied. You're a general user." — including
                        # one retry on a wrong/misheard attempt before it
                        # gives up. Nothing else to say here.
                        self.is_admin = self.security.authorize("become admin")
                    continue

                if _normalize_phrase(text) in self.face_login_phrases:
                    if self.is_admin:
                        self.registry.emit_all("Already recognized as admin.")
                    elif self.security.face_login():
                        self.is_admin = True
                        self.registry.emit_all("Admin recognized. Initializing full capabilities.")
                    else:
                        self.registry.emit_all("Face not recognized. Admin denied. You're a general user.")
                    continue

                if _normalize_phrase(text) in self.change_passphrase_phrases:
                    self._handle_change_passphrase()
                    continue

                if _normalize_phrase(text) in self.switch_eve_phrases:
                    self._switch_persona("eve")
                    continue
                if _normalize_phrase(text) in self.switch_jarvis_phrases:
                    self._switch_persona("jarvis")
                    continue

                skill = self.registry.find_skill(text)
                component = skill.name if skill is not None else "reasoning-or-core"
                turn_start = time.monotonic()
                try:
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
                    telemetry.log_event(component, "handle_turn",
                                        duration_ms=(time.monotonic() - turn_start) * 1000,
                                        outcome="ok")
                except Exception as e:
                    # A skill or the model raising used to crash the whole
                    # process (find_skill() already guards matches(), but
                    # nothing guarded handle()/respond()) — caught here so
                    # one bad turn doesn't end the session, and logged so
                    # self_modify.py's autonomous scanner has something real
                    # to work from.
                    self_modify.log_exception(f"turn:{component}", e)
                    telemetry.log_event(component, "handle_turn", severity="error",
                                        duration_ms=(time.monotonic() - turn_start) * 1000,
                                        outcome="error", error=str(e))
                    print(f"[jarvis] error handling that turn: {e}")
                    self.registry.emit_all("Something went wrong handling that — see the log.")

                if self.mascot is not None:
                    self.mascot.idle_flourish()
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
    ap.add_argument("--no-self-modify", action="store_true",
                    help="disable the self-modify skill (\"propose fix <path>: <problem>\", "
                         "\"approve proposal <id>\") — drafts+sandbox-tests only, applying a "
                         "proposal is security-gated regardless; this just removes the capability")
    ap.add_argument("--self-modify-autoscan", action="store_true",
                    help="run the autonomous issue scanner in the background — periodically drafts "
                         "and sandbox-tests proposals for logged issues, unattended. Never applies "
                         "anything on its own. Off by default, same as --voice/--chat-mode: a "
                         "continuously-running background capability should be opted into, not "
                         "silently on.")
    ap.add_argument("--self-modify-scan-interval", type=int, default=1800,
                    help="seconds between autonomous scan cycles (default 1800 = 30 min)")
    ap.add_argument("--no-mascot", action="store_true",
                    help="disable the animated terminal cat (modules/mascot.py) — purely "
                         "cosmetic, no effect on any actual capability")
    ap.add_argument("--no-session-log", action="store_true",
                    help="disable writing a full session transcript to data/logs/session_*.log "
                         "(on by default — hand that file to Claude to diagnose an issue instead "
                         "of a screenshot; known secret-revealing lines are redacted before they "
                         "reach the file, see _LOG_REDACTIONS)")
    args = ap.parse_args()

    if not args.no_session_log:
        log_path = _start_session_log()
        print(f"[jarvis] session log: {log_path}")

    print("Analyzing system specifications...")
    from jarvis.modules import hardware
    _profile = hardware.detect()
    gpu_desc = f"{_profile['gpu_name']} ({_profile['vram_gb']} GB VRAM)" if _profile["has_cuda"] else "no CUDA GPU"
    print(f"  CPU: {_profile['cpu_cores']} cores / {_profile['cpu_threads']} threads   "
          f"RAM: {_profile['ram_gb']} GB   GPU: {gpu_desc}")

    j = Jarvis(ckpt=args.ckpt, tokenizer=args.tokenizer, device=args.device,
               chat_mode=args.chat_mode, admin_name=args.admin_name)
    j.gen_kwargs["temperature"] = args.temperature
    j.load_model()

    if not args.no_mascot:
        from jarvis.modules.mascot import CatMascot
        j.mascot = CatMascot(persona=args.persona)

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
        j.register(SpeechOutput(engine=persona_engine, mascot=j.mascot))
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
    j.register(HealthCheckSkill(jarvis_ref=lambda: j,
                                 mcp_ref=(lambda: mcp_skill) if mcp_skill is not None else None))
    j.register(MemorySkill())
    reasoning_model = args.reasoning_model
    if reasoning_model is None and (not args.no_reasoning or not args.no_self_modify):
        from jarvis.modules import hardware
        reasoning_model = hardware.recommend_reasoning_model()
        print(f"[jarvis] auto-sized reasoning model for this machine: {reasoning_model}")
    if not args.no_reasoning:
        j.register(ReasoningSkill(model=reasoning_model, history_ref=lambda: j.history,
                                   mcp_ref=(lambda: mcp_skill) if mcp_skill is not None else None))

    scanner = None
    if not args.no_self_modify:
        j.register(self_modify.SelfModifySkill(
            security_ref=lambda: j.security, is_admin_ref=lambda: j.is_admin, model=reasoning_model))
        if args.self_modify_autoscan:
            scanner = self_modify.AutonomousScanner(
                interval_seconds=args.self_modify_scan_interval, model=reasoning_model)
            scanner.start()
            print(f"[jarvis] self-modify autonomous scanner running every "
                  f"{args.self_modify_scan_interval}s — drafts+tests only, never applies")

    try:
        j.run()
    finally:
        if scanner is not None:
            scanner.stop()
        if not args.no_session_log:
            _stop_session_log()


if __name__ == "__main__":
    main()
