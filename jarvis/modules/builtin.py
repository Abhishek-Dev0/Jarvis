"""
builtin.py — working modules plus the stubs you'll fill in.

ConsoleInput/ConsoleOutput and CalculatorSkill are complete and functional.
SpeechInput/SpeechOutput are deliberately left as stubs with the interface
already correct — drop your engine in, register the module, done. Nothing in
core/ changes.
"""

from __future__ import annotations
import ast
import operator
import re
import sys

from .base import InputModule, OutputModule, SkillModule


# --------------------------------------------------------------------- console

class ConsoleInput(InputModule):
    name = "console_in"
    description = "reads typed input"

    def listen(self):
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            # A hard OS-level interrupt (Ctrl+D/Ctrl+C) stops the process
            # immediately, same as closing the window — no software gate can
            # meaningfully stand in front of that, so this bypasses
            # verification by design. Typed/spoken "quit"/"shutdown" do not:
            # they flow through as plain text and are gated in Jarvis.run().
            return None
        return text


class ConsoleOutput(OutputModule):
    name = "console_out"
    description = "prints to terminal"

    def setup(self):
        # Windows consoles default to a legacy codepage (cp1252/cp932) that
        # can't encode most Unicode. Search results, Japanese text, even
        # curly quotes from a Wikipedia snippet will UnicodeEncodeError and
        # kill the output module mid-conversation. Force UTF-8 on stdout;
        # replace anything the terminal still can't render rather than crash.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def emit(self, text):
        print(f"jarvis> {text}")

    def emit_stream(self, chunk):
        print(chunk, end="", flush=True)

    def flush(self):
        print()


# ------------------------------------------------------------------ calculator

class CalculatorSkill(SkillModule):
    """Arithmetic, evaluated safely via AST walking rather than eval().

    A 25M-parameter model cannot reliably do arithmetic. Don't make it try.
    """

    name = "calculator"
    description = "evaluates arithmetic expressions"
    priority = 10

    _OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _PATTERN = re.compile(r"^[\s\d\.\+\-\*/%\(\)\^]+$")

    def matches(self, text):
        t = text.strip().lower()
        for prefix in ("calculate", "calc", "what is", "what's", "compute"):
            if t.startswith(prefix):
                t = t[len(prefix):].strip()
                break
        t = t.rstrip("?").replace("^", "**").strip()
        return bool(t) and bool(self._PATTERN.match(t.replace("**", "^"))) and any(c.isdigit() for c in t)

    def handle(self, text):
        t = text.strip().lower()
        for prefix in ("calculate", "calc", "what is", "what's", "compute"):
            if t.startswith(prefix):
                t = t[len(prefix):].strip()
                break
        t = t.rstrip("?").replace("^", "**").strip()
        try:
            result = self._eval(ast.parse(t, mode="eval").body)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except ZeroDivisionError:
            return "That's a division by zero."
        except Exception:
            return "I couldn't parse that expression."

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("non-numeric constant")
            return node.value
        if isinstance(node, ast.BinOp):
            op = self._OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported operator")
            return op(self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported unary operator")
            return op(self._eval(node.operand))
        raise ValueError("unsupported expression")


# ------------------------------------------------------------------------ voice

class SpeechInput(InputModule):
    """Speech-to-text input. Interface only — the actual recognizer is
    whatever `engine` is passed in (see modules/voice.py for the local
    faster-whisper engine). Nothing else in the system needs to know what
    engine you used.
    """

    name = "speech_in"
    description = "speech to text"

    def __init__(self, engine=None):
        self.engine = engine

    @property
    def available(self):
        return self.engine is not None

    def setup(self):
        pass

    def listen(self):
        audio = self.engine.record_until_silence()
        return self.engine.transcribe(audio)

    @property
    def last_language(self):
        """ISO 639-1 code Whisper detected for the most recent utterance, or
        None if the engine doesn't track that (e.g. an English-only model).
        Jarvis.run() reads this to pick the reply language and voice."""
        return getattr(self.engine, "last_language", None)


class SpeechOutput(OutputModule):
    """Text-to-speech output. Interface only — the actual synthesizer is
    whatever `engine` is passed in (see modules/voice.py for the local
    Kokoro engine).

    Skill replies go straight through emit(). Model-generated replies arrive
    token-by-token via emit_stream(); synthesizing per-token would mean
    per-token TTS calls and choppy audio, so those are buffered and spoken as
    one utterance in flush() once generation finishes.
    """

    name = "speech_out"
    description = "text to speech"

    def __init__(self, engine=None):
        self.engine = engine
        self._buf: list[str] = []
        # Jarvis.run() sets this before emitting a reply, from whatever
        # language Whisper detected in the user's speech (see SpeechInput
        # .last_language). Engines that only speak one language ignore it.
        self.current_lang: str | None = None

    @property
    def available(self):
        return self.engine is not None

    def emit(self, text):
        self.engine.speak(text, lang=self.current_lang)

    def emit_stream(self, chunk):
        self._buf.append(chunk)

    def flush(self):
        if self._buf:
            self.engine.speak("".join(self._buf), lang=self.current_lang)
            self._buf.clear()
