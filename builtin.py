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

from .base import InputModule, OutputModule, SkillModule


# --------------------------------------------------------------------- console

class ConsoleInput(InputModule):
    name = "console_in"
    description = "reads typed input"

    def listen(self):
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return None if text.lower() in {"quit", "exit"} else text


class ConsoleOutput(OutputModule):
    name = "console_out"
    description = "prints to terminal"

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


# ------------------------------------------------------------------ voice stubs

class SpeechInput(InputModule):
    """STUB — plug your speech recogniser in here.

    The interface is already right: listen() blocks, returns a string or None.
    Nothing else in the system needs to know what engine you used.
    """

    name = "speech_in"
    description = "speech to text (not yet implemented)"

    def __init__(self, engine=None):
        self.engine = engine

    @property
    def available(self):
        return self.engine is not None

    def setup(self):
        pass

    def listen(self):
        audio = self.engine.record_until_silence()   # your recorder
        return self.engine.transcribe(audio)         # your recogniser


class SpeechOutput(OutputModule):
    """STUB — plug your synthesiser in here.

    If you want to build the voice from scratch too, that's a second project of
    the same shape as this one: a vocoder plus an acoustic model. Worth doing
    AFTER the language model works — otherwise you're debugging two unknowns.
    """

    name = "speech_out"
    description = "text to speech (not yet implemented)"

    def __init__(self, engine=None):
        self.engine = engine

    @property
    def available(self):
        return self.engine is not None

    def emit(self, text):
        self.engine.speak(text)
