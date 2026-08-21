"""
base.py — the contract every add-on module obeys.

This is the boundary you asked for. The core (tokenizer + model + training) knows
nothing about voice, tools, or anything else. Modules attach from the outside.

Three kinds of module, distinguished by which hooks they implement:

  INPUT   turns something into text        (speech recognition, OCR, file reader)
  OUTPUT  turns text into something        (speech synthesis, display, robot command)
  SKILL   intercepts a request and handles it directly, bypassing the model
          (calculator, timer, system control — things a small model should not
           be guessing at)

The point of SKILL modules: your model will be small for a long time. Don't ask a
25M-parameter network to do arithmetic. Route it.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    name: str = "unnamed"
    description: str = ""

    def setup(self) -> None:
        """Called once at startup. Load files, open devices, warm caches."""
        pass

    def teardown(self) -> None:
        """Called on shutdown. Release devices, close handles."""
        pass

    @property
    def available(self) -> bool:
        """Return False if this module's dependencies aren't present.
        The runtime skips unavailable modules instead of crashing."""
        return True


class InputModule(Module):
    @abstractmethod
    def listen(self) -> str | None:
        """Block until input is available; return it as text, or None to stop."""
        ...


class OutputModule(Module):
    @abstractmethod
    def emit(self, text: str) -> None:
        ...

    def emit_stream(self, text_chunk: str) -> None:
        """Optional: handle partial output as it's generated. Defaults to buffering."""
        pass

    def flush(self) -> None:
        pass


class SkillModule(Module):
    """A capability that handles certain requests without the model."""

    priority: int = 0   # higher wins when several skills match

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Cheap check: should this skill handle the request?"""
        ...

    @abstractmethod
    def handle(self, text: str) -> str:
        """Produce the response."""
        ...


class Registry:
    """Holds the modules and finds the right one for a request."""

    def __init__(self):
        self.inputs: list[InputModule] = []
        self.outputs: list[OutputModule] = []
        self.skills: list[SkillModule] = []

    def register(self, module: Module) -> bool:
        if not module.available:
            print(f"[registry] skipping '{module.name}' — dependencies unavailable")
            return False
        module.setup()
        if isinstance(module, InputModule):
            self.inputs.append(module)
        elif isinstance(module, OutputModule):
            self.outputs.append(module)
        elif isinstance(module, SkillModule):
            self.skills.append(module)
            self.skills.sort(key=lambda s: -s.priority)
        else:
            raise TypeError(f"{module.name} is not an Input/Output/Skill module")
        print(f"[registry] loaded '{module.name}'")
        return True

    def find_skill(self, text: str) -> SkillModule | None:
        for s in self.skills:
            try:
                if s.matches(text):
                    return s
            except Exception as e:
                print(f"[registry] skill '{s.name}' errored while matching: {e}")
        return None

    def emit_all(self, text: str) -> None:
        for o in self.outputs:
            try:
                o.emit(text)
            except Exception as e:
                print(f"[registry] output '{o.name}' failed: {e}")

    def teardown_all(self) -> None:
        for m in [*self.inputs, *self.outputs, *self.skills]:
            try:
                m.teardown()
            except Exception:
                pass

    def summary(self) -> str:
        lines = []
        for label, mods in (("input", self.inputs), ("output", self.outputs), ("skill", self.skills)):
            for m in mods:
                lines.append(f"  {label:6s}  {m.name:16s}  {m.description}")
        return "\n".join(lines) or "  (no modules loaded)"
