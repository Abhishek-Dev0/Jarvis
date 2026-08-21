"""
jarvis.py — the orchestrator.

Owns the loop: get input -> check skills -> otherwise ask the model -> emit output.
Also owns conversation state and prompt formatting.

Deliberately thin. All the intelligence is in core/, all the capability is in
modules/. This file is just the wiring, which means you can rewrite it without
touching anything that matters.
"""

"""Compatibility wrapper for the package entry point."""

from jarvis.runtime.jarvis import main


if __name__ == "__main__":
    main()
