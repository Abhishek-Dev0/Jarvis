"""PyInstaller entry point for the JARVIS desktop app. Kept as a real,
minimal script (not a -m invocation) because that's what PyInstaller
expects to analyze. When run from source (not frozen), adds the repo root
to sys.path so `import jarvis` resolves the same way it does for everyone
else running this project from a checkout."""

import os
import sys

if not getattr(sys, "frozen", False):
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO_ROOT)

from jarvis.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
