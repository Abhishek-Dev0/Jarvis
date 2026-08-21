"""
vendor_models.py — backs up every already-downloaded model file this
project depends on into one durable local archive, so re-setup on this
machine (or moving to a new one) never needs Hugging Face / GitHub / Argos's
index to be reachable again.

This is the "true offline-forever setup" backlog item. The gap it closes:
every voice/reasoning engine in this project downloads its weights *once*
and runs fully offline after — but until now nothing kept a second copy of
that one-time download anywhere durable, so a fresh machine still needed
that first fetch to succeed once (see the JARVIS roadmap notes, 2026-08-22).

What's vendored, each only if actually present on this machine:
  - Piper/Kokoro voices        jarvis/data/models/ — already inside this
                                repo, gitignored because of GitHub's 100MB
                                file limit (several hundred MB total)
  - faster-whisper's cache     Hugging Face hub cache (HF_HUB_CACHE)
  - Argos Translate packages   argostranslate.settings.package_data_dir
  - Ollama model blobs         $OLLAMA_MODELS or ~/.ollama/models

What's deliberately NOT vendored: jarvis/data/security/ (passphrase hash,
salt, voiceprint embedding). That's authentication material, not a model
weight — a "here's a copy of some downloaded files" backup is the wrong
place for it. Move that yourself, deliberately, if you're setting up a new
machine and want the same enrollment carried over.

Usage:
    python -m jarvis.vendor_models status              # what's present, what's not, sizes
    python -m jarvis.vendor_models backup [--dest DIR]  # copy present sources to DIR
    python -m jarvis.vendor_models restore [--src DIR]  # copy DIR's contents back into place

Default DIR is <repo root>/vendor/ — gitignored, colocated with the project
so it's easy to find, easy to copy to a USB drive or external disk yourself.
This tool only ever copies to/from that local directory; it doesn't upload
anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
_DEFAULT_VENDOR_DIR = os.path.join(_REPO_ROOT, "vendor")


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def sources() -> list[dict]:
    """Every known model-cache location, keyed by name. Each entry's `path`
    may or may not exist on this machine — that's checked by the caller,
    not baked in here, so listing a source never requires having it."""
    out = [{"name": "piper_kokoro_voices", "path": os.path.join(_PKG_DIR, "data", "models")}]

    try:
        from huggingface_hub import constants
        out.append({"name": "huggingface_hub_cache", "path": constants.HF_HUB_CACHE})
    except ImportError:
        pass

    try:
        import argostranslate.settings as argos_settings
        out.append({"name": "argos_translate_packages", "path": str(argos_settings.package_data_dir)})
    except ImportError:
        pass

    ollama_dir = os.environ.get("OLLAMA_MODELS") or os.path.join(os.path.expanduser("~"), ".ollama", "models")
    out.append({"name": "ollama_models", "path": ollama_dir})

    return out


def status() -> None:
    print(f"Vendor directory: {_DEFAULT_VENDOR_DIR}\n")
    for src in sources():
        present = os.path.isdir(src["path"])
        size = _human(_dir_size(src["path"])) if present else "-"
        flag = "present" if present else "not found"
        print(f"  [{flag:9s}] {src['name']:26s} {size:>10s}  {src['path']}")


def backup(dest: str | None = None) -> None:
    dest = dest or _DEFAULT_VENDOR_DIR
    os.makedirs(dest, exist_ok=True)
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "sources": []}

    for src in sources():
        if not os.path.isdir(src["path"]):
            print(f"[vendor] skipping '{src['name']}' — not found at {src['path']}")
            continue
        target = os.path.join(dest, src["name"])
        print(f"[vendor] backing up '{src['name']}' ({_human(_dir_size(src['path']))}) -> {target}")
        shutil.copytree(src["path"], target, dirs_exist_ok=True)
        manifest["sources"].append({"name": src["name"], "original_path": src["path"],
                                     "bytes": _dir_size(target)})

    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[vendor] done — {len(manifest['sources'])} source(s) backed up to {dest}")


def restore(src_dir: str | None = None) -> None:
    src_dir = src_dir or _DEFAULT_VENDOR_DIR
    manifest_path = os.path.join(src_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"no manifest.json in {src_dir} — run 'backup' from the "
                                 f"machine that has the models first, or pass --src")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    by_name = {s["name"]: s["path"] for s in sources()}
    for entry in manifest["sources"]:
        name, original_path = entry["name"], entry["original_path"]
        target = by_name.get(name, original_path)  # prefer this machine's own resolved
        # path (e.g. a different username) over the one recorded at backup time
        backed_up = os.path.join(src_dir, name)
        if not os.path.isdir(backed_up):
            print(f"[vendor] skipping '{name}' — missing from {src_dir}")
            continue
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        print(f"[vendor] restoring '{name}' -> {target}")
        shutil.copytree(backed_up, target, dirs_exist_ok=True)
    print("[vendor] restore complete")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Back up / restore JARVIS's downloaded model weights")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    backup_p = sub.add_parser("backup")
    backup_p.add_argument("--dest", default=None)
    restore_p = sub.add_parser("restore")
    restore_p.add_argument("--src", default=None)
    args = ap.parse_args()

    if args.cmd == "status":
        status()
    elif args.cmd == "backup":
        backup(args.dest)
    elif args.cmd == "restore":
        restore(args.src)


if __name__ == "__main__":
    main()
