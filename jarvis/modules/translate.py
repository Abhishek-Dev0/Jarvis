"""
translate.py — local, offline machine translation via Argos Translate.

Bridges JARVIS's English-only core model and skills to other languages: what
you say in language X gets translated to English before it reaches skills or
the model, and the English reply gets translated back to X before it's
spoken. See modules/voice.py for the language codes that have a voice to
speak the reply in — translation doesn't imply a voice exists, and vice
versa.

Argos downloads each language-pair package once (small — tens of MB, not
gigabytes) and then translates fully offline, no API, no per-call network
request. Install what you need up front:

    python -m jarvis.modules.translate ja es fr ru ko
"""

from __future__ import annotations


class Translator:
    def __init__(self):
        import argostranslate.translate as at
        self._at = at

    def translate(self, text: str, from_code: str, to_code: str) -> str:
        text = text.strip()
        if not text or from_code == to_code:
            return text
        try:
            return self._at.translate(text, from_code, to_code)
        except Exception as e:
            print(f"[translate] {from_code}->{to_code} failed ({e}); using original text")
            return text


def ensure_installed(language_codes) -> None:
    """Installs en<->X Argos packages for each code not already installed.
    Needs internet the first time per language pair; fully offline after."""
    import argostranslate.package as ap

    ap.update_package_index()
    available = ap.get_available_packages()
    installed = {(p.from_code, p.to_code) for p in ap.get_installed_packages()}

    pairs = set()
    for code in language_codes:
        if code == "en":
            continue
        pairs.add(("en", code))
        pairs.add((code, "en"))

    for from_code, to_code in sorted(pairs):
        if (from_code, to_code) in installed:
            print(f"[translate] {from_code}->{to_code} already installed")
            continue
        pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
        if pkg is None:
            print(f"[translate] no package available for {from_code}->{to_code}, skipping")
            continue
        print(f"[translate] installing {from_code}->{to_code}...")
        ap.install_from_path(pkg.download())


def main():
    import argparse
    ap_ = argparse.ArgumentParser(description="Install Argos Translate language packages")
    ap_.add_argument("languages", nargs="+", help="language codes, e.g. ja es fr ru ko")
    args = ap_.parse_args()
    ensure_installed(args.languages)


if __name__ == "__main__":
    main()
