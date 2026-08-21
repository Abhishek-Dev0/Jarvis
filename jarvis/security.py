"""
security.py — the verification gate for anything risky: shutdown today, and
the hook point for self-modification / OS-level actions once those exist
(call SecurityGate.authorize("reason") before doing the risky thing; proceed
only if it returns True).

Two independent factors, both required when both are enrolled:
  1. something you know — a passphrase. The plaintext is never written to
     disk. Only a salted PBKDF2 hash of a *normalized* form of it is stored;
     verification re-normalizes and re-hashes the attempt and compares
     digests. There is no way to recover the passphrase from what's on disk.
  2. something you are — a voiceprint. An embedding vector extracted from
     enrolled audio with a pretrained speaker-verification model (ECAPA-TDNN
     via speechbrain, running locally). The embedding is a fingerprint for
     "does this voice match the one that enrolled" — it cannot be turned back
     into audio, and it has nothing to do with the passphrase's contents.

Everything runs locally; no network call at verification time (speechbrain
downloads its pretrained model once, from Hugging Face, then caches it).

Two enrollment paths, matching the two ways authorize() can later be asked
for the passphrase:
  - typed  (python -m jarvis.security enroll):
        console session, getpass hides the input, no mic involved.
  - spoken (python -m jarvis.security enroll --voice):
        you speak the passphrase once; that recording is transcribed for the
        passphrase hash *and* embedded for the voiceprint. Required if JARVIS
        will ever authorize() with no keyboard attached — e.g. the background
        launcher — since authorize() there speaks the prompt and listens
        instead of calling getpass(), which would just hang with no stdin.
        Enrolling by voice keeps enrollment and verification in the same
        modality, which matters: transcribing a *typed* phrase back from
        speech later is not guaranteed to reproduce the exact same text.
        Pick something short and phonetically distinctive — plain, unusual
        words transcribe more reliably than sentences full of similar-
        sounding ones.
"""

from __future__ import annotations
import getpass
import hashlib
import hmac
import os
import re

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SEC_DIR = os.path.join(_PKG_DIR, "data", "security")
_PASS_HASH_PATH = os.path.join(_SEC_DIR, "passphrase.hash")
_SALT_PATH = os.path.join(_SEC_DIR, "passphrase.salt")
_VOICEPRINT_PATH = os.path.join(_SEC_DIR, "voiceprint.npy")

_PBKDF2_ITERATIONS = 260_000
VOICE_MATCH_THRESHOLD = 0.55  # cosine similarity; biased toward rejecting impostors over convenience
_MIN_AUDIO_SAMPLES = 8000     # ~0.5s at 16kHz — below this, treat as "didn't catch anything"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _hash_passphrase(passphrase: str, salt: bytes) -> bytes:
    normalized = _normalize(passphrase)
    return hashlib.pbkdf2_hmac("sha256", normalized.encode("utf-8"), salt, _PBKDF2_ITERATIONS)


def is_enrolled() -> bool:
    return os.path.exists(_PASS_HASH_PATH) and os.path.exists(_SALT_PATH)


def has_voiceprint() -> bool:
    return os.path.exists(_VOICEPRINT_PATH)


def _patch_speechbrain_windows_lazy_import_bug() -> None:
    """speechbrain.utils.importutils.LazyModule.ensure_module guards against
    PyTorch's op-registration machinery incidentally triggering an unwanted
    lazy import of an optional speechbrain dependency (e.g.
    speechbrain.integrations.k2_fsa, which isn't installed here and isn't
    used by anything in this file) — its own docstring explains exactly
    this. The guard checks `importer_frame.filename.endswith("/inspect.py")`
    to detect "this is inspect.py walking the stack, not real code asking
    for this module" and raise AttributeError instead of a hard ImportError.
    That check assumes a forward slash, so it never matches on Windows,
    where inspect.py's path uses backslashes — confirmed by reading
    speechbrain 1.1.0's source directly. Left unpatched: the very first time
    anything (e.g. Argos Translate's first real translate() call, if it
    happens after speechbrain has been imported) triggers PyTorch to
    register a custom op, that walk hits the lazy module, gets a real
    ImportError instead of AttributeError, and crashes code that was just
    doing a plain hasattr() check — which Translator.translate() then
    silently swallows and falls back to returning untranslated text.
    Reimplements the same method with a platform-safe basename check;
    everything else is unchanged from the original.
    """
    try:
        import inspect as _inspect
        import os as _os
        import sys as _sys
        import warnings as _warnings
        from speechbrain.utils import importutils

        def ensure_module(self, stacklevel):
            importer_frame = None
            try:
                importer_frame = _inspect.getframeinfo(_sys._getframe(stacklevel + 1))
            except AttributeError:
                _warnings.warn(
                    "Failed to inspect frame to check if we should ignore "
                    "importing a module lazily.")
            if importer_frame is not None and _os.path.basename(importer_frame.filename) == "inspect.py":
                raise AttributeError()
            if self.lazy_module is None:
                try:
                    if self.package is None:
                        self.lazy_module = importutils.importlib.import_module(self.target)
                    else:
                        self.lazy_module = importutils.importlib.import_module(
                            f".{self.target}", self.package)
                except Exception as e:
                    raise ImportError(f"Lazy import of {self!r} failed") from e
            return self.lazy_module

        importutils.LazyModule.ensure_module = ensure_module
    except Exception as e:
        print(f"[security] couldn't apply the speechbrain Windows lazy-import patch ({e}); "
              f"translation may silently no-op if it runs after voice verification in "
              f"the same process")


_verifier = None  # lazy-loaded speechbrain model, shared across calls in this process


def _get_verifier():
    global _verifier
    if _verifier is None:
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
        _patch_speechbrain_windows_lazy_import_bug()
        # Windows needs Developer Mode or admin to create symlinks; COPY
        # avoids that requirement (costs a bit of extra disk, not speed —
        # this only runs once, the model is cached after).
        _verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(_SEC_DIR, "_speechbrain_cache"),
            local_strategy=LocalStrategy.COPY,
        )
    return _verifier


def _embed(audio: np.ndarray) -> np.ndarray:
    import torch
    model = _get_verifier()
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wav)
    return emb.squeeze().cpu().numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def enroll(passphrase: str | None = None, voice_audio: np.ndarray | None = None,
           transcribe_fn=None) -> None:
    """One-time setup. Overwrites any prior enrollment.

    If voice_audio is given, the passphrase is taken from what's spoken in
    it (via transcribe_fn) instead of typed — see the module docstring for
    why that matters. transcribe_fn: callable(audio) -> str | None, required
    together with voice_audio.
    """
    os.makedirs(_SEC_DIR, exist_ok=True)

    if voice_audio is not None:
        if transcribe_fn is None:
            raise ValueError("transcribe_fn is required to enroll a spoken passphrase")
        passphrase = transcribe_fn(voice_audio)
        if not passphrase:
            raise ValueError("couldn't transcribe anything from the recording")
        print(f'[security] heard: "{passphrase}" — say this back to verify later.')
    elif passphrase is None:
        passphrase = getpass.getpass("Set the verification passphrase (input hidden): ")

    if not passphrase or not _normalize(passphrase):
        raise ValueError("passphrase cannot be empty")

    salt = os.urandom(16)
    digest = _hash_passphrase(passphrase, salt)
    with open(_SALT_PATH, "wb") as f:
        f.write(salt)
    with open(_PASS_HASH_PATH, "wb") as f:
        f.write(digest)
    print("[security] passphrase stored as a salted hash — the passphrase itself was not saved.")

    if voice_audio is not None:
        np.save(_VOICEPRINT_PATH, _embed(voice_audio))
        print("[security] voiceprint enrolled from the same recording.")


def authorize_action(reason: str, security_ref=None, is_admin_ref=None) -> bool:
    """Shared gating check for skill modules that need SecurityGate
    verification before a state-changing action (os_control.py,
    hardware_io.py, mcp_client.py all had their own copy of exactly this
    logic — kept here once so it can't drift between them).

    security_ref: () -> SecurityGate, deferred so callers don't have to hold
    a live reference (main() may swap Jarvis.security after construction).
    is_admin_ref: () -> bool; an already-verified admin session (see
    Jarvis.run()'s wake_challenge) skips re-authorizing every action.
    """
    if is_admin_ref is not None and is_admin_ref():
        return True
    if security_ref is None:
        return False
    return security_ref().authorize(reason)


class SecurityGate:
    """Call authorize(reason) before anything irreversible or risky.
    Returns True only if every enrolled factor passes."""

    def __init__(self, mic_engine=None, tts_engine=None):
        # mic_engine: something with record_until_silence() -> mono float32
        # audio at 16kHz AND transcribe(audio) -> str|None (WhisperSTTEngine
        # satisfies this). When present, the passphrase is captured by
        # listening instead of getpass() — required for any headless/
        # background run, where there is no stdin to read a typed answer
        # from. tts_engine: anything with speak(text) — used to voice the
        # prompts/results when present; purely a UX nicety, verification
        # logic doesn't depend on it.
        self.mic_engine = mic_engine
        self.tts_engine = tts_engine

    def _say(self, text: str) -> None:
        print(f"[security] {text}")
        if self.tts_engine is not None:
            self.tts_engine.speak(text)

    def _check_passphrase(self, attempt: str) -> bool:
        with open(_SALT_PATH, "rb") as f:
            salt = f.read()
        with open(_PASS_HASH_PATH, "rb") as f:
            expected = f.read()
        return hmac.compare_digest(_hash_passphrase(attempt, salt), expected)

    def wake_challenge(self, admin_name: str, affirmative_phrases=None) -> bool:
        """Called once at voice-session start. If the very first thing heard
        matches the enrolled voiceprint, asks "<admin_name>, are you the
        admin?" and on a yes, runs the normal authorize() passphrase+voice
        check. Returns True only on a full successful admin login.

        Silent (no prompt spoken, no trace in output beyond the [security]
        log line) if the voice doesn't match or nothing is enrolled — a
        stranger talking to JARVIS never learns an admin gate exists at all.

        affirmative_phrases: optional set of normalized "yes"-equivalent
        strings to check the reply against (see runtime/jarvis.py, which
        builds one via Translator so this works in whatever language was
        just detected); falls back to a small English-only set.
        """
        if self.mic_engine is None or not is_enrolled() or not has_voiceprint():
            return False

        audio = self.mic_engine.record_until_silence()
        if audio.size < _MIN_AUDIO_SAMPLES:
            return False
        score = _cosine(np.load(_VOICEPRINT_PATH), _embed(audio))
        if score < VOICE_MATCH_THRESHOLD:
            return False  # not the enrolled voice — say nothing, proceed as a normal session

        self._say(f"Are you the admin, {admin_name}?")
        reply_audio = self.mic_engine.record_until_silence()
        if reply_audio.size < _MIN_AUDIO_SAMPLES:
            return False
        reply_text = self.mic_engine.transcribe(reply_audio) or ""
        affirmatives = affirmative_phrases or {"yes", "yeah", "yep", "correct", "affirmative", "yup"}
        if _normalize(reply_text) not in affirmatives:
            self._say("Understood — continuing as a regular user.")
            return False

        return self.authorize("become admin")

    def authorize(self, reason: str) -> bool:
        if not is_enrolled():
            self._say(f"'{reason}' requires verification, but nothing is enrolled yet. "
                       f"Run: python -m jarvis.security enroll")
            return False

        voice_mode = self.mic_engine is not None and hasattr(self.mic_engine, "transcribe")
        audio = None

        if voice_mode:
            self._say(f"Verification required for {reason}. Please say your passphrase.")
            audio = self.mic_engine.record_until_silence()
            if audio.size < _MIN_AUDIO_SAMPLES:
                self._say("Denied — didn't catch anything.")
                return False
            spoken = self.mic_engine.transcribe(audio)
            if not spoken:
                self._say("Denied — couldn't understand that.")
                return False
            passed = self._check_passphrase(spoken)
        else:
            print(f"[security] verification required for: {reason}")
            passed = self._check_passphrase(getpass.getpass("Passphrase: "))

        if not passed:
            self._say("Denied — passphrase did not match.")
            return False

        if has_voiceprint():
            if audio is None and self.mic_engine is not None:
                self._say("Now say something so I can check your voice.")
                audio = self.mic_engine.record_until_silence()
            if audio is None or audio.size < _MIN_AUDIO_SAMPLES:
                if self.mic_engine is not None:
                    self._say("Denied — didn't catch your voice for the biometric check.")
                    return False
                # no mic available at all (console-only session): voice
                # factor is silently skipped, same as when nothing is
                # enrolled for it — passphrase alone stands.
            else:
                score = _cosine(np.load(_VOICEPRINT_PATH), _embed(audio))
                if score < VOICE_MATCH_THRESHOLD:
                    self._say(f"Denied — voice did not match (score={score:.2f}).")
                    return False
                print(f"[security] voice matched (score={score:.2f}).")

        self._say("Verified.")
        return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="JARVIS security enrollment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    enroll_p = sub.add_parser("enroll", help="set the passphrase and (optionally) a voiceprint")
    enroll_p.add_argument(
        "--voice", action="store_true",
        help="enroll by speaking your passphrase once — required if you'll ever run "
             "JARVIS with --voice (especially the background launcher, which has no "
             "keyboard to type into); that same recording also sets your voiceprint.")
    enroll_p.add_argument("--whisper-model", default="base.en")
    args = ap.parse_args()

    if args.cmd == "enroll":
        if args.voice:
            try:
                from jarvis.modules.voice import WhisperSTTEngine
            except ImportError:  # pragma: no cover - legacy direct execution
                from modules.voice import WhisperSTTEngine
            print("[security] loading whisper...")
            engine = WhisperSTTEngine(model_size=args.whisper_model)
            print("[security] speak your passphrase now, then go quiet. Pick something short "
                  "and phonetically distinctive.")
            audio = engine.record_until_silence()
            enroll(voice_audio=audio, transcribe_fn=engine.transcribe)
        else:
            enroll()


if __name__ == "__main__":
    main()
