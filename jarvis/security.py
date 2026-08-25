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

try:
    from .paths import user_data_dir
except ImportError:  # pragma: no cover - legacy direct execution
    from paths import user_data_dir

_SEC_DIR = user_data_dir("security")
_PASS_HASH_PATH = os.path.join(_SEC_DIR, "passphrase.hash")
_SALT_PATH = os.path.join(_SEC_DIR, "passphrase.salt")
_VOICEPRINT_PATH = os.path.join(_SEC_DIR, "voiceprint.npy")
_FACE_EMBEDDING_PATH = os.path.join(_SEC_DIR, "face_embedding.npy")

_PBKDF2_ITERATIONS = 260_000
VOICE_MATCH_THRESHOLD = 0.55  # cosine similarity; biased toward rejecting impostors over convenience
_MIN_AUDIO_SAMPLES = 8000     # ~0.5s at 16kHz — below this, treat as "didn't catch anything"
# ArcFace (insightface's w600k_r50, via the buffalo_l pack) embeddings are
# already L2-normalized; 0.45 cosine similarity is a conservative starting
# point for "same person" on this model, same "reject impostors over
# convenience" bias as VOICE_MATCH_THRESHOLD — recalibrate against your own
# camera/lighting if it's ever too strict or too loose in practice, there's
# no universal correct value.
FACE_MATCH_THRESHOLD = 0.45


def _save_encrypted_array(path: str, array: np.ndarray) -> None:
    """Voiceprint/face embeddings are biometric data — encrypted at rest via
    Windows DPAPI (CryptProtectData), tied to the current Windows user
    account. No separate password to manage or lose: decryption only works
    for processes running as this same OS user, which is the actual threat
    this closes (another account on the machine, or the raw file copied
    elsewhere, reading your embeddings straight off disk) — it deliberately
    does NOT require your JARVIS passphrase, since face_login()'s whole
    point is working without one.
    """
    import io
    import win32crypt
    buf = io.BytesIO()
    np.save(buf, array)
    encrypted = win32crypt.CryptProtectData(buf.getvalue(), "jarvis-biometric", None, None, None, 0)
    with open(path, "wb") as f:
        f.write(encrypted)


def _load_encrypted_array(path: str) -> np.ndarray:
    import io
    import win32crypt
    with open(path, "rb") as f:
        encrypted = f.read()
    _, decrypted = win32crypt.CryptUnprotectData(encrypted, None, None, None, 0)
    return np.load(io.BytesIO(decrypted))


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


def has_face() -> bool:
    return os.path.exists(_FACE_EMBEDDING_PATH)


_face_app = None  # lazy-loaded insightface FaceAnalysis, shared across calls


def _get_face_app():
    """insightface's FaceAnalysis (buffalo_l pack): detection + a
    512-dim ArcFace recognition embedding per face, all ONNX — no C++
    compiler needed to install (unlike dlib-based alternatives), which
    matters on a Windows dev box that may not have build tools. Weights
    download once from insightface's own release assets, then cache under
    ~/.insightface/, same one-time-fetch-then-offline shape as every other
    model in this project.
    """
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


def get_face_app():
    """Public accessor for the shared, lazily-loaded FaceAnalysis instance
    — for other modules (e.g. modules/scene_watch.py's ambient presence
    detection) that need face detection/embedding too. Reuses the same
    loaded model rather than each caller loading its own copy."""
    return _get_face_app()


def _face_embed(frame) -> np.ndarray | None:
    """frame: a BGR image array (e.g. straight from cv2.VideoCapture.read()).
    Returns the L2-normalized embedding of the largest detected face, or
    None if no face was found — never raises on 'no face in frame', that's
    an expected, common outcome (bad angle, no one there yet), not an error."""
    faces = _get_face_app().get(frame)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.normed_embedding


def capture_frame(camera_index: int = 0, warmup_attempts: int = 15, warmup_delay: float = 0.15):
    """One frame from a local webcam via OpenCV. Raises RuntimeError if the
    camera can't be opened or a frame can't be read — callers decide how to
    surface that (this module doesn't know if it's running interactively or
    headless).

    warmup_attempts/warmup_delay: reading a frame immediately after
    VideoCapture opens routinely fails on real hardware — the sensor
    hasn't actually started streaming yet — with Media Foundation's
    ERROR_NOT_READY (HRESULT 0x80070015). Found by actually testing
    against a real webcam, not assumed: the fix is a short retry loop, not
    a longer one-shot wait, since the exact warm-up time varies by camera/
    driver. ~15 * 0.15s = 2.25s worst case before giving up.
    """
    import time
    import cv2
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"couldn't open camera {camera_index}")
        ok, frame = False, None
        for _ in range(warmup_attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(warmup_delay)
        if not ok or frame is None:
            raise RuntimeError(f"couldn't read a frame from camera {camera_index} "
                                f"(gave up after {warmup_attempts} attempts)")
        return frame
    finally:
        cap.release()


def enroll_face(frame=None, camera_index: int = 0) -> None:
    """One-time setup for the face factor. Overwrites any prior face
    enrollment. frame: pass a captured BGR image directly, or leave None to
    capture one now from `camera_index`. Deliberately a separate, explicit
    call from enroll() (passphrase/voice) — this is its own biometric
    factor with its own consent moment, not bundled silently into the
    others."""
    if frame is None:
        frame = capture_frame(camera_index)
    embedding = _face_embed(frame)
    if embedding is None:
        raise ValueError("no face detected in the captured frame — try again with better lighting/framing")
    os.makedirs(_SEC_DIR, exist_ok=True)
    _save_encrypted_array(_FACE_EMBEDDING_PATH, embedding)
    print("[security] face enrolled.")


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
        import torch
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
        _patch_speechbrain_windows_lazy_import_bug()
        # Windows needs Developer Mode or admin to create symlinks; COPY
        # avoids that requirement (costs a bit of extra disk, not speed —
        # this only runs once, the model is cached after).
        #
        # run_opts={"device": ...} matters more than it looks: left
        # unset, speechbrain's own internal default resolves to the bare
        # string "cuda" (no index) when CUDA is available, and something
        # downstream in speechbrain splits that on ":" expecting exactly
        # two parts ("cuda:0") — with just "cuda" that unpack raises and
        # gets caught, printing "Could not parse CUDA device string
        # 'cuda': ... Falling back to device 0" and silently continuing on
        # *some* device without it ever being clear which. Passing a fully
        # -qualified "cuda:0" ourselves sidesteps that entirely — found by
        # actually running this against a real GPU, not assumed.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(_SEC_DIR, "_speechbrain_cache"),
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device},
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
        _save_encrypted_array(_VOICEPRINT_PATH, _embed(voice_audio))
        print("[security] voiceprint enrolled from the same recording.")


def authorize_action(reason: str, security_ref=None, is_admin_ref=None, passphrase_provider=None) -> bool:
    """Shared gating check for skill modules that need SecurityGate
    verification before a state-changing action (os_control.py,
    hardware_io.py, mcp_client.py all had their own copy of exactly this
    logic — kept here once so it can't drift between them).

    security_ref: () -> SecurityGate, deferred so callers don't have to hold
    a live reference (main() may swap Jarvis.security after construction).
    is_admin_ref: () -> bool; an already-verified admin session (see
    Jarvis.run()'s wake_challenge) skips re-authorizing every action.
    passphrase_provider: optional () -> str, forwarded to
    SecurityGate.authorize() — see there for why (getpass() has no console
    to read from in a windowed GUI app).
    """
    if is_admin_ref is not None and is_admin_ref():
        return True
    if security_ref is None:
        return False
    return security_ref().authorize(reason, passphrase_provider=passphrase_provider)


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

    def face_login(self, camera_index: int = 0) -> bool:
        """On-demand, explicit face check — grants admin on a face match
        ALONE, no passphrase. This is deliberately single-factor, weaker
        than authorize()'s two-factor passphrase+voice check: Abi asked for
        it specifically as a discretion option (2026-08-22) — a way to
        become admin without speaking a passphrase out loud in public.
        That's a real, considered tradeoff he chose, not an oversight; if
        you're calling this from new code, that tradeoff is his to make
        again for that use case, not something to default to elsewhere.

        Returns False (not an exception) for "no face enrolled", "camera
        unavailable", "no face in frame", and "face didn't match" alike —
        callers don't need to distinguish those to know the answer is no.
        """
        if not has_face():
            return False
        try:
            frame = capture_frame(camera_index)
            embedding = _face_embed(frame)
        except Exception as e:
            print(f"[security] face_login camera/detection error: {e}")
            return False
        if embedding is None:
            return False
        score = float(np.dot(embedding, _load_encrypted_array(_FACE_EMBEDDING_PATH)))
        matched = score >= FACE_MATCH_THRESHOLD
        if matched:
            print(f"[security] face matched (score={score:.2f}).")
        return matched

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
        score = _cosine(_load_encrypted_array(_VOICEPRINT_PATH), _embed(audio))
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

    def authorize(self, reason: str, passphrase_provider=None) -> bool:
        """passphrase_provider: optional () -> str, used instead of
        getpass.getpass() to collect a typed passphrase when self.mic_engine
        is None. Console mode never passes one (getpass works fine from a
        real terminal); the GUI does (see gui/main_window.py) — a windowed
        app has no console for getpass to read from at all."""
        if not is_enrolled():
            self._say(f"'{reason}' requires verification, but nothing is enrolled yet. "
                       f"Run: python -m jarvis.security enroll")
            return False

        voice_mode = self.mic_engine is not None and hasattr(self.mic_engine, "transcribe")
        audio = None
        passed = False

        # One retry on a wrong/misheard passphrase (2 attempts total) — a
        # transcribed spoken passphrase is genuinely failure-prone, and
        # Abi asked for this explicitly (2026-08-22) for the "become admin"
        # flow specifically; applied to every reason uniformly since it's a
        # small, bounded usability improvement either way, not a weakening
        # (still capped at 2 attempts, not unlimited).
        for attempt in range(2):
            if voice_mode:
                if reason == "become admin":
                    prompt = "Admin control initiated. Please say the passphrase." if attempt == 0 \
                        else "That didn't match. Please say the passphrase again."
                else:
                    prompt = f"Verification required for {reason}. Please say your passphrase." \
                        if attempt == 0 else "That didn't match. Please say your passphrase again."
                self._say(prompt)
                audio = self.mic_engine.record_until_silence()
                if audio.size < _MIN_AUDIO_SAMPLES:
                    continue  # didn't catch anything -> falls through to the retry (or final denial)
                spoken = self.mic_engine.transcribe(audio)
                if not spoken:
                    continue
                passed = self._check_passphrase(spoken)
            else:
                print(f"[security] verification required for: {reason}"
                      if attempt == 0 else "[security] that didn't match, try again")
                get_passphrase = passphrase_provider or (lambda: getpass.getpass("Passphrase: "))
                typed = get_passphrase()
                if not typed:
                    continue  # e.g. GUI dialog cancelled -> falls through to retry/denial
                passed = self._check_passphrase(typed)
            if passed:
                break

        if not passed:
            self._say("Admin denied. You're a general user." if reason == "become admin"
                       else "Denied — passphrase did not match.")
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
                score = _cosine(_load_encrypted_array(_VOICEPRINT_PATH), _embed(audio))
                if score < VOICE_MATCH_THRESHOLD:
                    self._say(f"Denied — voice did not match (score={score:.2f}).")
                    return False
                print(f"[security] voice matched (score={score:.2f}).")

        self._say("Full capabilities online." if reason == "become admin" else "Verified.")
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
    enroll_p.add_argument(
        "--face", action="store_true",
        help="also enroll a face embedding via webcam — lets \"recognize me\" grant admin "
             "on a face match alone, no spoken passphrase, for when you don't want to say "
             "it out loud in public. This is a separate, weaker (single-factor) path than "
             "the passphrase+voice check; see security.SecurityGate.face_login's docstring.")
    enroll_p.add_argument("--camera-index", type=int, default=0)
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

        if args.face:
            print("[security] loading face recognition (first run downloads the model, "
                  "cached after)...")
            print(f"[security] look at camera {args.camera_index} now...")
            enroll_face(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
