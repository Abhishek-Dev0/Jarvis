"""
voice.py — local speech engines: faster-whisper for STT, Kokoro for TTS.

Both run fully offline at inference time (faster-whisper fetches its model
from Hugging Face on first use only, then caches it; Kokoro's weights are
already on disk under data/models/kokoro/). No API key, no per-call network
request, no cost.

These are "engine" objects — plug them into the SpeechInput/SpeechOutput
shells in builtin.py. Nothing else in the system needs to know an engine
exists; the Registry only ever sees the InputModule/OutputModule interface.
"""

from __future__ import annotations
import os
import sys
import numpy as np

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PKG_DIR, "data", "models")

SAMPLE_RATE = 16000


def _register_cuda_dll_dirs() -> None:
    """faster-whisper's backend (ctranslate2) links against CUDA 12's
    cublas64_12.dll/cudnn64_9.dll. PyTorch on this project is a cu118 build
    (bundles its own cublas64_11.dll instead) — a real version mismatch
    found by actually running --voice, not assumed away: WhisperSTTEngine's
    constructor succeeds either way, but the first real transcribe() call
    on cuda raised "Library cublas64_12.dll is not found or cannot be
    loaded" and crashed the whole process (see the run() loop's exception
    handling below, which is the second half of this fix).

    `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` puts the right DLLs
    on disk under site-packages/nvidia/*/bin/, but pip does not add that
    directory to Windows' DLL search path — ctranslate2 still can't find
    them until something calls os.add_dll_directory() on it, which is what
    this does, once, before any CUDA-backed engine loads. Safe to call even
    if those packages aren't installed (skips) or on non-Windows (no-op —
    Linux/Mac resolve shared libraries differently and don't have this
    problem in the same way).
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    # nvidia-cublas-cu12/nvidia-cudnn-cu12 both contribute to the same
    # `nvidia` PEP 420 namespace package (no single __init__.py, so no
    # nvidia.__file__) — __path__ is the real, always-present way to find
    # where each was actually installed.
    for nvidia_dir in nvidia.__path__:
        for pkg in ("cublas", "cudnn"):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass


_register_cuda_dll_dirs()


def record_until_silence(silence_duration=1.2, silence_threshold=0.02,
                          max_seconds=30) -> np.ndarray:
    """Records mono float32 audio at SAMPLE_RATE from the default microphone
    until `silence_duration` seconds of quiet follow speech.

    Standalone (not tied to any STT engine) so anything needing a raw voice
    sample — transcription, biometric verification — can call it without
    loading a model it doesn't need.
    """
    import sounddevice as sd

    block_ms = 100
    block_size = int(SAMPLE_RATE * block_ms / 1000)
    silence_blocks_needed = max(1, int(silence_duration * 1000 / block_ms))
    max_blocks = int(max_seconds * 1000 / block_ms)

    chunks = []
    silence_run = 0
    heard_speech = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=block_size) as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(block_size)
            block = block[:, 0]
            chunks.append(block)
            level = float(np.abs(block).mean())
            if level > silence_threshold:
                heard_speech = True
                silence_run = 0
            elif heard_speech:
                silence_run += 1
                if silence_run >= silence_blocks_needed:
                    break
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


class WhisperSTTEngine:
    """Records from the default microphone, transcribes with faster-whisper.

    Uses record_until_silence()'s energy-based VAD for turn-taking — Whisper's
    own `vad_filter` cleans up whatever silence still leaks into the clip.
    Returns raw transcribed text, including things like "quit" or "shutdown" —
    the orchestrator decides what those mean, this engine just transcribes.

    No `language=` is passed to transcribe(), so Whisper auto-detects it from
    the audio (multilingual model, not an `.en`-suffixed one) — `last_language`
    holds the ISO 639-1 code (e.g. "en", "ja", "ru") from the most recent call,
    for callers that want to react to what language was actually spoken.
    """

    def __init__(self, model_size="small", device="cpu", compute_type="int8",
                 silence_duration=1.2, silence_threshold=0.02, max_seconds=30):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.max_seconds = max_seconds
        self.last_language: str | None = None

    def record_until_silence(self) -> np.ndarray:
        return record_until_silence(self.silence_duration, self.silence_threshold,
                                     self.max_seconds)

    def transcribe(self, audio: np.ndarray) -> str | None:
        if audio.size < SAMPLE_RATE * 0.3:   # too short to be real speech
            return None
        segments, info = self.model.transcribe(audio, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            self.last_language = info.language
        return text or None


class KokoroTTSEngine:
    """Synthesizes speech with Kokoro (ONNX, local weights) and plays it back."""

    def __init__(self, voice="af_heart", speed=1.0, lang="en-us",
                 model_path=None, voices_path=None):
        from kokoro_onnx import Kokoro
        model_path = model_path or os.path.join(_MODELS_DIR, "kokoro", "kokoro-v1.0.int8.onnx")
        voices_path = voices_path or os.path.join(_MODELS_DIR, "kokoro", "voices-v1.0.bin")
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            raise FileNotFoundError(
                f"Kokoro model files missing — expected {model_path} and {voices_path}")
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed
        self.lang = lang

    def speak(self, text: str, lang: str | None = None) -> None:
        # lang is accepted (and ignored — this engine is a fixed single
        # voice/language) so callers can treat every TTS engine uniformly;
        # see PersonaTTSEngine.speak for the one that actually uses it.
        text = text.strip()
        if not text:
            return
        import sounddevice as sd
        samples, sample_rate = self.kokoro.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang)
        sd.play(samples, sample_rate)
        sd.wait()


class PiperTTSEngine:
    """Synthesizes speech with Piper (ONNX, local weights) for languages
    Kokoro doesn't cover (Russian, Korean, Arabic, Bengali) plus the French
    male voice Kokoro lacks (Kokoro's French is female-only, ff_siwis).

    speaker_id selects a voice within a multi-speaker model (only the
    Bengali "google" voice is multi-speaker here, 16 speakers bundled in one
    file) — ignored by single-speaker models.
    """

    def __init__(self, model_file, speaker_id: int | None = None):
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        model_path = os.path.join(_MODELS_DIR, "piper", model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper voice missing — expected {model_path}")
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(speaker_id=speaker_id) if speaker_id is not None else None

    def speak(self, text: str, lang: str | None = None) -> None:
        text = text.strip()
        if not text:
            return
        import sounddevice as sd
        chunks = list(self.voice.synthesize(text, syn_config=self.syn_config))
        if not chunks:
            return
        audio = np.concatenate([c.audio_float_array for c in chunks])
        sd.play(audio, chunks[0].sample_rate)
        sd.wait()


# language -> {"jarvis": <selector>, "eve": <selector>}, selector being
# "kokoro:<voice_id>:<kokoro_lang_code>" or "piper:<model_filename>[:speaker_id]".
# Only languages with actual voice weights on disk
# (jarvis/data/models/{kokoro,piper}/) are listed.
#
# Known single-voice gaps, shared by both personas rather than pretending a
# second voice exists: Korean (only public Piper voice: kss), Arabic (only
# public Piper voice: kareem). Bengali's Piper voice is multi-speaker (16
# speakers in one model) — speaker ids 0 and 1 are used to give jarvis/eve
# distinct voices, but their actual gender wasn't verified by ear, only that
# they're different from each other.
_VOICE_TABLE = {
    "en": {"jarvis": "kokoro:am_michael:en-us", "eve": "kokoro:af_heart:en-us"},
    "ja": {"jarvis": "kokoro:jm_kumo:ja",        "eve": "kokoro:jf_alpha:ja"},
    "es": {"jarvis": "kokoro:em_alex:es",        "eve": "kokoro:ef_dora:es"},
    "fr": {"jarvis": "piper:fr_FR-tom-medium.onnx", "eve": "kokoro:ff_siwis:fr-fr"},
    "ru": {"jarvis": "piper:ru_RU-dmitri-medium.onnx", "eve": "piper:ru_RU-irina-medium.onnx"},
    "ko": {"jarvis": "piper:ko_KR-kss-medium.onnx", "eve": "piper:ko_KR-kss-medium.onnx"},
    "zh": {"jarvis": "kokoro:zm_yunjian:cmn",    "eve": "kokoro:zf_xiaoxiao:cmn"},
    "hi": {"jarvis": "kokoro:hm_omega:hi",       "eve": "kokoro:hf_alpha:hi"},
    "pt": {"jarvis": "kokoro:pm_alex:pt-br",     "eve": "kokoro:pf_dora:pt-br"},
    "ar": {"jarvis": "piper:ar_JO-kareem-medium.onnx", "eve": "piper:ar_JO-kareem-medium.onnx"},
    "bn": {"jarvis": "piper:bn_BD-google-medium.onnx:0", "eve": "piper:bn_BD-google-medium.onnx:1"},
}
SUPPORTED_LANGUAGES = tuple(_VOICE_TABLE)


class PersonaTTSEngine:
    """Routes speech to Kokoro or Piper depending on language, and to one of
    two fixed voice identities — "jarvis" (male) or "eve" (female) — within
    whichever language is asked for. Loads engines lazily and caches them
    (each is a real model in memory; no point loading all 6 languages'
    worth up front for a conversation that mostly stays in one).
    """

    def __init__(self, persona="jarvis", default_lang="en"):
        if persona not in ("jarvis", "eve"):
            raise ValueError('persona must be "jarvis" or "eve"')
        self.persona = persona
        self.default_lang = default_lang
        self._engines: dict[str, object] = {}  # selector string -> loaded engine

    def _engine_for(self, selector: str):
        if selector in self._engines:
            return self._engines[selector]
        kind, rest = selector.split(":", 1)
        if kind == "kokoro":
            voice_id, lang_code = rest.split(":")
            engine = KokoroTTSEngine(voice=voice_id, lang=lang_code)
        elif kind == "piper":
            parts = rest.split(":")
            model_file = parts[0]
            speaker_id = int(parts[1]) if len(parts) > 1 else None
            engine = PiperTTSEngine(model_file=model_file, speaker_id=speaker_id)
        else:
            raise ValueError(f"unknown TTS engine kind: {kind}")
        self._engines[selector] = engine
        return engine

    def speak(self, text: str, lang: str | None = None) -> None:
        lang = lang or self.default_lang
        table = _VOICE_TABLE.get(lang)
        if table is None:
            print(f"[voice] no TTS voice for language '{lang}', falling back to '{self.default_lang}'")
            table = _VOICE_TABLE[self.default_lang]
        self._engine_for(table[self.persona]).speak(text)
