"""
hardware.py — detects what this machine can actually run, so JARVIS scales
its own footprint to the hardware instead of assuming a fixed target.

Two things are auto-sized here:
  - faster-whisper's model size and device (cuda/cpu) — recommend_whisper(),
    wired into `python -m jarvis --voice` when --whisper-model/--whisper-device
    aren't passed explicitly.
  - the local reasoning LLM's size — recommend_reasoning_model(), wired into
    `python -m jarvis` when --reasoning-model isn't passed explicitly. This is
    the "if a system can run 4B params, quantize to that; if it can run full,
    don't" ask from the 2026-08-22 roadmap decision. It's a size picker, not a
    literal quantization-format picker: Ollama's default pull for every tag
    below is already a 4-bit (Q4_K_M) quantization, so "size" and "quant
    level" collapse into the same knob here — there's no separate fp16/int8/
    int4 dial to turn per model the way there was nothing to turn before this
    existed. Picking the biggest tag that comfortably fits *is* the
    quantization-aware choice for a fixed quant format.
"""

from __future__ import annotations
import os


def detect() -> dict:
    """Returns what's actually on this machine — real numbers, not guesses."""
    import psutil
    info = {
        "cpu_cores": psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        "cpu_threads": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "has_cuda": False,
        "gpu_name": None,
        "vram_gb": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["has_cuda"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    return info


# (min_gb, model_size), richest-first, checked against VRAM if CUDA is
# available else system RAM. Picks the best tier this machine clears; falls
# back to "tiny" if nothing else fits. Thresholds are rough headroom for
# faster-whisper's int8/float16 compute, not exact requirements — the actual
# footprint is well under the model's raw parameter count.
_WHISPER_TIERS = [
    (8, "medium"),
    (4, "small"),
    (2, "base"),
    (0, "tiny"),
]


def recommend_whisper(profile: dict | None = None) -> tuple[str, str]:
    """Returns (model_size, device) sized to what this machine can run."""
    profile = profile or detect()
    device = "cuda" if profile["has_cuda"] else "cpu"
    budget = profile["vram_gb"] if profile["has_cuda"] else profile["ram_gb"]
    for min_gb, size in _WHISPER_TIERS:
        if (budget or 0) >= min_gb:
            return size, device
    return "tiny", device


# (min_gb, ollama_tag), richest-first. Thresholds assume Ollama's default
# Q4_K_M pull (~0.7-0.8 GB per billion params, plus headroom for KV-cache/
# context and OS/runtime overhead) and CUDA VRAM if available, else system
# RAM for CPU-only inference — same budget logic as _WHISPER_TIERS. Measured
# anchor point: qwen2.5:3b on this project's 4GB-VRAM RTX 3050 ran at ~1.6s/
# response warm; qwen2.5:14b or bigger on 4GB would mostly run on CPU the
# same way the originally-tried 12B Gemma model did (~3m48s for one
# sentence, see modules/reasoning.py) — that measured failure is the actual
# basis for these thresholds, not a guess.
_REASONING_TIERS = [
    (24, "qwen2.5:32b"),
    (16, "qwen2.5:14b"),
    (8, "qwen2.5:7b"),
    (4, "qwen2.5:3b"),
    (2, "qwen2.5:1.5b"),
    (0, "qwen2.5:0.5b"),
]


def recommend_reasoning_model(profile: dict | None = None) -> str:
    """Returns an Ollama model tag sized to what this machine can run,
    biggest-that-fits. Overridable with `--reasoning-model` on the CLI if you
    want a different model family or know your real headroom better than
    this heuristic does."""
    profile = profile or detect()
    budget = profile["vram_gb"] if profile["has_cuda"] else profile["ram_gb"]
    for min_gb, tag in _REASONING_TIERS:
        if (budget or 0) >= min_gb:
            return tag
    return "qwen2.5:0.5b"


def main():
    profile = detect()
    size, device = recommend_whisper(profile)
    reasoning_tag = recommend_reasoning_model(profile)
    print(f"CPU: {profile['cpu_cores']} cores / {profile['cpu_threads']} threads")
    print(f"RAM: {profile['ram_gb']} GB")
    if profile["has_cuda"]:
        print(f"GPU: {profile['gpu_name']} ({profile['vram_gb']} GB VRAM)")
    else:
        print("GPU: none (CUDA not available)")
    print(f"-> recommended whisper: model={size} device={device}")
    print(f"-> recommended reasoning model: {reasoning_tag}")


if __name__ == "__main__":
    main()
