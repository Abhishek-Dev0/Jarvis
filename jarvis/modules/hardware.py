"""
hardware.py — detects what this machine can actually run, so JARVIS scales
its own footprint to the hardware instead of assuming a fixed target.

Right now this decides faster-whisper's model size and device (cuda/cpu)
automatically — see recommend_whisper(), wired into `python -m jarvis
--voice` as the default when --whisper-model/--whisper-device aren't passed
explicitly. It's also the intended hook point for the bigger version of this
idea: auto-picking a quantization level for the local reasoning LLM once
that module exists (see the JARVIS roadmap — hybrid reasoning module, not
built yet). Not built here yet because there's no swappable-precision model
in the codebase to quantize; a hardware profile is the prerequisite either
way, so it's built first and reused when that lands.
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


def main():
    profile = detect()
    size, device = recommend_whisper(profile)
    print(f"CPU: {profile['cpu_cores']} cores / {profile['cpu_threads']} threads")
    print(f"RAM: {profile['ram_gb']} GB")
    if profile["has_cuda"]:
        print(f"GPU: {profile['gpu_name']} ({profile['vram_gb']} GB VRAM)")
    else:
        print("GPU: none (CUDA not available)")
    print(f"-> recommended whisper: model={size} device={device}")


if __name__ == "__main__":
    main()
