from jarvis.modules import hardware


def test_recommend_whisper_picks_richest_tier_that_fits_vram():
    profile = {"has_cuda": True, "vram_gb": 8.0, "ram_gb": 16.0}
    size, device = hardware.recommend_whisper(profile)
    assert device == "cuda"
    assert size == "medium"


def test_recommend_whisper_falls_back_to_ram_without_cuda():
    profile = {"has_cuda": False, "vram_gb": None, "ram_gb": 4.0}
    size, device = hardware.recommend_whisper(profile)
    assert device == "cpu"
    assert size == "small"


def test_recommend_whisper_tiny_on_very_low_resources():
    profile = {"has_cuda": False, "vram_gb": None, "ram_gb": 0.5}
    size, device = hardware.recommend_whisper(profile)
    assert size == "tiny"


def test_recommend_reasoning_model_matches_this_project_s_measured_anchor():
    # qwen2.5:3b was the actually-benchmarked, actually-working choice for a
    # 4GB-VRAM RTX 3050 this session -- this pins that regression-proof.
    profile = {"has_cuda": True, "vram_gb": 4.0, "ram_gb": 16.0}
    assert hardware.recommend_reasoning_model(profile) == "qwen2.5:3b"


def test_recommend_reasoning_model_scales_up_with_more_vram():
    profile = {"has_cuda": True, "vram_gb": 24.0, "ram_gb": 32.0}
    assert hardware.recommend_reasoning_model(profile) == "qwen2.5:32b"


def test_recommend_reasoning_model_smallest_tier_on_almost_nothing():
    profile = {"has_cuda": False, "vram_gb": None, "ram_gb": 1.0}
    assert hardware.recommend_reasoning_model(profile) == "qwen2.5:0.5b"


def test_recommend_reasoning_model_tiers_are_monotonic_in_budget():
    budgets = [0, 2, 4, 8, 16, 24, 64]
    picks = [hardware.recommend_reasoning_model({"has_cuda": True, "vram_gb": b, "ram_gb": b})
             for b in budgets]
    # every step up in budget should never pick a *smaller* model than the step before
    order = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"]
    ranks = [order.index(p) for p in picks]
    assert ranks == sorted(ranks)
