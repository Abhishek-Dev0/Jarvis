import os

import numpy as np
import pytest

skimage = pytest.importorskip("skimage")

from jarvis.modules.design_engine import (
    DISCLAIMER, DesignEngineSkill, cantilever_bracket, intersect, sd_box,
    sd_sphere, subtract, to_mesh, union, voxelize, write_stl,
)


def _grid_points(n=9, extent=2.0):
    xs = np.linspace(-extent, extent, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


def test_sd_sphere_sign_inside_outside_and_on_surface():
    p = _grid_points()
    d = sd_sphere(p, radius=1.0)
    center = d[p.shape[0] // 2, p.shape[1] // 2, p.shape[2] // 2]
    assert center < 0  # origin is inside
    far = sd_sphere(np.array([[5.0, 0.0, 0.0]]), radius=1.0)
    assert far[0] > 0
    on_surface = sd_sphere(np.array([[1.0, 0.0, 0.0]]), radius=1.0)
    assert on_surface[0] == pytest.approx(0.0, abs=1e-9)


def test_union_is_the_closer_surface():
    p = np.array([[0.3, 0.0, 0.0]])
    d1 = sd_sphere(p, radius=0.2)   # outside (0.1)
    d2 = sd_sphere(p, radius=0.5)   # inside (-0.2)
    assert union(d1, d2)[0] == pytest.approx(-0.2)


def test_intersect_and_subtract_basic_cases():
    p = np.array([[0.0, 0.0, 0.0]])
    d1 = sd_sphere(p, radius=1.0)   # inside, -1
    d2 = sd_sphere(p, radius=0.5)   # inside, -0.5
    assert intersect(d1, d2)[0] == pytest.approx(-0.5)  # the tighter surface wins
    assert subtract(d1, d2)[0] == pytest.approx(0.5)    # d2 carved out -> outside


def test_sd_box_matches_known_distances():
    box = np.array([[3.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    d = sd_box(box, size=(1.0, 1.0, 1.0))
    assert d[0] == pytest.approx(2.0)   # 2 units outside the +x face
    assert d[1] == pytest.approx(-0.5)  # 0.5 inside the +x face


def test_voxelize_and_mesh_sphere_is_plausible():
    sdf = lambda p: sd_sphere(p, radius=5.0)
    bounds = ((-6, 6), (-6, 6), (-6, 6))
    grid, spacing, origin = voxelize(sdf, bounds, resolution=1.0)
    verts, faces = to_mesh(grid, spacing)
    assert len(verts) > 0 and len(faces) > 0
    world = verts + np.array(origin)
    radii = np.linalg.norm(world, axis=1)
    assert radii.mean() == pytest.approx(5.0, rel=0.1)


def test_to_mesh_raises_when_field_never_crosses_zero():
    sdf = lambda p: sd_sphere(p, radius=100.0)  # entirely inside a small grid
    grid, spacing, _ = voxelize(sdf, ((-1, 1), (-1, 1), (-1, 1)), resolution=1.0)
    with pytest.raises(ValueError):
        to_mesh(grid, spacing)


def test_write_stl_produces_valid_binary_stl(tmp_path):
    sdf = lambda p: sd_sphere(p, radius=3.0)
    grid, spacing, origin = voxelize(sdf, ((-4, 4), (-4, 4), (-4, 4)), resolution=1.0)
    verts, faces = to_mesh(grid, spacing)
    path = str(tmp_path / "sphere.stl")
    write_stl(path, verts, faces)

    size = os.path.getsize(path)
    assert size == 84 + 50 * len(faces)  # 80-byte header + 4-byte count + 50 bytes/triangle
    with open(path, "rb") as f:
        f.read(80)
        count = int.from_bytes(f.read(4), "little")
    assert count == len(faces)


def test_cantilever_bracket_matches_hand_computed_beam_bending():
    result = cantilever_bracket(load_n=500.0, span_mm=100.0, width_mm=20.0,
                                 safety_factor=2.0, material="6061-T6 aluminum")
    report = result["report"]
    # Independently hand-computed: sigma_allow = 276/2 = 138 MPa;
    # M = 500N * 100mm = 50000 N*mm; Z_needed = M/sigma_allow = 362.3188 mm^3;
    # t_root = sqrt(6*Z_needed/width) = sqrt(6*362.3188/20) = 10.42572 mm
    assert report["max_bending_stress_mpa"] == pytest.approx(138.0)
    assert report["root_thickness_mm"] == pytest.approx(10.42572, abs=1e-4)
    assert report["achieved_safety_factor"] == pytest.approx(2.0, rel=1e-6)
    assert result["sdf"] is not None


def test_cantilever_bracket_rejects_unknown_material():
    with pytest.raises(ValueError):
        cantilever_bracket(load_n=100.0, span_mm=50.0, width_mm=10.0, material="unobtainium")


def test_skill_matches_only_design_triggers():
    skill = DesignEngineSkill()
    assert skill.matches("design a bracket load=500N span=150mm")
    assert skill.matches("design a heat sink power=15")
    assert not skill.matches("what is 2 + 2")


def test_skill_handle_bracket_end_to_end_writes_stl(tmp_path):
    skill = DesignEngineSkill(output_dir=str(tmp_path))
    report = skill.handle("design a bracket load=500 span=100 width=20")
    assert DISCLAIMER in report
    assert "root_thickness_mm" in report
    files = list(tmp_path.glob("bracket_*.stl"))
    assert len(files) == 1
    assert files[0].stat().st_size > 84


def test_skill_handle_unknown_kind_prompts_instead_of_crashing():
    skill = DesignEngineSkill()
    assert skill.matches("design a bracket")
    reply = skill.handle("design a widget")
    assert "Design what" in reply
