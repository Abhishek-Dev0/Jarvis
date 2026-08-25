"""
design_engine.py — computational-design skill: parametric functions that
encode closed-form engineering physics and emit 3D-printable geometry
directly, no CAD step in between.

The inspiration is LEAP71's Noyron: a "computational engineering model"
where executable functions carry the design logic (physics, constraints,
manufacturing rules) and generate geometry directly as an implicit/voxel
field, instead of a human sketching curves in a CAD tool. What's here is
the same idea at hobbyist scale, honestly scoped down:

  - geometry is represented as signed-distance-ish scalar fields (SDFs),
    combined with CSG (union/intersect/subtract), evaluated on a voxel
    grid, and turned into a triangle mesh via marching cubes;
  - "design functions" (bracket, duct/nozzle, heat sink) take physical
    requirements and solve simple closed-form formulas — beam bending,
    natural-convection heat transfer, duct-contour interpolation — for
    the geometry that satisfies them, then build the field.

What's NOT here, on purpose: no FEA, no CFD, no topology-optimization
solver, no certified material database. Every report carries the same
disclaimer. Treat the output as a first-pass parametric starting point,
not a validated design — verify before manufacturing or load-bearing use.
"""

from __future__ import annotations

import os
import re
import struct
import time

import numpy as np

try:
    from .base import SkillModule
    from ..paths import user_data_dir
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule
    from paths import user_data_dir

DISCLAIMER = (
    "This geometry comes from closed-form textbook engineering formulas "
    "(beam bending / natural convection / simple duct contours) evaluated "
    "on a signed-distance field — not FEA, not CFD, and not a certified "
    "design. Verify before manufacturing or load-bearing use."
)

_DEFAULT_OUTPUT_DIR = user_data_dir("design_engine")


# --------------------------------------------------------------- primitives
# Each primitive/combinator takes points p of shape (..., 3) and returns a
# scalar field of shape (...). Negative = inside the solid. The box-style
# combinators below aren't exact Euclidean SDFs everywhere (some are
# max/min approximations) but they're sign-correct at the isosurface, which
# is all marching cubes needs.

def sd_sphere(p: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(p, axis=-1) - radius


def sd_box(p: np.ndarray, size) -> np.ndarray:
    """size: (sx, sy, sz) half-extents."""
    size = np.asarray(size, dtype=float)
    q = np.abs(p) - size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def sd_cylinder(p: np.ndarray, radius: float, height: float) -> np.ndarray:
    """Axis along z, centered at the origin."""
    radial = np.linalg.norm(p[..., :2], axis=-1) - radius
    axial = np.abs(p[..., 2]) - height / 2.0
    d = np.stack([radial, axial], axis=-1)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


def sd_capsule(p: np.ndarray, a, b, radius: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ba = b - a
    pa = p - a
    h = np.clip(np.sum(pa * ba, axis=-1) / np.dot(ba, ba), 0.0, 1.0)
    return np.linalg.norm(pa - ba * h[..., None], axis=-1) - radius


def union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.minimum(d1, d2)


def intersect(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.maximum(d1, d2)


def subtract(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """d1 with d2 removed."""
    return np.maximum(d1, -d2)


def smooth_union(d1: np.ndarray, d2: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth-min union — blends the two into a filleted joint."""
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def shell(d: np.ndarray, thickness: float) -> np.ndarray:
    return np.abs(d) - thickness / 2.0


def gyroid_infill(p: np.ndarray, period: float, thickness: float) -> np.ndarray:
    """TPMS gyroid lattice — the kind of self-supporting infill Noyron-style
    engines use for cooling channels / lightweighting. Usable as an infill
    field intersected with any solid above."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    w = 2 * np.pi / period
    g = (np.sin(w * x) * np.cos(w * y) + np.sin(w * y) * np.cos(w * z)
         + np.sin(w * z) * np.cos(w * x))
    return np.abs(g) - thickness


# ------------------------------------------------------- voxelize + export

def voxelize(sdf_fn, bounds, resolution: float):
    """bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in mm.
    Returns (grid, spacing, origin)."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    nx = max(2, int(round((xmax - xmin) / resolution)) + 1)
    ny = max(2, int(round((ymax - ymin) / resolution)) + 1)
    nz = max(2, int(round((zmax - zmin) / resolution)) + 1)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)
    grid = sdf_fn(pts)
    spacing = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
    origin = (xmin, ymin, zmin)
    return grid, spacing, origin


def to_mesh(grid: np.ndarray, spacing):
    """Marching cubes at the zero isosurface. Raises if the field never
    crosses zero within the grid (nothing to mesh — check bounds/params)."""
    from skimage.measure import marching_cubes
    if grid.min() > 0 or grid.max() < 0:
        raise ValueError(
            "field doesn't cross zero within bounds — nothing to mesh "
            "(check bounds/resolution/parameters)"
        )
    verts, faces, _normals, _values = marching_cubes(grid, level=0.0, spacing=spacing)
    return verts, faces


def write_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> str:
    """Hand-rolled binary STL writer — no extra dependency for this part."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tri = verts[faces]  # (F, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = (normals / lengths).astype(np.float32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"JARVIS design_engine binary STL".ljust(80, b"\0")[:80])
        f.write(struct.pack("<I", len(faces)))
        for n, t in zip(normals, tri):
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *t[0]))
            f.write(struct.pack("<3f", *t[1]))
            f.write(struct.pack("<3f", *t[2]))
            f.write(struct.pack("<H", 0))
    return path


# --------------------------------------------------------- design functions
# Each takes physical parameters, solves closed-form formulas for the
# geometry that satisfies them, and returns
#   {"sdf": callable, "bounds": ..., "report": {...}, "warnings": [...]}

_MATERIALS = {
    "6061-t6 aluminum": {"yield_mpa": 276.0, "density_kg_m3": 2700.0},
    "steel a36": {"yield_mpa": 250.0, "density_kg_m3": 7850.0},
    "pla": {"yield_mpa": 50.0, "density_kg_m3": 1240.0},
    "petg": {"yield_mpa": 45.0, "density_kg_m3": 1270.0},
}


def cantilever_bracket(load_n: float, span_mm: float, width_mm: float,
                        safety_factor: float = 2.0,
                        material: str = "6061-t6 aluminum") -> dict:
    """Rectangular cantilever, fixed at x=0, point load at the free tip
    (x=span). Solves beam bending (sigma = M/Z, Z = width*t^2/6) for the
    root thickness that hits the target safety factor, then tapers the
    thickness along x to roughly track the bending-moment diagram (max at
    the root, a printable minimum near the tip) instead of shipping a
    constant-thickness block."""
    mat = _MATERIALS.get(material.lower())
    if mat is None:
        raise ValueError(f"unknown material '{material}' (choices: {list(_MATERIALS)})")
    if load_n <= 0 or span_mm <= 0 or width_mm <= 0 or safety_factor <= 0:
        raise ValueError("load_n, span_mm, width_mm, safety_factor must all be positive")

    yield_mpa = mat["yield_mpa"]
    allow_mpa = yield_mpa / safety_factor
    moment_root = load_n * span_mm  # N*mm
    z_needed = moment_root / allow_mpa  # mm^3
    t_root = float(np.sqrt(6.0 * z_needed / width_mm))
    t_tip = max(2.0, t_root * 0.15)  # printable floor near the tip

    warnings = []
    if t_root < 1.0:
        warnings.append("computed root thickness under 1mm — likely too thin to print reliably")

    def sdf(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        frac = np.clip(1.0 - x / span_mm, 0.0, 1.0)  # 1 at root, 0 at tip
        half_t = 0.5 * (t_tip + (t_root - t_tip) * frac)
        half_w = width_mm / 2.0
        d_x = np.maximum(-x, x - span_mm)
        d_y = np.abs(y) - half_w
        d_z = np.abs(z) - half_t
        return np.maximum(np.maximum(d_x, d_y), d_z)

    pad = max(5.0, t_root)
    bounds = ((-1.0, span_mm + 1.0),
              (-width_mm / 2 - 1.0, width_mm / 2 + 1.0),
              (-t_root / 2 - pad / 4, t_root / 2 + pad / 4))

    avg_t = (t_root + t_tip) / 2.0
    volume_mm3 = span_mm * width_mm * avg_t
    mass_g = volume_mm3 * 1e-9 * mat["density_kg_m3"] * 1e3  # mm^3->m^3, kg->g

    report = {
        "material": material, "load_n": load_n, "span_mm": span_mm, "width_mm": width_mm,
        "yield_mpa": yield_mpa, "target_safety_factor": safety_factor,
        "max_bending_stress_mpa": allow_mpa, "achieved_safety_factor": yield_mpa / allow_mpa,
        "root_thickness_mm": t_root, "tip_thickness_mm": t_tip, "mass_g": mass_g,
    }
    return {"sdf": sdf, "bounds": bounds, "report": report, "warnings": warnings}


def duct_nozzle(inlet_d: float, outlet_d: float, length_mm: float, wall_mm: float,
                 contour: str = "conical") -> dict:
    """General converging/diverging duct: a hollow shell swept along z whose
    outer radius interpolates from inlet_d/2 to outlet_d/2. 'conical' is a
    straight-line taper; 'bell' eases in/out (cosine blend) — a rough shape
    cue, not a real characteristic-method bell nozzle contour. Works for
    anything from a nozzle to a venturi duct, not rocket-specific."""
    if inlet_d <= 0 or outlet_d <= 0 or length_mm <= 0 or wall_mm <= 0:
        raise ValueError("inlet_d, outlet_d, length_mm, wall_mm must all be positive")
    if contour not in ("conical", "bell"):
        raise ValueError("contour must be 'conical' or 'bell'")

    r_in, r_out = inlet_d / 2.0, outlet_d / 2.0
    warnings = []
    if wall_mm >= min(r_in, r_out):
        warnings.append("wall thickness exceeds the smaller radius — duct may be solid, not hollow")

    def sdf(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        zc = np.clip(z, 0.0, length_mm)
        frac = zc / length_mm
        ease = frac if contour == "conical" else 0.5 - 0.5 * np.cos(np.pi * frac)
        r_outer = r_in + (r_out - r_in) * ease
        r_inner = np.maximum(r_outer - wall_mm, 0.0)
        rad = np.sqrt(x ** 2 + y ** 2)
        d_radial = np.maximum(rad - r_outer, r_inner - rad)
        d_axial = np.maximum(-z, z - length_mm)
        return np.maximum(d_radial, d_axial)

    max_r = max(r_in, r_out) + 1.0
    bounds = ((-max_r, max_r), (-max_r, max_r), (-1.0, length_mm + 1.0))

    avg_r_outer = (r_in + r_out) / 2.0
    avg_r_inner = max(avg_r_outer - wall_mm, 0.0)
    volume_mm3 = np.pi * (avg_r_outer ** 2 - avg_r_inner ** 2) * length_mm

    report = {
        "inlet_d_mm": inlet_d, "outlet_d_mm": outlet_d, "length_mm": length_mm,
        "wall_mm": wall_mm, "contour": contour,
        "area_ratio_outlet_to_inlet": (r_out / r_in) ** 2,
        "material_volume_mm3": float(volume_mm3),
    }
    return {"sdf": sdf, "bounds": bounds, "report": report, "warnings": warnings}


def heat_sink(power_w: float, ambient_c: float, target_c: float, base_mm: float = 40.0,
              h_coefficient: float = 10.0) -> dict:
    """Sizes a finned block from a natural-convection heat balance:
    required surface area A = power / (h * dT), h a fixed representative
    natural-convection coefficient for air (~10 W/m^2K — a rough default,
    not measured for this specific geometry/orientation). Fin count comes
    from a natural-convection channel-spacing rule of thumb (~8mm between
    fins); fin height is solved to make up whatever surface area the base
    plate alone doesn't cover."""
    dT = target_c - ambient_c
    if dT <= 0:
        raise ValueError("target_c must be greater than ambient_c")
    if power_w <= 0 or base_mm <= 0 or h_coefficient <= 0:
        raise ValueError("power_w, base_mm, h_coefficient must all be positive")

    base_thick_mm = 3.0
    fin_thick_mm = 1.5
    spacing_mm = 8.0
    n_fins = max(1, int(base_mm // spacing_mm))

    area_required_m2 = power_w / (h_coefficient * dT)
    base_area_m2 = (base_mm / 1000.0) ** 2
    extra_area_m2 = max(0.0, area_required_m2 - base_area_m2)
    per_fin_area_m2 = extra_area_m2 / n_fins
    fin_height_mm = max(3.0, (per_fin_area_m2 / (2 * (base_mm / 1000.0))) * 1000.0)

    warnings = []
    if fin_height_mm > 150.0:
        warnings.append("required fin height is very large — this power/dT target may not be "
                         "reachable with natural convection alone; consider a fan (forced convection)")

    half_base = base_mm / 2.0
    if n_fins > 1:
        fin_xs = np.linspace(-half_base + spacing_mm / 2, half_base - spacing_mm / 2, n_fins)
    else:
        fin_xs = np.array([0.0])

    def sdf(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        field = np.maximum(np.maximum(np.abs(x) - half_base, np.abs(y) - half_base),
                            np.abs(z - base_thick_mm / 2) - base_thick_mm / 2)
        fin_center_z = base_thick_mm + fin_height_mm / 2
        for fx in fin_xs:
            fin = np.maximum(np.maximum(np.abs(x - fx) - fin_thick_mm / 2, np.abs(y) - half_base),
                              np.abs(z - fin_center_z) - fin_height_mm / 2)
            field = np.minimum(field, fin)
        return field

    total_height = base_thick_mm + fin_height_mm
    bounds = ((-half_base - 1.0, half_base + 1.0), (-half_base - 1.0, half_base + 1.0),
              (-1.0, total_height + 1.0))

    volume_mm3 = (base_mm ** 2 * base_thick_mm) + n_fins * (fin_thick_mm * base_mm * fin_height_mm)
    density_kg_m3 = _MATERIALS["6061-t6 aluminum"]["density_kg_m3"]
    mass_g = volume_mm3 * 1e-9 * density_kg_m3 * 1e3

    report = {
        "power_w": power_w, "ambient_c": ambient_c, "target_c": target_c,
        "assumed_h_w_m2k": h_coefficient, "required_area_m2": area_required_m2,
        "n_fins": n_fins, "fin_height_mm": fin_height_mm, "base_thick_mm": base_thick_mm,
        "mass_g": mass_g,
    }
    return {"sdf": sdf, "bounds": bounds, "report": report, "warnings": warnings}


_DESIGN_FUNCTIONS = {"bracket": cantilever_bracket, "nozzle": duct_nozzle, "heatsink": heat_sink}

_DEFAULTS = {
    "bracket": dict(load_n=500.0, span_mm=100.0, width_mm=20.0, safety_factor=2.0),
    "nozzle": dict(inlet_d=20.0, outlet_d=40.0, length_mm=60.0, wall_mm=2.0),
    "heatsink": dict(power_w=10.0, ambient_c=25.0, target_c=60.0, base_mm=40.0),
}

_RESOLUTION_MM = 1.0


def format_report(kind: str, report: dict, warnings: list[str], path: str) -> str:
    lines = [f"Design: {kind} — computed parameters:"]
    for k, v in report.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.3g}")
        else:
            lines.append(f"  {k}: {v}")
    for w in warnings:
        lines.append(f"  warning: {w}")
    lines.append(f"STL written to: {path}")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# --------------------------------------------------------------------- skill

_PARAM_RE = re.compile(r"([a-zA-Z_]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+)")

_TRIGGERS = {
    "design a bracket": "bracket", "design bracket": "bracket",
    "design a nozzle": "nozzle", "design nozzle": "nozzle",
    "design a duct": "nozzle", "design duct": "nozzle",
    "design a heat sink": "heatsink", "design heat sink": "heatsink",
    "design a heatsink": "heatsink",
}

# Short, natural-language parameter names -> the design function's actual
# keyword args. Both forms are accepted (the exact keyword always works too).
_PARAM_ALIASES = {
    "bracket": {"load": "load_n", "span": "span_mm", "width": "width_mm",
                "sf": "safety_factor", "safety": "safety_factor"},
    "nozzle": {"inlet": "inlet_d", "outlet": "outlet_d", "length": "length_mm", "wall": "wall_mm"},
    "heatsink": {"power": "power_w", "ambient": "ambient_c", "target": "target_c", "base": "base_mm"},
}


def _parse_params(text: str, kind: str) -> dict:
    aliases = _PARAM_ALIASES.get(kind, {})
    raw = {k.lower(): float(v) for k, v in _PARAM_RE.findall(text)}
    return {aliases.get(k, k): v for k, v in raw.items()}


class DesignEngineSkill(SkillModule):
    """Computational-design skill: 'design a bracket load=500N span=150mm'
    -> solves the physics, generates geometry, writes an STL."""

    name = "design_engine"
    description = "generates 3D-printable geometry from parametric engineering physics (bracket/nozzle/heat sink)"
    priority = 8  # informational/file output, same tier as market_analysis — no gating needed

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or _DEFAULT_OUTPUT_DIR

    @property
    def available(self) -> bool:
        try:
            import skimage  # noqa: F401
            return True
        except ImportError:
            return False

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(trigger) for trigger in _TRIGGERS)

    def handle(self, text: str) -> str:
        t = text.strip().lower()
        kind = None
        for trigger, k in _TRIGGERS.items():
            if t.startswith(trigger):
                kind = k
                break
        if kind is None:
            return "Design what — a bracket, a nozzle/duct, or a heat sink?"

        params = dict(_DEFAULTS[kind])
        params.update(_parse_params(text, kind))
        try:
            result = _DESIGN_FUNCTIONS[kind](**params)
        except Exception as e:
            return f"Couldn't design that {kind} ({e})."

        grid, spacing, origin = voxelize(result["sdf"], result["bounds"], _RESOLUTION_MM)
        try:
            verts, faces = to_mesh(grid, spacing)
        except ValueError as e:
            return f"Couldn't build geometry for that {kind} ({e})."
        verts = verts + np.array(origin)

        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.stl"
        path = write_stl(os.path.join(self.output_dir, filename), verts, faces)

        return format_report(kind, result["report"], result["warnings"], path)
