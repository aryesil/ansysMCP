"""
Closed-form cross-checks
------------------------
Textbook solutions an FEA result can be compared against. A solver number that
lands within a few percent of the analytic answer is evidence the model is set
up correctly; one that does not is a reason to look at the supports, the units
or the mesh before trusting anything else.

Every function takes SI-consistent inputs in the units named in its arguments
(mm, N, MPa) and returns the figures it computed, so the caller can show the
working.
"""

import math
from typing import Any, Dict

from ..validation.input_checks import require_positive, require_finite, InputError


def _section(shape: str, p: Dict[str, float]) -> Dict[str, float]:
    """Area, second moment about the bending axis and outer fibre distance (mm units)."""
    if shape == "solid_round":
        d = require_positive("outer_diameter_mm", p.get("outer_diameter_mm"))
        return {"area": math.pi * d * d / 4.0, "inertia": math.pi * d ** 4 / 64.0, "c": d / 2.0}
    if shape == "hollow_round":
        do = require_positive("outer_diameter_mm", p.get("outer_diameter_mm"))
        di = require_positive("inner_diameter_mm", p.get("inner_diameter_mm"))
        if di >= do:
            raise InputError("inner_diameter_mm, outer_diameter_mm değerinden küçük olmalıdır.")
        return {
            "area": math.pi * (do * do - di * di) / 4.0,
            "inertia": math.pi * (do ** 4 - di ** 4) / 64.0,
            "c": do / 2.0,
        }
    if shape == "rectangle":
        b = require_positive("width_mm", p.get("width_mm"))
        h = require_positive("height_mm", p.get("height_mm"))
        return {"area": b * h, "inertia": b * h ** 3 / 12.0, "c": h / 2.0}
    raise InputError("shape geçersiz: %r (solid_round, hollow_round, rectangle)" % shape)


def cantilever_end_load(
    shape: str, length_mm: float, force_n: float, youngs_modulus_gpa: float, **section: float
) -> Dict[str, Any]:
    """Tip deflection and root bending stress of a cantilever with an end load."""
    length = require_positive("length_mm", length_mm)
    force = require_positive("force_n", abs(require_finite("force_n", force_n)))
    e_mpa = require_positive("youngs_modulus_gpa", youngs_modulus_gpa) * 1000.0
    s = _section(shape, section)
    deflection = force * length ** 3 / (3.0 * e_mpa * s["inertia"])
    stress = force * length * s["c"] / s["inertia"]
    return {
        "model": "cantilever_end_load",
        "deflection_mm": deflection,
        "max_bending_stress_mpa": stress,
        "root_moment_nmm": force * length,
        "section": s,
        "formulas": "delta = F L^3 / (3 E I);  sigma = F L c / I",
    }


def axial_bar(
    shape: str, length_mm: float, force_n: float, youngs_modulus_gpa: float, **section: float
) -> Dict[str, Any]:
    length = require_positive("length_mm", length_mm)
    force = require_finite("force_n", force_n)
    e_mpa = require_positive("youngs_modulus_gpa", youngs_modulus_gpa) * 1000.0
    s = _section(shape, section)
    return {
        "model": "axial_bar",
        "elongation_mm": force * length / (e_mpa * s["area"]),
        "axial_stress_mpa": force / s["area"],
        "section": s,
        "formulas": "delta = F L / (E A);  sigma = F / A",
    }


def simply_supported_center_load(
    shape: str, length_mm: float, force_n: float, youngs_modulus_gpa: float, **section: float
) -> Dict[str, Any]:
    length = require_positive("length_mm", length_mm)
    force = require_positive("force_n", abs(require_finite("force_n", force_n)))
    e_mpa = require_positive("youngs_modulus_gpa", youngs_modulus_gpa) * 1000.0
    s = _section(shape, section)
    return {
        "model": "simply_supported_center_load",
        "deflection_mm": force * length ** 3 / (48.0 * e_mpa * s["inertia"]),
        "max_bending_stress_mpa": force * length / 4.0 * s["c"] / s["inertia"],
        "section": s,
        "formulas": "delta = F L^3 / (48 E I);  sigma = (F L / 4) c / I",
    }


def thin_wall_pressure_vessel(
    internal_pressure_mpa: float, mean_radius_mm: float, wall_thickness_mm: float
) -> Dict[str, Any]:
    p = require_positive("internal_pressure_mpa", internal_pressure_mpa)
    r = require_positive("mean_radius_mm", mean_radius_mm)
    t = require_positive("wall_thickness_mm", wall_thickness_mm)
    if r / t < 10.0:
        raise InputError("İnce cidar kabulü r/t >= 10 ister (verilen %.1f); kalın cidar çözümü gerekir." % (r / t))
    return {
        "model": "thin_wall_pressure_vessel",
        "hoop_stress_mpa": p * r / t,
        "axial_stress_mpa": p * r / (2.0 * t),
        "formulas": "sigma_hoop = p r / t;  sigma_axial = p r / (2 t)",
    }


def plate_with_hole(
    far_field_stress_mpa: float, plate_width_mm: float, hole_diameter_mm: float
) -> Dict[str, Any]:
    """Peterson's finite-width stress concentration for a central hole, tension loaded."""
    s0 = require_finite("far_field_stress_mpa", far_field_stress_mpa)
    w = require_positive("plate_width_mm", plate_width_mm)
    d = require_positive("hole_diameter_mm", hole_diameter_mm)
    ratio = d / w
    if ratio >= 0.65:
        raise InputError("d/w=%.2f; bağıntı d/w < 0.65 için geçerlidir." % ratio)
    kt = 3.0 - 3.14 * ratio + 3.667 * ratio ** 2 - 1.527 * ratio ** 3
    net = s0 / (1.0 - ratio)
    return {
        "model": "plate_with_hole",
        "kt_on_net_section": kt,
        "net_section_stress_mpa": net,
        "peak_stress_mpa": kt * net,
        "formulas": "Kt = 3 - 3.14(d/w) + 3.667(d/w)^2 - 1.527(d/w)^3 (Peterson, on the net section)",
    }


MODELS = {
    "cantilever_end_load": cantilever_end_load,
    "axial_bar": axial_bar,
    "simply_supported_center_load": simply_supported_center_load,
    "thin_wall_pressure_vessel": thin_wall_pressure_vessel,
    "plate_with_hole": plate_with_hole,
}


def deviation_pct(fea_value: float, analytic_value: float) -> float:
    """Signed difference of the FEA figure from the analytic one, in percent."""
    if analytic_value == 0.0:
        raise InputError("Analitik değer sıfır; yüzde sapma tanımsız.")
    return (fea_value - analytic_value) / abs(analytic_value) * 100.0
