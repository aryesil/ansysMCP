"""
Support and load specifications
-------------------------------
Turns the plain dictionaries an AI client sends (``extra_supports`` and
``extra_loads``) into validated boundary conditions. Every field a physical
setup needs must be present: nothing is guessed, and an unknown key is an error
so a misspelt option cannot be silently ignored.
"""

from typing import Any, Dict, List, Optional

from ..domain.models import BoundaryCondition, OrthotropicElasticity, Quantity, SemanticSelector
from .input_checks import InputError, require_finite, require_positive, require_selection, unit_vector

STANDARD_GRAVITY_M_S2 = 9.80665

_FACE_KEYS = {"face_normal", "face_axis_extreme", "all_matching", "both_directions"}

SUPPORT_TYPES = {
    "fixed": {"type"} | _FACE_KEYS,
    "frictionless": {"type"} | _FACE_KEYS,
    "displacement": {"type", "ux_mm", "uy_mm", "uz_mm"} | _FACE_KEYS,
    "cylindrical": {"type", "radial", "axial", "tangential", "face_area_range_mm2", "face_centroid_mm"} | _FACE_KEYS,
    "compression_only": {"type"} | _FACE_KEYS,
}
LOAD_TYPES = {
    "force": {"type", "magnitude_n", "direction"} | _FACE_KEYS,
    "pressure": {"type", "magnitude_mpa"} | _FACE_KEYS,
    "moment": {"type", "magnitude_nm", "axis"} | _FACE_KEYS,
    "remote_force": {"type", "magnitude_n", "direction", "origin_mm"} | _FACE_KEYS,
    "gravity": {"type", "magnitude_m_s2", "direction"},
}


def _check_keys(kind: str, label: str, spec: Dict[str, Any], allowed: Dict[str, set]) -> str:
    if not isinstance(spec, dict):
        raise InputError("%s bir sözlük olmalıdır, verilen: %r" % (label, spec))
    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type not in allowed:
        raise InputError("%s: bilinmeyen %s türü %r. Geçerli: %s" % (label, kind, spec.get("type"), sorted(allowed)))
    unknown = sorted(set(spec) - allowed[spec_type])
    if unknown:
        raise InputError(
            "%s (%s): tanınmayan anahtar(lar) %s. Geçerli anahtarlar: %s"
            % (label, spec_type, unknown, sorted(allowed[spec_type]))
        )
    return spec_type


def _selector(label: str, name: str, spec: Dict[str, Any]) -> SemanticSelector:
    area_range = spec.get("face_area_range_mm2")
    centroid = spec.get("face_centroid_mm")
    curved = area_range is not None or centroid is not None
    if curved:
        # A curved face has no single normal; it is picked by its area and axis point.
        if area_range is None or centroid is None:
            raise InputError("%s: eğri yüz için face_area_range_mm2 ve face_centroid_mm birlikte verilmelidir." % label)
        if (not isinstance(area_range, (list, tuple)) or len(area_range) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in area_range)
                or area_range[0] >= area_range[1]):
            raise InputError("%s.face_area_range_mm2 [en_az, en_çok] biçiminde pozitif sayılar olmalıdır." % label)
        if (not isinstance(centroid, (list, tuple)) or len(centroid) != 3
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in centroid)):
            raise InputError("%s.face_centroid_mm [x, y, z] (mm) olmalıdır." % label)
        if spec.get("face_normal") is not None or spec.get("face_axis_extreme") is not None:
            raise InputError("%s: eğri yüz seçimi face_normal / face_axis_extreme ile birlikte kullanılamaz." % label)
        normal, flt = None, None
    else:
        normal, flt = require_selection(label, spec.get("face_normal"), spec.get("face_axis_extreme"))
    multi = spec.get("all_matching", False)
    if not isinstance(multi, bool):
        raise InputError("%s.all_matching true/false olmalıdır, verilen: %r" % (label, multi))
    both = spec.get("both_directions", False)
    if not isinstance(both, bool):
        raise InputError("%s.both_directions true/false olmalıdır, verilen: %r" % (label, both))
    if both and normal is None:
        raise InputError("%s.both_directions için face_normal gerekir." % label)
    return SemanticSelector(
        name=name, normal_vector=normal, position_filter=flt, confidence_threshold=0.85,
        multi_face=multi, either_direction=both,
        area_range_mm2=[float(v) for v in area_range] if curved else None,
        centroid_hint=[float(v) for v in centroid] if curved else None,
    )


def build_support(spec: Dict[str, Any], index: int) -> BoundaryCondition:
    label = "extra_supports[%d]" % index
    kind = _check_keys("mesnet", label, spec, SUPPORT_TYPES)
    selector = _selector(label, "Mesnet_Yuzeyi_%d" % (index + 2), spec)
    name = "Mesnet_%d_%s" % (index + 2, kind)

    if kind == "fixed":
        return BoundaryCondition(name=name, type="fixed_support", target_selector=selector)
    if kind == "frictionless":
        return BoundaryCondition(name=name, type="frictionless_support", target_selector=selector)
    if kind == "compression_only":
        return BoundaryCondition(name=name, type="compression_only_support", target_selector=selector)
    if kind == "cylindrical":
        dofs = {}
        for key in ("radial", "axial", "tangential"):
            value = spec.get(key)
            if value not in ("fixed", "free"):
                raise InputError("%s.%s 'fixed' ya da 'free' olmalıdır, verilen: %r" % (label, key, value))
            dofs[key] = value
        if "fixed" not in dofs.values():
            raise InputError("%s: en az bir yön 'fixed' olmalıdır; hepsi serbest bir mesnet değildir." % label)
        return BoundaryCondition(name=name, type="cylindrical_support", target_selector=selector, dofs=dofs)

    comps: List[Optional[float]] = []
    for key in ("ux_mm", "uy_mm", "uz_mm"):
        comps.append(None if spec.get(key) is None else require_finite("%s.%s" % (label, key), spec[key]))
    if all(c is None for c in comps):
        raise InputError("%s: en az bir bileşen (ux_mm, uy_mm, uz_mm) verilmelidir." % label)
    return BoundaryCondition(name=name, type="displacement", target_selector=selector, components_mm=comps)


def build_load(spec: Dict[str, Any], index: int) -> BoundaryCondition:
    label = "extra_loads[%d]" % index
    kind = _check_keys("yük", label, spec, LOAD_TYPES)
    name = "Yuk_%d_%s" % (index + 2, kind)

    if kind == "gravity":
        # Standard gravity is a physical constant, so it may be omitted; the
        # direction it acts in is the user's to state.
        g = spec.get("magnitude_m_s2")
        g = STANDARD_GRAVITY_M_S2 if g is None else require_positive("%s.magnitude_m_s2" % label, g)
        direction = unit_vector("%s.direction" % label, spec.get("direction"))
        return BoundaryCondition(
            name=name, type="gravity", target_selector=SemanticSelector(name="Tum_Model"),
            magnitude=Quantity(value=g, unit="m/s^2"), direction=direction,
        )

    selector = _selector(label, "Yuk_Yuzeyi_%d" % (index + 2), spec)
    if kind == "pressure":
        p = require_positive("%s.magnitude_mpa" % label, spec.get("magnitude_mpa"))
        return BoundaryCondition(
            name=name, type="pressure", target_selector=selector, magnitude=Quantity(value=p, unit="MPa")
        )
    if kind == "moment":
        m = require_positive("%s.magnitude_nm" % label, spec.get("magnitude_nm"))
        axis = unit_vector("%s.axis" % label, spec.get("axis"))
        return BoundaryCondition(
            name=name, type="moment", target_selector=selector,
            magnitude=Quantity(value=m, unit="N*m"), direction=axis,
        )

    f = require_positive("%s.magnitude_n" % label, spec.get("magnitude_n"))
    direction = unit_vector("%s.direction" % label, spec.get("direction"))
    if kind == "force":
        return BoundaryCondition(
            name=name, type="force", target_selector=selector,
            magnitude=Quantity(value=f, unit="N"), direction=direction,
        )
    origin = spec.get("origin_mm")
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise InputError("%s.origin_mm üç bileşenli [x, y, z] olmalıdır." % label)
    origin = [require_finite("%s.origin_mm[%d]" % (label, i), c) for i, c in enumerate(origin)]
    return BoundaryCondition(
        name=name, type="remote_force", target_selector=selector,
        magnitude=Quantity(value=f, unit="N"), direction=direction, origin_mm=origin,
    )


ORTHOTROPIC_KEYS = (
    "ex_gpa", "ey_gpa", "ez_gpa", "nu_xy", "nu_yz", "nu_xz", "gxy_gpa", "gyz_gpa", "gxz_gpa",
)


def build_orthotropic(spec: Dict[str, Any]) -> OrthotropicElasticity:
    """All nine constants are required; a missing or extra key is an error."""
    if not isinstance(spec, dict):
        raise InputError("orthotropic bir sözlük olmalıdır, verilen: %r" % (spec,))
    missing = [k for k in ORTHOTROPIC_KEYS if spec.get(k) is None]
    if missing:
        raise InputError("orthotropic için eksik sabit(ler): %s" % ", ".join(missing))
    unknown = sorted(set(spec) - set(ORTHOTROPIC_KEYS))
    if unknown:
        raise InputError("orthotropic içinde tanınmayan anahtar(lar): %s" % unknown)
    value = {k: require_finite("orthotropic.%s" % k, spec[k]) for k in ORTHOTROPIC_KEYS}
    return OrthotropicElasticity(
        ex=Quantity(value=value["ex_gpa"], unit="GPa"),
        ey=Quantity(value=value["ey_gpa"], unit="GPa"),
        ez=Quantity(value=value["ez_gpa"], unit="GPa"),
        nu_xy=value["nu_xy"], nu_yz=value["nu_yz"], nu_xz=value["nu_xz"],
        gxy=Quantity(value=value["gxy_gpa"], unit="GPa"),
        gyz=Quantity(value=value["gyz_gpa"], unit="GPa"),
        gxz=Quantity(value=value["gxz_gpa"], unit="GPa"),
    )


def build_layers(layers: Any) -> List[Dict[str, float]]:
    """Validated ply list: each ply needs a positive thickness and an angle; nothing is defaulted."""
    if not isinstance(layers, (list, tuple)) or not (1 <= len(layers) <= 100):
        raise InputError("layers 1 ile 100 arasında ply içeren bir liste olmalıdır.")
    out: List[Dict[str, float]] = []
    for i, ply in enumerate(layers):
        label = "layers[%d]" % i
        if not isinstance(ply, dict):
            raise InputError("%s bir sözlük olmalıdır." % label)
        unknown = sorted(set(ply) - {"thickness_mm", "angle_deg"})
        if unknown:
            raise InputError("%s: tanınmayan anahtar(lar) %s; geçerli: thickness_mm, angle_deg." % (label, unknown))
        if ply.get("thickness_mm") is None or ply.get("angle_deg") is None:
            raise InputError("%s için thickness_mm ve angle_deg birlikte verilmelidir." % label)
        thickness = require_positive("%s.thickness_mm" % label, ply["thickness_mm"])
        angle = require_finite("%s.angle_deg" % label, ply["angle_deg"])
        if not (-180.0 <= angle <= 180.0):
            raise InputError("%s.angle_deg -180 ile 180 arasında olmalıdır." % label)
        out.append({"thickness_mm": thickness, "angle_deg": angle})
    return out


PLY_STRENGTH_REQUIRED = ("xt_mpa", "xc_mpa", "yt_mpa", "yc_mpa", "s12_mpa")
PLY_STRENGTH_OPTIONAL = ("zt_mpa", "zc_mpa", "s23_mpa", "s13_mpa")


def build_ply_strength(spec: Any) -> "PlyStrength":
    """Ply strengths in MPa as positive magnitudes: in-plane limits are required, the rest only when stated."""
    from ..domain.models import PlyStrength

    if not isinstance(spec, dict):
        raise InputError("ply_strengths_mpa bir sözlük olmalıdır, verilen: %r" % (spec,))
    missing = [k for k in PLY_STRENGTH_REQUIRED if spec.get(k) is None]
    if missing:
        raise InputError("ply_strengths_mpa için eksik değer(ler): %s" % ", ".join(missing))
    unknown = sorted(set(spec) - set(PLY_STRENGTH_REQUIRED) - set(PLY_STRENGTH_OPTIONAL))
    if unknown:
        raise InputError("ply_strengths_mpa içinde tanınmayan anahtar(lar): %s" % unknown)
    values = {k: require_positive("ply_strengths_mpa.%s" % k, v) for k, v in spec.items() if v is not None}
    q = lambda key: Quantity(value=values[key], unit="MPa") if key in values else None
    return PlyStrength(
        xt=q("xt_mpa"), xc=q("xc_mpa"), yt=q("yt_mpa"), yc=q("yc_mpa"), s12=q("s12_mpa"),
        zt=q("zt_mpa"), zc=q("zc_mpa"), s23=q("s23_mpa"), s13=q("s13_mpa"),
    )

