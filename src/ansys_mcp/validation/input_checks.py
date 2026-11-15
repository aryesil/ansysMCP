"""
Input checks
------------
Pure-Python guards applied to everything an AI client hands the server before
it is stored or sent to a solver. They reject what a solver would either
choke on or, worse, accept silently: NaN and infinite numbers, zero-length
direction vectors, physically impossible material cards and unknown face
selectors.

Nothing in here substitutes a value. A bad input produces an error message
naming the argument, so the caller can correct it.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..domain.units import UnitEngine, UnitError

_AXES = ("x", "y", "z")
_EXTREMES = ("min", "max")

# Bounds a real engineering material cannot leave. They are wide on purpose:
# the point is to catch unit slips (200 instead of 200e9 Pa, g/cm3 instead of
# kg/m3), not to second-guess an unusual alloy.
YOUNGS_MODULUS_PA = (1.0e6, 1.0e12)
DENSITY_KG_M3 = (10.0, 25000.0)


class InputError(ValueError):
    """Raised when an argument cannot be used as given."""


def require_finite(name: str, value: Any) -> float:
    """Return ``value`` as a float, or raise if it is missing, NaN or infinite."""
    if value is None:
        raise InputError("'%s' verilmedi." % name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InputError("'%s' sayı olmalıdır, verilen: %r" % (name, value))
    if not math.isfinite(number):
        raise InputError("'%s' sonlu bir sayı olmalıdır (NaN/Inf kabul edilmez), verilen: %r" % (name, value))
    return number


def require_positive(name: str, value: Any) -> float:
    number = require_finite(name, value)
    if number <= 0.0:
        raise InputError("'%s' sıfırdan büyük olmalıdır, verilen: %s" % (name, number))
    return number


def unit_vector(name: str, vector: Optional[Sequence[float]]) -> List[float]:
    """Normalise a 3-vector. A zero or malformed vector is an error, not a default."""
    if vector is None:
        raise InputError("'%s' verilmedi." % name)
    if len(vector) != 3:
        raise InputError("'%s' üç bileşenli olmalıdır [x, y, z], verilen: %r" % (name, list(vector)))
    comps = [require_finite("%s[%d]" % (name, i), c) for i, c in enumerate(vector)]
    norm = math.sqrt(sum(c * c for c in comps))
    if norm < 1.0e-9:
        raise InputError("'%s' sıfır vektör olamaz; yön belirsiz." % name)
    return [c / norm for c in comps]


def parse_axis_extreme(name: str, spec: Optional[str]) -> Optional[Dict[str, str]]:
    """Turn 'min_x' into a position filter. ``None`` when nothing was given.

    A string that does not parse is an error: falling back to some other face
    would put a support or a load somewhere the caller never asked for.
    """
    if spec is None or str(spec).strip() == "":
        return None
    token = str(spec).strip().lower().replace("-", "_")
    parts = token.split("_")
    if len(parts) != 2 or parts[0] not in _EXTREMES or parts[1] not in _AXES:
        raise InputError(
            "'%s' geçersiz: %r. Beklenen biçim 'min_x', 'max_x', 'min_y', 'max_y', 'min_z' veya 'max_z'."
            % (name, spec)
        )
    return {"axis": parts[1], "extreme": parts[0]}


def require_selection(
    label: str,
    normal: Optional[Sequence[float]],
    axis_extreme: Optional[str],
) -> Tuple[Optional[List[float]], Optional[Dict[str, str]]]:
    """A face must be named by a normal, an axis extreme, or both."""
    flt = parse_axis_extreme("%s_axis_extreme" % label, axis_extreme)
    nrm = unit_vector("%s_normal" % label, normal) if normal else None
    if nrm is None and flt is None:
        raise InputError(
            "%s yüzeyi belirtilmedi: '%s_normal' veya '%s_axis_extreme' verin. "
            "Yüzleri görmek için 'inspect_live_geometry_faces' kullanın."
            % (label, label, label)
        )
    return nrm, flt


def check_material_card(
    youngs_modulus: Any,
    poissons_ratio: float,
    yield_strength: Any,
    ultimate_strength: Any = None,
    density: Any = None,
    orthotropic: bool = False,
) -> List[str]:
    """Return one message per physically impossible property; empty when sound.

    Each argument is a ``Quantity`` (value/unit) as used by the domain models.
    """
    problems: List[str] = []

    def si(label: str, quantity: Any) -> Optional[float]:
        try:
            return float(UnitEngine.to_si(quantity).value)
        except (UnitError, TypeError, ValueError) as exc:
            problems.append("%s birim hatası: %s" % (label, exc))
            return None

    e_pa = si("Elastisite modülü", youngs_modulus)
    if e_pa is not None:
        if not math.isfinite(e_pa) or e_pa <= 0.0:
            problems.append("Elastisite modülü sonlu ve pozitif olmalıdır, verilen: %r" % (youngs_modulus,))
        elif not (YOUNGS_MODULUS_PA[0] <= e_pa <= YOUNGS_MODULUS_PA[1]):
            problems.append(
                "Elastisite modülü %.3g Pa gerçek bir malzeme için makul aralığın (%.0e-%.0e Pa) dışında; "
                "birim hatası olabilir (GPa yerine Pa/MPa?)."
                % (e_pa, YOUNGS_MODULUS_PA[0], YOUNGS_MODULUS_PA[1])
            )

    # An orthotropic material has nine independent constants checked by
    # ``check_orthotropic``; the single nu stored beside it is only nu_xy and
    # is not confined to the isotropic range.
    if not math.isfinite(poissons_ratio) or (
        not orthotropic and not (0.0 <= poissons_ratio < 0.5)
    ):
        problems.append("Poisson oranı [0, 0.5) aralığında olmalıdır, verilen: %r" % (poissons_ratio,))

    sy_pa = si("Akma mukavemeti", yield_strength)
    if sy_pa is not None and (not math.isfinite(sy_pa) or sy_pa <= 0.0):
        problems.append("Akma mukavemeti sonlu ve pozitif olmalıdır, verilen: %r" % (yield_strength,))

    if ultimate_strength is not None:
        su_pa = si("Kopma mukavemeti", ultimate_strength)
        if su_pa is not None and sy_pa is not None and su_pa < sy_pa:
            problems.append(
                "Kopma mukavemeti (%.3g Pa) akma mukavemetinden (%.3g Pa) küçük olamaz." % (su_pa, sy_pa)
            )

    if density is not None:
        rho = si("Yoğunluk", density)
        if rho is not None and not (DENSITY_KG_M3[0] <= rho <= DENSITY_KG_M3[1]):
            problems.append(
                "Yoğunluk %.3g kg/m3 makul aralığın (%g-%g kg/m3) dışında; g/cm3 ile kg/m3 karışmış olabilir."
                % (rho, DENSITY_KG_M3[0], DENSITY_KG_M3[1])
            )
    return problems


def safe_label(text: str, limit: int = 80) -> str:
    """A name that is safe to place inside a generated script or XML document."""
    cleaned = "".join(ch if (ch.isalnum() or ch in " _-.") else "_" for ch in str(text))
    return cleaned.strip()[:limit] or "unnamed"


def check_orthotropic(ortho: Any) -> List[str]:
    """Problems with an orthotropic elastic set; empty when it is physically admissible.

    Besides positive moduli, the constants must make the compliance matrix
    positive definite. A set that fails that describes a material that would
    release energy when deformed, and the solver would return nonsense or stop.
    """
    problems: List[str] = []
    try:
        e = [float(UnitEngine.to_si(q).value) for q in (ortho.ex, ortho.ey, ortho.ez)]
        g = [float(UnitEngine.to_si(q).value) for q in (ortho.gxy, ortho.gyz, ortho.gxz)]
    except (UnitError, TypeError, ValueError) as exc:
        return ["Ortotropik sabitlerde birim hatası: %s" % exc]
    nus = [ortho.nu_xy, ortho.nu_yz, ortho.nu_xz]

    for label, value in zip(("Ex", "Ey", "Ez", "Gxy", "Gyz", "Gxz"), e + g):
        if not math.isfinite(value) or not (YOUNGS_MODULUS_PA[0] <= value <= YOUNGS_MODULUS_PA[1]):
            problems.append("%s = %.3g Pa makul aralığın dışında." % (label, value))
    for label, value in zip(("nu_xy", "nu_yz", "nu_xz"), nus):
        if not math.isfinite(value) or abs(value) >= 1.0:
            problems.append("%s = %r geçersiz (|nu| < 1 olmalı)." % (label, value))
    if problems:
        return problems

    ex, ey, ez = e
    gxy, gyz, gxz = g
    nu_xy, nu_yz, nu_xz = nus
    # Compliance matrix of the normal-stress block; symmetry gives nu_ji = nu_ij * E_j / E_i.
    compliance = [
        [1.0 / ex, -nu_xy / ex, -nu_xz / ex],
        [-nu_xy / ex, 1.0 / ey, -nu_yz / ey],
        [-nu_xz / ex, -nu_yz / ey, 1.0 / ez],
    ]
    # Leading principal minors must all be positive (Sylvester).
    d1 = compliance[0][0]
    d2 = compliance[0][0] * compliance[1][1] - compliance[0][1] ** 2
    d3 = (
        compliance[0][0] * (compliance[1][1] * compliance[2][2] - compliance[1][2] ** 2)
        - compliance[0][1] * (compliance[0][1] * compliance[2][2] - compliance[1][2] * compliance[0][2])
        + compliance[0][2] * (compliance[0][1] * compliance[1][2] - compliance[1][1] * compliance[0][2])
    )
    if not (d1 > 0.0 and d2 > 0.0 and d3 > 0.0):
        problems.append(
            "Ortotropik sabitler fiziksel olarak kabul edilemez: uyum matrisi pozitif tanımlı değil "
            "(Poisson oranları modüllerle uyumsuz)."
        )
    return problems
