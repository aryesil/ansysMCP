"""
Classical laminate theory for the layered-section set-up.

Used to describe the laminate the solver is given and to hand-check results.  All inputs
come from the caller; nothing is assumed.  Moduli in GPa, thickness in mm, angles in
degrees (0 degrees = fibre along the layer reference axis).
"""

import math
from typing import Dict, List, Sequence


def _inverse3(m: List[List[float]]) -> List[List[float]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-30:
        raise ValueError("Tabaka rijitlik matrisi tekil.")
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def reduced_stiffness(e1: float, e2: float, g12: float, nu12: float) -> List[List[float]]:
    nu21 = nu12 * e2 / e1
    d = 1.0 - nu12 * nu21
    if d <= 0:
        raise ValueError("Poisson oranları kararlı bir ply vermiyor (1 - nu12 nu21 <= 0).")
    return [[e1 / d, nu12 * e2 / d, 0.0], [nu12 * e2 / d, e2 / d, 0.0], [0.0, 0.0, g12]]


def transformed_stiffness(q: List[List[float]], angle_deg: float) -> List[List[float]]:
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    q11, q12, q22, q66 = q[0][0], q[0][1], q[1][1], q[2][2]
    c2, s2, c4, s4 = c * c, s * s, c ** 4, s ** 4
    b11 = q11 * c4 + 2 * (q12 + 2 * q66) * s2 * c2 + q22 * s4
    b22 = q11 * s4 + 2 * (q12 + 2 * q66) * s2 * c2 + q22 * c4
    b12 = (q11 + q22 - 4 * q66) * s2 * c2 + q12 * (s4 + c4)
    b66 = (q11 + q22 - 2 * q12 - 2 * q66) * s2 * c2 + q66 * (s4 + c4)
    b16 = (q11 - q12 - 2 * q66) * s * c ** 3 + (q12 - q22 + 2 * q66) * s ** 3 * c
    b26 = (q11 - q12 - 2 * q66) * s ** 3 * c + (q12 - q22 + 2 * q66) * s * c ** 3
    return [[b11, b12, b16], [b12, b22, b26], [b16, b26, b66]]


def laminate_summary(plies: Sequence[Dict[str, float]], e1: float, e2: float, g12: float, nu12: float) -> Dict[str, float]:
    """In-plane laminate stiffness (A matrix, N/mm) and the effective moduli it implies."""
    q = reduced_stiffness(e1, e2, g12, nu12)
    a = [[0.0] * 3 for _ in range(3)]
    total = 0.0
    for ply in plies:
        t = float(ply["thickness_mm"])
        qb = transformed_stiffness(q, float(ply["angle_deg"]))
        total += t
        for i in range(3):
            for j in range(3):
                a[i][j] += qb[i][j] * t * 1000.0            # GPa * mm = 1000 N/mm
    inv = _inverse3(a)
    return {
        "total_thickness_mm": total,
        "A11_n_per_mm": a[0][0], "A12_n_per_mm": a[0][1], "A22_n_per_mm": a[1][1], "A66_n_per_mm": a[2][2],
        "A16_n_per_mm": a[0][2], "A26_n_per_mm": a[1][2],
        "ex_effective_gpa": 1.0 / (inv[0][0] * total) / 1000.0,
        "ey_effective_gpa": 1.0 / (inv[1][1] * total) / 1000.0,
        "gxy_effective_gpa": 1.0 / (inv[2][2] * total) / 1000.0,
    }
