"""
Orients a STEP body to the wind frame.

The Fluent set-up always runs the free stream along +X with lift along +Y.  To
analyse any other free-stream direction the body is turned instead of the tunnel:
every point and direction in the STEP file is rotated by one proper rotation, so
the B-rep stays exact (radii, surface types and topology are untouched).  The
result is checked against the original by volume and bounding box.
"""

import math
import os
import re
from typing import Dict, List, Sequence, Tuple

Matrix = List[List[float]]

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TUPLE = r"\(\s*(%s(?:\s*,\s*%s)*)\s*\)" % (_NUM, _NUM)
_NAME = r"'(?:[^']|'')*'"
_ENTITY = re.compile(
    r"\b(CARTESIAN_POINT|DIRECTION)\s*\(\s*(%s)\s*,\s*%s\s*\)" % (_NAME, _TUPLE), re.DOTALL
)


_DEF = re.compile(r"#(\d+)\s*=\s*(CARTESIAN_POINT|DIRECTION)\s*\(\s*%s\s*,\s*%s\s*\)" % (_NAME, _TUPLE), re.DOTALL)
_PLACEMENT = re.compile(
    r"(#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\(\s*%s\s*,\s*)#(\d+)(\s*,\s*)#(\d+)(\s*,\s*)#(\d+)(\s*\))" % _NAME, re.DOTALL
)
_REPRESENTATION = re.compile(
    r"(?:ADVANCED_BREP_)?SHAPE_REPRESENTATION\s*\(\s*%s\s*,\s*\(([^)]*)\)" % _NAME, re.DOTALL
)
NL_JOIN = chr(10)
_ASSEMBLY_MARKERS = ("ITEM_DEFINED_TRANSFORMATION", "MAPPED_ITEM", "NEXT_ASSEMBLY_USAGE_OCCURRENCE")


class StepRotateError(ValueError):
    pass


def _unit(name: str, v: Sequence[float]) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(
        isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in v
    ):
        raise StepRotateError("%s [x, y, z] biçiminde 3 sonlu sayı olmalıdır." % name)
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise StepRotateError("%s sıfır vektör olamaz." % name)
    return [c / n for c in v]


def wind_frame(flow_direction: Sequence[float], lift_direction: Sequence[float]) -> Matrix:
    """Rows are the wind axes in the CAD frame: drag (flow), lift, side = flow x lift.

    The rotation sends the flow to +X and the lift to +Y.  The lift hint need not be
    exactly perpendicular to the flow: its component along the flow is removed.
    """
    d = _unit("flow_direction", flow_direction)
    l_raw = _unit("lift_direction", lift_direction)
    dot = sum(a * b for a, b in zip(d, l_raw))
    lift = [b - dot * a for a, b in zip(d, l_raw)]
    norm = math.sqrt(sum(c * c for c in lift))
    if norm < 1e-6:
        raise StepRotateError("lift_direction akış yönüne paralel olamaz.")
    lift = [c / norm for c in lift]
    side = [d[1] * lift[2] - d[2] * lift[1], d[2] * lift[0] - d[0] * lift[2], d[0] * lift[1] - d[1] * lift[0]]
    return [d, lift, side]


def is_identity(matrix: Matrix, tol: float = 1e-12) -> bool:
    return all(abs(matrix[i][j] - (1.0 if i == j else 0.0)) <= tol for i in range(3) for j in range(3))


def apply(matrix: Matrix, v: Sequence[float]) -> List[float]:
    return [sum(matrix[i][j] * v[j] for j in range(3)) for i in range(3)]


def rotate_mesh_file(src: str, dst: str, matrix: Matrix) -> Dict[str, int]:
    """Turn a faceted body (STL/OBJ/PLY) by ``matrix`` and write a binary STL."""
    import numpy as np
    import trimesh

    if not os.path.exists(src):
        raise StepRotateError("Geometri dosyası bulunamadı: %s" % src)
    mesh = trimesh.load(src, force="mesh")
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        raise StepRotateError("Geometri dosyasından üçgen okunamadı: %s" % src)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(matrix, dtype=float)
    mesh.apply_transform(transform)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    mesh.export(dst, file_type="stl")
    return {"points": int(len(mesh.vertices)), "directions": int(len(mesh.faces))}


def rotate_step(src: str, dst: str, matrix: Matrix) -> Dict[str, int]:
    """Write ``dst`` with every 3-D point and direction of ``src`` rotated by ``matrix``."""
    if not os.path.exists(src):
        raise StepRotateError("STEP dosyası bulunamadı: %s" % src)
    if os.path.splitext(src)[1].lower() not in (".step", ".stp"):
        raise StepRotateError("STEP döndürme yalnızca .step/.stp dosyalarıyla çalışır; ağ dosyaları için rotate_mesh_file kullanın.")
    with open(src, "r", encoding="latin-1", newline="") as handle:
        text = handle.read()

    # An assembly places its parts with frame-to-frame transformations.  Turning every point,
    # direction and frame by the same rotation keeps those relations (the rotation cancels in
    # a frame-to-frame map), so the assembly is turned uniformly and the result is then checked
    # against the original by ``verify_rotation``; a reader that treats the frames differently
    # fails that check instead of producing a wrongly placed body.
    assembly = any(marker in text for marker in _ASSEMBLY_MARKERS)

    # The representation's own placement is a frame the readers may apply on top of the
    # geometry; turning the geometry as well would turn the body twice.  It is only
    # safe to leave when it is the identity frame, so any other frame is refused.
    defs = {int(m.group(1)): [float(x) for x in m.group(3).split(",")] for m in _DEF.finditer(text)}
    top_level = set()
    for rep in _REPRESENTATION.finditer(text):
        top_level.update(int(i) for i in re.findall(r"#(\d+)", rep.group(1)))
    next_id = max(list(defs) + [int(i) for i in re.findall(r"^#(\d+)\s*=", text, re.M)] + [0]) + 1
    appended: List[str] = []

    def neutralise(match: "re.Match") -> str:
        nonlocal next_id
        if int(match.group(2)) not in top_level:
            return match.group(0)
        loc, axis, ref = (defs.get(int(match.group(i))) for i in (3, 5, 7))
        identity = (
            loc is not None and axis is not None and ref is not None
            and max(abs(c) for c in loc) < 1e-9 and abs(axis[2] - 1.0) < 1e-9 and abs(ref[0] - 1.0) < 1e-9
            and abs(axis[0]) + abs(axis[1]) + abs(ref[1]) + abs(ref[2]) < 1e-9
        )
        if not identity:
            raise StepRotateError("Gövde gösteriminin yerel eksen takımı birim değil; bu STEP güvenle döndürülemez.")
        ids = (next_id, next_id + 1, next_id + 2)
        next_id += 3
        appended.append("#%d = CARTESIAN_POINT('',(0.,0.,0.));" % ids[0])
        appended.append("#%d = DIRECTION('',(0.,0.,1.));" % ids[1])
        appended.append("#%d = DIRECTION('',(1.,0.,0.));" % ids[2])
        g = match.group
        return "%s#%d%s#%d%s#%d%s" % (g(1), ids[0], g(4), ids[1], g(6), ids[2], g(8))

    if not assembly:
        text = _PLACEMENT.sub(neutralise, text)

    counts = {"points": 0, "directions": 0}

    def swap(match: "re.Match") -> str:
        kind, name = match.group(1), match.group(2)
        values = [float(x) for x in match.group(3).split(",")]
        if len(values) != 3:
            return match.group(0)                       # 2-D parameter-space entry
        x, y, z = apply(matrix, values)
        counts["points" if kind == "CARTESIAN_POINT" else "directions"] += 1
        return "%s(%s,(%.15g,%.15g,%.15g))" % (kind, name, x, y, z)

    out = _ENTITY.sub(swap, text)
    if appended:
        end = out.rfind("ENDSEC;")
        out = out[:end] + NL_JOIN.join(appended) + NL_JOIN + out[end:]
    if counts["points"] == 0 or counts["directions"] == 0:
        raise StepRotateError("STEP dosyasında döndürülecek 3-B nokta/yön bulunamadı; dosya okunamadı.")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="latin-1", newline="") as handle:
        handle.write(out)
    return counts


def verify_rotation(src: str, dst: str, matrix: Matrix, volume_tol: float = 5e-3, length_tol_mm: float = 0.5) -> Dict[str, float]:
    """Compare the turned file against the original: same volume, bounding box of the turned points."""
    from .cad_parser import CadParser

    a = CadParser.load_mesh(src)
    b = CadParser.load_mesh(dst)
    if abs(b.volume - a.volume) > volume_tol * abs(a.volume):
        raise StepRotateError("Döndürülen gövdenin hacmi değişti (%.6g -> %.6g mm3)." % (a.volume, b.volume))
    rotated = [apply(matrix, v) for v in a.vertices]
    lo = [min(p[i] for p in rotated) for i in range(3)]
    hi = [max(p[i] for p in rotated) for i in range(3)]
    for i in range(3):
        if abs(b.bounds[0][i] - lo[i]) > length_tol_mm or abs(b.bounds[1][i] - hi[i]) > length_tol_mm:
            raise StepRotateError("Döndürülen gövdenin sınır kutusu beklenenle uyuşmuyor (eksen %d)." % i)
    return {"volume_mm3": float(b.volume), "extents_mm": [float(hi[i] - lo[i]) for i in range(3)]}
