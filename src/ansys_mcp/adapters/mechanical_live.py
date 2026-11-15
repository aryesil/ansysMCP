"""
Live PyMechanical Session Driver (ANSYS 2025 R1 / v251)
--------------------------------------------------------
Drives a REAL, running ANSYS Mechanical application over gRPC.

Unlike a script generator, every method in this module mutates an actual
Mechanical session: geometry is imported into the real tree, Engineering Data
receives a real MatML material, the real mesher runs, the real MAPDL solver
executes, and results are read back from the solved model.

When ``visible=True`` the Mechanical GUI is launched on the user's desktop so
each step can be watched as it happens.

Design notes
~~~~~~~~~~~~
* Remote scripts execute inside Mechanical's IronPython 2.7 engine. They are
  therefore written without f-strings, and newlines inside remote string
  expressions are produced with ``chr(10)`` rather than backslash escapes.
* Mechanical 251 without Service Pack 4 does not support secure gRPC, so the
  transport mode falls back to ``insecure`` on loopback.
* ``DataModel.GeoData`` reports lengths in the CAD file's own unit while
  ``Body.Volume`` honours the active unit system. The scale factor between the
  two is measured at runtime instead of assumed.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

from ..domain.models import (
    BoundaryCondition,
    BoundaryConditionType,
    Material,
    MeshConfig,
    Quantity,
    ResultMetric,
    SemanticSelector,
)
from ..domain.units import UnitEngine


DEFAULT_GRPC_PORT = 10000

# Mechanical's gRPC layer serialises the script's return value through the
# machine's ANSI code page, so any non-ASCII character in a CAD-derived body
# name or a localised solver message aborts the call. Every remote script that
# can return model text therefore folds it to ASCII first.
_ASCII_PRELUDE = (
    "def _a(v):" + chr(10)
    + "    try:" + chr(10)
    + "        return v.encode('ascii', 'replace')" + chr(10)
    + "    except Exception:" + chr(10)
    + "        try:" + chr(10)
    + "            return str(v)" + chr(10)
    + "        except Exception:" + chr(10)
    + "            return '?'" + chr(10)
)

# Mesh metric readings come back as either a bare number or a unit-carrying
# Quantity depending on the metric, so both shapes are unwrapped.
_NUM_PRELUDE = (
    "def _n(v):" + chr(10)
    + "    try:" + chr(10)
    + "        return float(v.Value)" + chr(10)
    + "    except Exception:" + chr(10)
    + "        return float(v)" + chr(10)
)


class LiveMechanicalError(RuntimeError):
    """Raised when the real Mechanical session rejects or fails a step."""


def _fmt(value: float) -> str:
    """Render a float for embedding in a remote IronPython script."""
    return repr(float(value))


class LiveMechanicalSession:
    """A real, running ANSYS Mechanical application driven over gRPC."""

    def __init__(self, port: int = DEFAULT_GRPC_PORT, visible: bool = True):
        self.port = port
        self.visible = visible
        self._mechanical = None
        self.step_log: List[Dict[str, Any]] = []
        self._geodata_scale_to_mm: float = 1.0
        self.project_directory: str = ""
        # Index of the analysis system every subsequent call operates on. A
        # model can hold a static structural, a modal and a thermal system at
        # once, so the target must be explicit rather than assumed to be [0].
        self.analysis_index: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        """True only while the remote Mechanical still answers.

        When Mechanical crashes, the client handle stays in place but its gRPC
        channel is gone, and every later call fails with "Stream removed".
        Probing with a trivial script tells a live session from a dead handle,
        and dropping the dead one lets the caller launch a fresh Mechanical.
        """
        if self._mechanical is None:
            return False
        try:
            self._mechanical.run_python_script("1")
        except Exception:
            self._mechanical = None
            return False
        return True

    @property
    def _an(self) -> str:
        """Remote expression for the analysis system currently being driven."""
        return "Model.Analyses[" + str(self.analysis_index) + "]"

    def connect_or_launch(self, timeout_s: int = 600) -> Dict[str, Any]:
        """
        Attach to a Mechanical already listening on ``port``; launch one if not.

        Launching with ``visible=True`` opens the Mechanical window so the user
        can watch every subsequent step in the GUI.
        """
        from ansys.mechanical.core import Mechanical, launch_mechanical

        t0 = time.time()
        mode = "attached"
        try:
            self._mechanical = Mechanical(
                port=self.port, transport_mode="insecure", cleanup_on_exit=False
            )
            self._mechanical.run_python_script("1")
        except Exception:
            self._mechanical = None

        if self._mechanical is None:
            mode = "launched"
            self._mechanical = launch_mechanical(
                batch=not self.visible,
                port=self.port,
                transport_mode="insecure",
                cleanup_on_exit=False,
                start_timeout=timeout_s,
            )

        self.project_directory = str(self._mechanical.project_directory)
        info = {
            "mode": mode,
            "port": self.port,
            "gui_visible": self.visible,
            "project_directory": self.project_directory,
            "elapsed_s": round(time.time() - t0, 1),
        }
        self._log("connect", info)
        return info

    def close(self, exit_app: bool = False):
        if self._mechanical is None:
            return
        try:
            if exit_app:
                self._mechanical.exit(force=True)
        finally:
            self._mechanical = None

    # ------------------------------------------------------------------
    # Low-level remote execution
    # ------------------------------------------------------------------

    def _run(self, tag: str, script: str) -> str:
        if self._mechanical is None:
            raise LiveMechanicalError(
                "Canlı Mechanical oturumu yok. Önce 'connect_or_launch' çağrılmalıdır."
            )
        t0 = time.time()
        try:
            # A mesh or a solve can hold the call open for minutes.  With
            # logging on, the server keeps sending progress messages down the
            # same stream, which is what stops an otherwise idle connection
            # from being reset mid-solve ("Stream removed ... 10054").
            result = self._mechanical.run_python_script(
                script, enable_logging=True, log_level="ERROR", progress_interval=2000
            )
        except Exception as exc:  # gRPC surfaces solver/API errors here
            detail = str(exc).split("details =")[-1].strip().strip('"')
            if _is_channel_loss(exc):
                # The remote end is gone, not merely unhappy with the script.
                # Releasing the handle lets the next call start a fresh
                # Mechanical instead of talking to a closed socket.
                self._mechanical = None
                self._log(tag, {"ok": False, "error": detail, "channel_lost": True},
                          time.time() - t0)
                raise LiveMechanicalError(
                    "[%s] Mechanical bağlantısı koptu: %s. Oturum bırakıldı; "
                    "işlemi tekrar çalıştırın, yeni bir Mechanical başlatılacaktır."
                    % (tag, detail)
                ) from exc
            self._log(tag, {"ok": False, "error": detail}, time.time() - t0)
            raise LiveMechanicalError(f"[{tag}] Mechanical hatası: {detail}") from exc
        self._log(tag, {"ok": True, "result": result}, time.time() - t0)
        return result

    def _log(self, tag: str, payload: Dict[str, Any], elapsed: float = 0.0):
        self.step_log.append(
            {
                "step": tag,
                "elapsed_s": round(elapsed, 2),
                "timestamp": time.strftime("%H:%M:%S"),
                **payload,
            }
        )

    # ------------------------------------------------------------------
    # Project / units
    # ------------------------------------------------------------------

    #: Mesh properties this adapter writes. A design parameter on any of them
    #: makes the property read-only to scripting, which surfaces much later as
    #: "This property is parameterized and is read-only" from the mesher.
    _PARAMETERISED_MESH_PROPERTIES = ("ElementSize", "ElementOrder", "GrowthRate")

    def reset_project(self) -> str:
        """Discard the current model so a new analysis starts from a clean tree.

        A Mechanical attached to a Workbench project can carry design
        parameters that survive the new project and lock a mesh control
        ("This property is parameterized and is read-only").  ``RemoveParameter``
        does not always clear it, and probing the control on an empty model can
        take Mechanical down, so the lock is handled where it shows up: the
        adapter restarts Mechanical and replays the job when ``generate_mesh``
        reports it (see ``MechanicalAdapter.generate_mesh``).
        """
        try:
            return self._new_project()
        except LiveMechanicalError:
            if self._mechanical is not None:
                raise
            # The previous job's Mechanical died; nothing of this job is in it yet.
            self._log("reconnect_after_channel_loss", {})
            self.connect_or_launch()
            return self._new_project()

    def restart(self) -> None:
        """Kill this Mechanical process and start a fresh one."""
        self._log("restart_mechanical", {})
        self.close(exit_app=True)
        # Attaching before the old process has released the port would reconnect
        # to the very Mechanical that is being replaced.
        deadline = time.time() + 90
        while time.time() < deadline:
            with socket.socket() as probe:
                probe.settimeout(1.0)
                if probe.connect_ex(("127.0.0.1", self.port)) != 0:
                    break
            time.sleep(1.0)
        else:
            raise LiveMechanicalError(
                "Eski Mechanical süreci 90 sn içinde %d portunu bırakmadı; yeniden başlatılamadı." % self.port
            )
        self.connect_or_launch()

    def _new_project(self) -> str:
        self.analysis_index = 0
        # The transcript belongs to one job; carrying earlier jobs' steps over
        # would put steps in this job's record that it never ran.
        self.step_log.clear()
        names = ", ".join(
            "'" + name + "'" for name in self._PARAMETERISED_MESH_PROPERTIES
        )
        return self._run(
            "reset_project",
            "import json" + chr(10)
            + "ExtAPI.DataModel.Project.New()" + chr(10)
            + "for _n in (" + names + "):" + chr(10)
            + "    try:" + chr(10)
            + "        Model.Mesh.RemoveParameter(_n)" + chr(10)
            + "    except Exception:" + chr(10)
            + "        pass" + chr(10)
            + "json.dumps({'reset': True})",
        )

    def set_unit_system_mks(self) -> str:
        return self._run(
            "set_units",
            "ExtAPI.Application.ActiveUnitSystem = MechanicalUnitSystem.StandardMKS"
            + chr(10)
            + "'StandardMKS'",
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def import_geometry(self, cad_file_path: str) -> Dict[str, Any]:
        """Import a real CAD file (STEP / IGES / Parasolid / SAT) into the tree."""
        if not os.path.exists(cad_file_path):
            raise LiveMechanicalError(f"CAD dosyası bulunamadı: {cad_file_path}")

        ext = os.path.splitext(cad_file_path)[1].lower()
        if ext in (".stl", ".obj", ".3mf", ".ply"):
            raise LiveMechanicalError(
                f"'{ext}' bir tesselasyon (yüzey ağı) formatıdır ve katı FEA geometrisi "
                "olarak içe aktarılamaz. STEP, IGES, Parasolid (.x_t) veya SAT kullanın."
            )

        if not cad_file_path.isascii():
            cad_file_path = stage_cad_ascii(cad_file_path)
        safe_path = cad_file_path.replace("\\", "/")

        script = (
            'gi = Model.GeometryImportGroup.AddGeometryImport()' + chr(10)
            + 'fmt = Ansys.Mechanical.DataModel.Enums.GeometryImportPreference.Format.Automatic'
            + chr(10)
            + 'pref = Ansys.ACT.Mechanical.Utilities.GeometryImportPreferences()' + chr(10)
            + 'pref.ProcessNamedSelections = True' + chr(10)
            + 'gi.Import("' + safe_path + '", fmt, pref)' + chr(10)
            + 'str(gi.ObjectState)'
        )
        state = self._run("import_geometry", script)
        if "Solved" not in state and "FullyDefined" not in state:
            raise LiveMechanicalError(f"Geometri içe aktarılamadı, durum: {state}")

        summary = self._geometry_summary()
        self._geodata_scale_to_mm = self._detect_geodata_scale(summary["volume_m3"])
        summary["geodata_unit_scale_to_mm"] = self._geodata_scale_to_mm
        summary["file"] = os.path.basename(cad_file_path)
        return summary

    def _geometry_summary(self) -> Dict[str, Any]:
        raw = self._run(
            "geometry_summary",
            'import json' + chr(10) + _ASCII_PRELUDE
            + 'cat = Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory' + chr(10)
            + 'bodies = Model.Geometry.GetChildren(cat.Body, True)' + chr(10)
            + 'vol = 0.0' + chr(10)
            + 'names = []' + chr(10)
            + 'for b in bodies:' + chr(10)
            + '    vol = vol + b.Volume.Value' + chr(10)
            + '    names.append(_a(b.Name))' + chr(10)
            + 'json.dumps({"body_count": bodies.Count, "volume_m3": vol, "bodies": names})',
        )
        return json.loads(raw)

    def _detect_geodata_scale(self, body_volume_m3: float) -> float:
        """
        Measure whether GeoData reports millimetres or metres.

        The GeoData bounding box is compared against the body volume already
        known in m^3, so no unit assumption is baked in.
        """
        raw = self._run(
            "geodata_scale_probe",
            'import json' + chr(10)
            + 'geo = DataModel.GeoData' + chr(10)
            + 'xs = []' + chr(10)
            + 'ys = []' + chr(10)
            + 'zs = []' + chr(10)
            + 'for a in geo.Assemblies:' + chr(10)
            + '    for p in a.Parts:' + chr(10)
            + '        for b in p.Bodies:' + chr(10)
            + '            for f in b.Faces:' + chr(10)
            + '                c = f.Centroid' + chr(10)
            + '                xs.append(c[0])' + chr(10)
            + '                ys.append(c[1])' + chr(10)
            + '                zs.append(c[2])' + chr(10)
            + 'json.dumps({"dx": max(xs)-min(xs), "dy": max(ys)-min(ys), "dz": max(zs)-min(zs)})',
        )
        box = json.loads(raw)
        span = max(box["dx"], box["dy"], box["dz"])
        if span <= 0:
            return 1.0
        # A body of V m^3 has a characteristic length of at least V^(1/3) metres.
        char_len_m = max(body_volume_m3, 1e-12) ** (1.0 / 3.0)
        # If GeoData were metres, span/char_len would be O(1..1000).
        # If it were millimetres, that ratio is 1000x larger.
        ratio = span / char_len_m
        return 1.0 if ratio > 5.0e4 else 1000.0 if ratio < 5.0e1 else 1.0

    def list_faces(self) -> List[Dict[str, Any]]:
        """
        Return every face with its real id, area, centroid, surface type and normal.

        The normal is averaged over the face's tessellation points. For a planar
        face that average is the exact outward normal; for a cylinder the
        opposing points cancel and the magnitude collapses toward zero, which is
        how ``resolve_selector`` tells curved faces apart from flat ones.
        """
        raw = self._run(
            "list_faces",
            'import json' + chr(10)
            + 'geo = DataModel.GeoData' + chr(10)
            + 'out = []' + chr(10)
            + 'for a in geo.Assemblies:' + chr(10)
            + '    for p in a.Parts:' + chr(10)
            + '        for b in p.Bodies:' + chr(10)
            + '            for f in b.Faces:' + chr(10)
            + '                c = f.Centroid' + chr(10)
            + '                nl = list(f.Normals) if f.Normals else []' + chr(10)
            + '                nx = 0.0' + chr(10)
            + '                ny = 0.0' + chr(10)
            + '                nz = 0.0' + chr(10)
            + '                cnt = len(nl) / 3' + chr(10)
            + '                for i in range(int(cnt)):' + chr(10)
            + '                    nx = nx + nl[3*i]' + chr(10)
            + '                    ny = ny + nl[3*i+1]' + chr(10)
            + '                    nz = nz + nl[3*i+2]' + chr(10)
            + '                if cnt > 0:' + chr(10)
            + '                    nx = nx / cnt' + chr(10)
            + '                    ny = ny / cnt' + chr(10)
            + '                    nz = nz / cnt' + chr(10)
            + '                out.append({"id": f.Id, "area": f.Area,'
            + ' "centroid": [c[0], c[1], c[2]], "body": b.Id,'
            + ' "mean_normal": [nx, ny, nz],'
            + ' "surface_type": str(f.SurfaceType)})' + chr(10)
            + 'json.dumps(out)',
        )
        faces = json.loads(raw)
        s = self._geodata_scale_to_mm
        for f in faces:
            f["area_mm2"] = f.pop("area") * s * s
            f["centroid_mm"] = [v * s for v in f.pop("centroid")]
            n = f["mean_normal"]
            mag = math.sqrt(sum(c * c for c in n))
            f["normal_magnitude"] = round(mag, 4)
            f["is_planar"] = "Plane" in f.get("surface_type", "") or mag > 0.95
            f["mean_normal"] = [c / mag for c in n] if mag > 1e-9 else [0.0, 0.0, 0.0]
        return faces

    def list_edges(self) -> List[Dict[str, Any]]:
        """Every edge with its real id, length and centroid, shaped like a face entry.

        Surface (shell) bodies have no end faces: supports and loads on them go on edges.
        ``area_mm2`` holds the edge length so an area-range selector still works.
        """
        raw = self._run(
            "list_edges",
            'import json' + chr(10)
            + 'out = []' + chr(10)
            + 'for a in DataModel.GeoData.Assemblies:' + chr(10)
            + '    for p in a.Parts:' + chr(10)
            + '        for b in p.Bodies:' + chr(10)
            + '            for e in b.Edges:' + chr(10)
            + '                c = e.Centroid' + chr(10)
            + '                out.append({"id": e.Id, "length": e.Length, "centroid": [c[0], c[1], c[2]], "body": b.Id})' + chr(10)
            + 'json.dumps(out)',
        )
        s = self._geodata_scale_to_mm
        return [
            {
                "id": e["id"], "area_mm2": e["length"] * s, "centroid_mm": [v * s for v in e["centroid"]],
                "body": e["body"], "mean_normal": [0.0, 0.0, 0.0], "normal_magnitude": 0.0,
                "is_planar": False, "surface_type": "edge", "entity": "edge",
            }
            for e in json.loads(raw)
        ]

    def add_layered_section(self, face_ids: List[int], material_name: str, plies: List[Dict[str, float]]) -> Dict[str, Any]:
        """Layered section on surface bodies: one layer per ply, bottom to top, angle from the global X axis."""
        rows = [(float(p["thickness_mm"]) * 1.0e-3, float(p["angle_deg"])) for p in plies]
        raw = self._run(
            "add_layered_section",
            'import json' + chr(10)
            + 'ls = Model.Geometry.AddLayeredSection()' + chr(10)
            + 'sel = ExtAPI.SelectionManager.CreateSelectionInfo(Ansys.ACT.Interfaces.Common.SelectionTypeEnum.GeometryEntities)' + chr(10)
            + 'sel.Ids = ' + json.dumps([int(i) for i in face_ids]) + chr(10)
            + 'ls.Location = sel' + chr(10)
            # The section needs a coordinate system of its own: the global one is rejected as invalid.
            + 'lcs = Model.CoordinateSystems.AddCoordinateSystem()' + chr(10)
            + 'lcs.Name = "Katman_Ekseni"' + chr(10)
            + 'ls.CoordinateSystem = lcs' + chr(10)
            # The body itself must carry a thickness too; it is set to the stack's total so the two agree.
            + 'total_m = ' + repr(sum(r[0] for r in rows)) + chr(10)
            + 'for b in Model.Geometry.GetChildren(Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory.Body, True):' + chr(10)
            + '    b.Thickness = Quantity(total_m, "m")' + chr(10)
            + 'lw = ls.Layers' + chr(10)
            + 'for t, a in ' + json.dumps(rows) + ':' + chr(10)
            + '    lw.AddThicknessLayer("' + material_name + '", t, a)' + chr(10)
            + 'json.dumps({"rows": lw.RowCount, "scope": [int(i) for i in ls.Location.Ids],'
            + ' "total_thickness": str(ls.TotalThickness), "state": str(ls.ObjectState),'
            + ' "materials": [str(lw.GetMaterial(i)) for i in range(lw.RowCount)],'
            + ' "angles": [float(lw.GetAngle(i)) for i in range(lw.RowCount)],'
            + ' "thicknesses": [float(lw.GetThickness(i)) for i in range(lw.RowCount)]})',
        )
        info = json.loads(raw)
        if info["rows"] != len(rows) or sorted(info["scope"]) != sorted(int(i) for i in face_ids):
            raise LiveMechanicalError("Katmanlı kesit beklendiği gibi kurulamadı: %s" % info)
        for (t, a), got_t, got_a in zip(rows, info["thicknesses"], info["angles"]):
            if abs(got_t - t) > 1e-12 or abs(got_a - a) > 1e-9:
                raise LiveMechanicalError("Katman değerleri geri okunamadı: %s" % info)
        return info

    # ------------------------------------------------------------------
    # Semantic face resolution
    # ------------------------------------------------------------------

    def resolve_selector(
        self, selector: SemanticSelector, faces: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Resolve a :class:`SemanticSelector` against the real imported topology.

        Scoring is geometric only — no topological index is ever hard-coded —
        so the same selector survives a CAD revision that renumbers faces.
        """
        faces = faces if faces is not None else self.list_faces()
        if not faces:
            raise LiveMechanicalError("Geometride yüz bulunamadı.")

        scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
        for face in faces:
            score = 0.0
            weight = 0.0
            why: List[str] = []

            if selector.normal_vector:
                weight += 1.0
                target = selector.normal_vector
                tmag = math.sqrt(sum(c * c for c in target)) or 1.0
                unit_target = [c / tmag for c in target]
                dot = sum(a * b for a, b in zip(face["mean_normal"], unit_target))
                if selector.either_direction:
                    dot = abs(dot)
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))
                # A normal selector is a physical claim about which way a face
                # points, so grading is deliberately steep: full credit inside
                # the stated tolerance, partial credit only just outside it, and
                # nothing beyond that. A linear falloff over 180 degrees would
                # score a perpendicular face 0.5, which is not a match at all.
                tol = selector.normal_tolerance_deg
                if angle_deg <= tol:
                    s = 1.0
                elif angle_deg <= 2.0 * tol:
                    s = 0.5
                else:
                    s = 0.0
                # A curved face has no single normal, so it cannot satisfy one.
                if not face.get("is_planar", False):
                    s = 0.0
                    why.append("yuzey egrisel (%s), normal eslesmesi gecersiz"
                               % face.get("surface_type", "?"))
                else:
                    why.append("normal aci farki=%.1f deg (skor %.2f)" % (angle_deg, s))
                score += s

            if selector.centroid_hint:
                weight += 1.0
                target = selector.centroid_hint
                dist = math.dist(face["centroid_mm"][: len(target)], target)
                span = max(1.0, self._model_span_mm(faces))
                s = max(0.0, 1.0 - dist / span)
                score += s
                why.append("centroid uzaklık=%.2f mm (skor %.2f)" % (dist, s))

            if selector.area_range_mm2:
                weight += 1.0
                lo, hi = selector.area_range_mm2
                inside = lo <= face["area_mm2"] <= hi
                score += 1.0 if inside else 0.0
                why.append("alan=%.2f mm2 aralikta=%s" % (face["area_mm2"], inside))

            if selector.position_filter:
                weight += 1.0
                pf = selector.position_filter
                axis = {"x": 0, "y": 1, "z": 2}[str(pf.get("axis", "x")).lower()]
                coord = face["centroid_mm"][axis]
                extreme = str(pf.get("extreme", "")).lower()
                if extreme in ("min", "max"):
                    vals = [f["centroid_mm"][axis] for f in faces]
                    target = min(vals) if extreme == "min" else max(vals)
                    s = 1.0 if abs(coord - target) < 1e-6 else 0.0
                    score += s
                    why.append("%s-%s ekstremum eslesme=%s" % (pf.get("axis"), extreme, bool(s)))
                else:
                    lo = float(pf.get("min", -1e30))
                    hi = float(pf.get("max", 1e30))
                    s = 1.0 if lo <= coord <= hi else 0.0
                    score += s
                    why.append("%s=%.2f mm araliginda=%s" % (pf.get("axis"), coord, bool(s)))

            confidence = (score / weight) if weight else 0.0
            scored.append((confidence, face, why))

        scored.sort(key=lambda t: t[0], reverse=True)

        # A surface body's end is several edges (arcs); every edge that ties for the best
        # score is one part of the same end, so the whole tie set is selected.
        if (faces and all(f.get("entity") == "edge" for f in faces) and not selector.multi_face
                and scored[0][0] >= selector.confidence_threshold):
            selector = selector.model_copy(update={"multi_face": True,
                                                   "confidence_threshold": scored[0][0] - 1e-9})

        if selector.multi_face:
            matches = [(c, f, w) for c, f, w in scored if c >= selector.confidence_threshold]
            if matches:
                return {
                    "selector": selector.name,
                    "face_id": matches[0][1]["id"],
                    "face_ids": [f["id"] for _, f, _ in matches],
                    "confidence": round(min(c for c, _, _ in matches), 3),
                    "margin_over_runner_up": None,
                    "area_mm2": round(sum(f["area_mm2"] for _, f, _ in matches), 3),
                    "centroid_mm": [round(v, 3) for v in matches[0][1]["centroid_mm"]],
                    "reasoning": matches[0][2],
                }
            # No face met the threshold: fall through to the usual error report.

        best_conf, best_face, best_why = scored[0]

        if best_conf < selector.confidence_threshold:
            raise LiveMechanicalError(
                "Semantik seçici '%s' güven eşiğini geçemedi (%.2f < %.2f). "
                "Gerekçe: %s. Adaylar: %s"
                % (
                    selector.name,
                    best_conf,
                    selector.confidence_threshold,
                    "; ".join(best_why),
                    [(f["id"], round(f["area_mm2"], 1)) for _, f, _ in scored[:5]],
                )
            )

        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        return {
            "selector": selector.name,
            "face_id": best_face["id"],
            "confidence": round(best_conf, 3),
            "margin_over_runner_up": round(best_conf - runner_up, 3),
            "area_mm2": round(best_face["area_mm2"], 3),
            "centroid_mm": [round(v, 3) for v in best_face["centroid_mm"]],
            "reasoning": best_why,
        }

    @staticmethod
    def _model_span_mm(faces: List[Dict[str, Any]]) -> float:
        xs = [f["centroid_mm"][0] for f in faces]
        ys = [f["centroid_mm"][1] for f in faces]
        zs = [f["centroid_mm"][2] for f in faces]
        return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def apply_material(self, material: Material, work_dir: str) -> Dict[str, Any]:
        """Write a real MatML file, import it into Engineering Data, assign it."""
        xml_path = self.write_matml(material, work_dir)
        safe = xml_path.replace("\\", "/")
        names = self._run(
            "import_material",
            _ASCII_PRELUDE
            + 'Model.Materials.Import("' + safe + '")' + chr(10)
            + 'names = []' + chr(10)
            + 'for c in Model.Materials.Children:' + chr(10)
            + '    names.append(_a(c.Name))' + chr(10)
            + '"; ".join(names)',
        )
        if material.name not in names:
            raise LiveMechanicalError(
                f"Malzeme '{material.name}' Engineering Data'ya girmedi. Mevcut: {names}"
            )

        assigned = self._run(
            "assign_material",
            'cat = Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory' + chr(10)
            + 'bodies = Model.Geometry.GetChildren(cat.Body, True)' + chr(10)
            + 'out = []' + chr(10)
            + 'for b in bodies:' + chr(10)
            + '    b.Activate()' + chr(10)
            + '    b.Material = "' + material.name + '"' + chr(10)
            + '    out.append(b)' + chr(10)
            + 'str(len(out)) + " body -> " + "' + material.name + '"',
        )
        return {
            "matml_path": xml_path,
            "engineering_data_materials": names,
            "assignment": assigned,
        }

    @staticmethod
    def write_matml(material: Material, work_dir: str) -> str:
        """Generate an Engineering Data MatML 3.1 file for a real material."""
        os.makedirs(work_dir, exist_ok=True)
        e_pa = UnitEngine.to_si(material.youngs_modulus).value
        nu = material.poissons_ratio
        yield_pa = UnitEngine.to_si(material.yield_strength).value
        ult_pa = (
            UnitEngine.to_si(material.ultimate_strength).value
            if material.ultimate_strength
            else yield_pa
        )
        if not material.density:
            raise LiveMechanicalError(
                "Malzeme '%s' için yoğunluk tanımlı değil; yoğunluk uydurulmaz." % material.name
            )
        rho = UnitEngine.to_si(material.density).value
        bulk = e_pa / (3.0 * (1.0 - 2.0 * nu))
        shear = e_pa / (2.0 * (1.0 + nu))
        t0 = "7.88860905221012e-31"

        def prop(pid: str, pairs: List[Tuple[str, float]], quals: str = "") -> str:
            body = "".join(
                '<ParameterValue parameter="%s" format="float">'
                '<Data>%s</Data>'
                '<Qualifier name="Variable Type">Dependent</Qualifier>'
                "</ParameterValue>" % (pa, repr(val))
                for pa, val in pairs
            )
            return (
                '<PropertyData property="%s"><Data format="string">-</Data>%s%s'
                '<ParameterValue parameter="pa1" format="float"><Data>%s</Data>'
                '<Qualifier name="Variable Type">Independent</Qualifier>'
                "</ParameterValue></PropertyData>" % (pid, quals, body, t0)
            )

        params = [
            ("pa0", "Density"),
            ("pa1", "Temperature"),
            ("pa2", "Young's Modulus"),
            ("pa3", "Poisson's Ratio"),
            ("pa4", "Bulk Modulus"),
            ("pa5", "Shear Modulus"),
            ("pa6", "Tensile Yield Strength"),
            ("pa7", "Compressive Yield Strength"),
            ("pa8", "Tensile Ultimate Strength"),
        ]
        props = [
            ("pr0", "Density"),
            ("pr1", "Elasticity"),
            ("pr2", "Tensile Yield Strength"),
            ("pr3", "Compressive Yield Strength"),
            ("pr4", "Tensile Ultimate Strength"),
        ]

        # Thermal properties are optional: a structural-only run must not carry
        # a conductivity nobody supplied, and a thermal run cannot solve without
        # one, so each is emitted only when the caller actually provided it.
        thermal_blocks = ""
        if material.thermal_conductivity is not None:
            params.append(("pa9", "Thermal Conductivity"))
            props.append(("pr5", "Thermal Conductivity"))
            thermal_blocks += prop(
                "pr5",
                [("pa9", UnitEngine.to_si(material.thermal_conductivity).value)],
                '<Qualifier name="Behavior">Isotropic</Qualifier>',
            )
        if material.specific_heat is not None:
            params.append(("pa10", "Specific Heat"))
            props.append(("pr6", "Specific Heat"))
            thermal_blocks += prop(
                "pr6", [("pa10", UnitEngine.to_si(material.specific_heat).value)]
            )
        if material.thermal_expansion is not None:
            params.append(("pa11", "Coefficient of Thermal Expansion"))
            props.append(("pr7", "Coefficient of Thermal Expansion"))
            thermal_blocks += prop(
                "pr7",
                [("pa11", UnitEngine.to_si(material.thermal_expansion).value)],
                '<Qualifier name="Definition">Secant</Qualifier>'
                '<Qualifier name="Behavior">Isotropic</Qualifier>',
            )

        if material.orthotropic is not None:
            o = material.orthotropic
            ortho_names = [
                ("pa12", "Young's Modulus X direction", UnitEngine.to_si(o.ex).value),
                ("pa13", "Young's Modulus Y direction", UnitEngine.to_si(o.ey).value),
                ("pa14", "Young's Modulus Z direction", UnitEngine.to_si(o.ez).value),
                ("pa15", "Poisson's Ratio XY", o.nu_xy),
                ("pa16", "Poisson's Ratio YZ", o.nu_yz),
                ("pa17", "Poisson's Ratio XZ", o.nu_xz),
                ("pa18", "Shear Modulus XY", UnitEngine.to_si(o.gxy).value),
                ("pa19", "Shear Modulus YZ", UnitEngine.to_si(o.gyz).value),
                ("pa20", "Shear Modulus XZ", UnitEngine.to_si(o.gxz).value),
            ]
            params.extend((pid, label) for pid, label, _ in ortho_names)
            elasticity_block = prop(
                "pr1",
                [(pid, value) for pid, _, value in ortho_names],
                '<Qualifier name="Definition">Secant</Qualifier>'
                '<Qualifier name="Behavior">Orthotropic</Qualifier>',
            )
        else:
            elasticity_block = prop(
                "pr1",
                [("pa2", e_pa), ("pa3", nu), ("pa4", bulk), ("pa5", shear)],
                '<Qualifier name="Definition">Secant</Qualifier>'
                '<Qualifier name="Behavior">Isotropic</Qualifier>',
            )

        limits_block = ""
        if material.ply_strength is not None:
            s = material.ply_strength
            labels = [
                ("Tensile X direction", s.xt, 1.0), ("Tensile Y direction", s.yt, 1.0), ("Tensile Z direction", s.zt, 1.0),
                ("Compressive X direction", s.xc, -1.0), ("Compressive Y direction", s.yc, -1.0),
                ("Compressive Z direction", s.zc, -1.0),
                ("Shear XY", s.s12, 1.0), ("Shear YZ", s.s23, 1.0), ("Shear XZ", s.s13, 1.0),
            ]
            given = [(label, UnitEngine.to_si(q).value * sign) for label, q, sign in labels if q is not None]
            pairs = []
            for k, (label, value) in enumerate(given):
                pid = "pa%d" % (30 + k)
                params.append((pid, label))
                pairs.append((pid, value))
            props.append(("pr8", "Orthotropic Stress Limits"))
            limits_block = prop("pr8", pairs, '<Qualifier name="Definition">Secant</Qualifier>')

        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<EngineeringData version="18.0.0.60" versiondate="29.08.2013 15:02:00">'
            "<Notes />"
            "<Materials><MatML_Doc><Material><BulkDetails>"
            "<Name>" + xml_escape(material.name) + "</Name>"
            + prop("pr0", [("pa0", rho)])
            + elasticity_block
            + prop("pr2", [("pa6", yield_pa)])
            + prop("pr3", [("pa7", yield_pa)])
            + prop("pr4", [("pa8", ult_pa)])
            + limits_block
            + thermal_blocks
            + "</BulkDetails></Material><Metadata>"
            + "".join(
                '<ParameterDetails id="%s"><Unitless /><Name>%s</Name></ParameterDetails>'
                % (pid, name.replace("'", "&apos;"))
                for pid, name in params
            )
            + "".join(
                '<PropertyDetails id="%s"><Unitless /><Name>%s</Name></PropertyDetails>'
                % (pid, name)
                for pid, name in props
            )
            + "</Metadata></MatML_Doc></Materials></EngineeringData>"
        )

        path = os.path.join(work_dir, "material_%s.xml" % _slug(material.name))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(xml)
        return path

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    def generate_mesh(self, mesh_config: MeshConfig) -> Dict[str, Any]:
        """Run the real ANSYS mesher and read back real element/node counts."""
        size_mm = UnitEngine.convert(mesh_config.global_size, "mm").value
        script = (
            'mesh = Model.Mesh' + chr(10)
            + 'mesh.Activate()' + chr(10)
            # A property that Workbench has exposed as a design parameter is
            # read-only to scripting, so the parameter link is dropped first.
            # Mesh controls must come from this analysis, not from a leftover
            # parameter set that survived an earlier project in the same
            # Mechanical process.
            + 'for _p in ("ElementSize", "ElementOrder", "GrowthRate"):' + chr(10)
            + '    try:' + chr(10)
            + '        mesh.RemoveParameter(_p)' + chr(10)
            + '    except Exception:' + chr(10)
            + '        pass' + chr(10)
            + 'mesh.ElementSize = Quantity(' + _fmt(size_mm) + ', "mm")' + chr(10)
            + 'mesh.ElementOrder = Ansys.Mechanical.DataModel.Enums.ElementOrder.Quadratic'
            + chr(10)
            + 'mesh.GrowthRate = ' + _fmt(mesh_config.growth_rate) + chr(10)
            + 'mesh.GenerateMesh()' + chr(10)
            + 'import json' + chr(10)
            + 'json.dumps({"nodes": mesh.Nodes, "elements": mesh.Elements,'
            + ' "element_size_mm": ' + _fmt(size_mm) + ','
            + ' "element_order": str(mesh.ElementOrder),'
            + ' "state": str(mesh.ObjectState)})'
        )
        data = json.loads(self._run("generate_mesh", script))
        if data.get("elements", 0) <= 0:
            raise LiveMechanicalError("Mesh üretilemedi: eleman sayısı sıfır.")
        data.update(self.mesh_quality())
        return data

    def mesh_quality(self) -> Dict[str, Any]:
        """Read real element-quality and skewness statistics from the mesher."""
        try:
            raw = self._run(
                "mesh_quality",
                'import json' + chr(10) + _NUM_PRELUDE
                + 'mm = Model.Mesh' + chr(10)
                + 'E = Ansys.Mechanical.DataModel.Enums.MeshMetricType' + chr(10)
                + 'mm.MeshMetric = E.Skewness' + chr(10)
                + 'sk_max = _n(mm.Maximum)' + chr(10)
                + 'sk_avg = _n(mm.Average)' + chr(10)
                + 'mm.MeshMetric = E.ElementQuality' + chr(10)
                + 'eq_min = _n(mm.Minimum)' + chr(10)
                + 'eq_avg = _n(mm.Average)' + chr(10)
                + 'mm.MeshMetric = E.None' + chr(10)
                + 'json.dumps({"max_skewness": sk_max, "avg_skewness": sk_avg,'
                + ' "min_element_quality": eq_min, "avg_element_quality": eq_avg})',
            )
            return json.loads(raw)
        except LiveMechanicalError:
            # Quality metrics are advisory; never fail a solve over them.
            return {"mesh_quality": "unavailable"}

    # ------------------------------------------------------------------
    # Analysis + boundary conditions
    # ------------------------------------------------------------------

    def add_static_structural(self) -> str:
        return self._add_analysis("add_analysis", "Model.AddStaticStructuralAnalysis()")

    def add_modal(self, modes: int = 6) -> str:
        name = self._add_analysis("add_modal", "Model.AddModalAnalysis()")
        self._run(
            "modal_settings",
            'a = ' + self._an + chr(10)
            + 'a.AnalysisSettings.MaximumModesToFind = ' + str(int(modes)) + chr(10)
            + 'str(a.AnalysisSettings.MaximumModesToFind)',
        )
        return name

    def add_random_vibration(self, modal_index: int, axis: str, frequencies_hz: List[float],
                             asd_m2_s4_hz: List[float], damping_ratio: float) -> str:
        """Add a random-vibration system fed by a solved-modal system and a base-acceleration PSD.

        The PSD acts at every fixed support of the linked modal system, along ``axis``.
        ``asd_m2_s4_hz`` is the acceleration spectral density in (m/s^2)^2/Hz.  Damping is
        a manual constant ratio of critical.
        """
        name = self._add_analysis("add_random_vibration", "Model.AddRandomVibrationAnalysis()")
        axis_name = {"x": "XAxis", "y": "YAxis", "z": "ZAxis"}[axis]
        state = self._run(
            "random_vibration_setup",
            'import json' + chr(10)
            + 'E = Ansys.Mechanical.DataModel.Enums' + chr(10)
            + 'rv = ' + self._an + chr(10)
            + 'ic = [c for c in rv.Children if c.GetType().Name == "InitialCondition"][0]' + chr(10)
            + 'ic.ModalEnvironmentPSDIC = Model.Analyses[' + str(int(modal_index)) + ']' + chr(10)
            + 'if ic.ModalEnvironmentPSDIC is None:' + chr(10)
            + '    raise Exception("Rastgele titreşim sistemi modal sisteme bağlanamadı.")' + chr(10)
            + 'a = rv.AnalysisSettings' + chr(10)
            + 'a.ConstantDamping = E.ConstantDampingType.Manual' + chr(10)
            + 'a.ConstantDampingRatio = ' + _fmt(damping_ratio) + chr(10)
            + 'if abs(float(str(a.ConstantDampingRatio).split()[0]) - ' + _fmt(damping_ratio) + ') > 1e-9:' + chr(10)
            + '    raise Exception("Sönüm oranı ayarlanamadı: " + str(a.ConstantDampingRatio))' + chr(10)
            + 'p = rv.AddPSDAcceleration()' + chr(10)
            + 'p.Name = "PSD_Taban_Ivmesi"' + chr(10)
            + 'p.BoundaryCondition = E.PSDBoundaryConditionSelectionType.AllFixedSupports' + chr(10)
            + 'p.Direction = E.NormalOrientationType.' + axis_name + chr(10)
            + 'p.LoadData.Inputs[0].DiscreteValues = [Quantity(f, "Hz") for f in ' + json.dumps([float(f) for f in frequencies_hz]) + ']' + chr(10)
            + 'p.LoadData.Output.DiscreteValues = [Quantity(v, "(m s^-2)^2 Hz^-1") for v in ' + json.dumps([float(v) for v in asd_m2_s4_hz]) + ']' + chr(10)
            + 'json.dumps({"psd": str(p.ObjectState), "damping": str(a.ConstantDampingRatio), "points": int(p.LoadData.Inputs[0].DiscreteValueCount)})',
        )
        info = json.loads(state)
        if info["psd"] != "FullyDefined":
            raise LiveMechanicalError("PSD yükü tanımlanamadı. Durum: %s" % info["psd"])
        if info["points"] != len(frequencies_hz):
            raise LiveMechanicalError("PSD tablosunda %d nokta var, %d bekleniyordu." % (info["points"], len(frequencies_hz)))
        return name

    def add_random_vibration_results(self, axis: str) -> Dict[str, Any]:
        """One-sigma directional deformation along the base axis, and one-sigma equivalent stress."""
        axis_name = {"x": "XAxis", "y": "YAxis", "z": "ZAxis"}[axis]
        raw = self._run(
            "add_random_vibration_results",
            'import json' + chr(10)
            + 'E = Ansys.Mechanical.DataModel.Enums' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'd = s.AddDirectionalDeformation()' + chr(10)
            + 'd.NormalOrientation = E.NormalOrientationType.' + axis_name + chr(10)
            + 'd.Name = "RV_Yon_Deformasyon_1sigma"' + chr(10)
            + 's.AddEquivalentStress()' + chr(10)
            + 'v = [c for c in s.Children if c.GetType().Name == "EquivalentStressPSD"][-1]' + chr(10)
            + 'v.Name = "RV_Von_Mises_1sigma"' + chr(10)
            + 'json.dumps({"deformation_scale": str(d.ScaleFactor), "stress_scale": str(v.ScaleFactor)})',
        )
        info = json.loads(raw)
        for key in ("deformation_scale", "stress_scale"):
            if info[key] != "Sigma1":
                raise LiveMechanicalError("Rastgele titreşim sonucu 1 sigma değil: %s=%s" % (key, info[key]))
        return info

    def extract_random_vibration_results(self) -> Dict[str, Any]:
        """One-sigma maxima in SI units (m, Pa) read off the solved result objects."""
        raw = self._run(
            "extract_random_vibration",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 's.EvaluateAllResults()' + chr(10)
            + 'out = {}' + chr(10)
            + 'for c in s.Children:' + chr(10)
            + '    n = c.GetType().Name' + chr(10)
            + '    if n in ("DirectionalDeformation", "EquivalentStressPSD"):' + chr(10)
            + '        q = c.Maximum' + chr(10)
            + '        out[n] = {"value": float(q.Value), "unit": str(q.Unit)}' + chr(10)
            + 'json.dumps(out)',
        )
        return json.loads(raw)

    def solve_after(self, prerequisite_index: int) -> Dict[str, Any]:
        """Solve a prerequisite system (the linked modal one), then the current system."""
        self._run(
            "solve_prerequisite",
            'm = Model.Analyses[' + str(int(prerequisite_index)) + ']' + chr(10)
            + 'm.Solution.Activate()' + chr(10)
            + 'm.Solve(True)' + chr(10)
            + 'str(m.Solution.ObjectState)',
        )
        return self.solve()

    def add_harmonic(self, f_min_hz: float, f_max_hz: float, intervals: int, method: str,
                     damping_ratio: float) -> str:
        """Add a harmonic response system.

        ``method`` is ``"mode_superposition"`` (damping as a constant ratio of
        critical) or ``"full"`` (no mode truncation; damping as the equivalent
        hysteretic coefficient, twice the ratio at resonance).
        """
        name = self._add_analysis("add_harmonic", "Model.AddHarmonicResponseAnalysis()")
        full = method == "full"
        self._run(
            "harmonic_settings",
            'a = ' + self._an + '.AnalysisSettings' + chr(10)
            + 'a.RangeMinimum = Quantity(' + _fmt(f_min_hz) + ', "Hz")' + chr(10)
            + 'a.RangeMaximum = Quantity(' + _fmt(f_max_hz) + ', "Hz")' + chr(10)
            + 'a.SolutionIntervals = ' + str(int(intervals)) + chr(10)
            + 'a.SolutionMethod = Ansys.Mechanical.DataModel.Enums.HarmonicMethod.'
            + ('Full' if full else 'ModeSuperposition') + chr(10)
            + ('a.StructuralDampingCoefficient = ' + _fmt(2.0 * damping_ratio) + chr(10)
               if full else 'a.ConstantDampingRatio = ' + _fmt(damping_ratio) + chr(10))
            + 'str(a.SolutionMethod)',
        )
        return name

    def add_transient(self, end_time_s: float, time_step_s: float, mass_coefficient: float,
                      stiffness_coefficient: float) -> str:
        """Add a transient structural system with a fixed time step and Rayleigh damping.

        ``mass_coefficient`` is alpha [1/s] and ``stiffness_coefficient`` is beta [s]
        of C = alpha M + beta K.  Results are stored at every time point.
        """
        name = self._add_analysis("add_transient", "Model.AddTransientStructuralAnalysis()")
        self._run(
            "transient_settings",
            'a = ' + self._an + '.AnalysisSettings' + chr(10)
            + 'a.StepEndTime = Quantity(' + _fmt(end_time_s) + ', "s")' + chr(10)
            + 'a.AutomaticTimeStepping = Ansys.Mechanical.DataModel.Enums.AutomaticTimeStepping.Off' + chr(10)
            + 'a.TimeStep = Quantity(' + _fmt(time_step_s) + ', "s")' + chr(10)
            + 'a.MassCoefficient = ' + _fmt(mass_coefficient) + chr(10)
            + 'a.StiffnessCoefficient = ' + _fmt(stiffness_coefficient) + chr(10)
            + 'str(a.StepEndTime) + " " + str(a.TimeStep)',
        )
        return name

    def add_transient_result(self) -> str:
        return self._run(
            "add_transient_result",
            'd = ' + self._an + '.Solution.AddTotalDeformation()' + chr(10)
            + 'd.Name = "Transient_Toplam_Deformasyon"' + chr(10)
            + 'str(d.Name)',
        )

    def extract_transient_history(self, times_s: List[float]) -> List[Dict[str, Any]]:
        """Largest total displacement over the model at each requested solved time."""
        raw = self._run(
            "transient_history",
            'import json' + chr(10)
            + 'sol = ' + self._an + '.Solution' + chr(10)
            + 'd = [c for c in sol.Children if c.GetType().Name == "TotalDeformation"][0]' + chr(10)
            + 'out = []' + chr(10)
            + 'for t in ' + json.dumps([float(x) for x in times_s]) + ':' + chr(10)
            + '    d.DisplayTime = Quantity(t, "s")' + chr(10)
            + '    sol.EvaluateAllResults()' + chr(10)
            + '    out.append({"time_s": t, "reported_time_s": float(d.ReportedTime.Value) if hasattr(d, "ReportedTime") else float(d.Time.Value),'
            + ' "max_displacement_m": float(d.Maximum.Value)})' + chr(10)
            + 'json.dumps(out)',
        )
        return json.loads(raw)

    def add_harmonic_response_result(self) -> str:
        """One Total Deformation result whose frequency is stepped through the sweep."""
        return self._run(
            "add_harmonic_result",
            'd = ' + self._an + '.Solution.AddTotalDeformation()' + chr(10)
            + 'd.Name = "Harmonik_Toplam_Deformasyon"' + chr(10)
            + 'str(d.Name)',
        )

    def extract_harmonic_sweep(self, frequencies_hz: List[float]) -> List[Dict[str, Any]]:
        """Deformation amplitude (max over the model) at each requested frequency."""
        raw = self._run(
            "harmonic_sweep",
            'import json' + chr(10)
            + 'sol = ' + self._an + '.Solution' + chr(10)
            + 'd = [c for c in sol.Children if c.GetType().Name == "TotalDeformation"][0]' + chr(10)
            + 'd.Amplitude = True' + chr(10)
            + 'out = []' + chr(10)
            + 'for f in ' + json.dumps([float(f) for f in frequencies_hz]) + ':' + chr(10)
            + '    d.Frequency = Quantity(f, "Hz")' + chr(10)
            + '    sol.EvaluateAllResults()' + chr(10)
            + '    out.append({"frequency_hz": f, "reported_frequency_hz": float(d.ReportedFrequency.Value),'
            + ' "max_amplitude_m": float(d.Maximum.Value)})' + chr(10)
            + 'json.dumps(out)',
        )
        return json.loads(raw)

    def add_buckling(self, modes: int = 3) -> str:
        """Add a linear eigenvalue buckling system pre-stressed by the current static one.

        The static system is the one the session currently points at; it carries
        the supports and the reference load. Once the buckling system exists the
        session points at it, so the mode results and the solve go to it.
        """
        static_index = self.analysis_index
        name = self._add_analysis("add_buckling", "Model.AddEigenvalueBucklingAnalysis()")
        linked = self._run(
            "link_buckling",
            'static = Model.Analyses[' + str(static_index) + ']' + chr(10)
            + 'b = ' + self._an + chr(10)
            + 'ic = None' + chr(10)
            + 'for c in b.Children:' + chr(10)
            + '    if c.GetType().Name == "InitialCondition":' + chr(10)
            + '        ic = c' + chr(10)
            + 'if ic is None:' + chr(10)
            + '    raise Exception("Burkulma sisteminde ön gerilme (InitialCondition) nesnesi yok.")' + chr(10)
            + 'ic.PreStressICEnvironment = static' + chr(10)
            + 'b.AnalysisSettings.MaximumModesToFind = ' + str(int(modes)) + chr(10)
            + 'str(ic.PreStressICEnvironment.Name)',
        )
        if not linked:
            raise LiveMechanicalError("Burkulma analizi statik sisteme bağlanamadı.")
        return name

    def add_buckling_mode_results(self, modes: int = 3) -> Dict[str, Any]:
        """One Total Deformation result per buckling mode; each carries its load multiplier."""
        raw = self._run(
            "add_buckling_modes",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'added = []' + chr(10)
            + 'for i in range(1, ' + str(int(modes)) + ' + 1):' + chr(10)
            + '    d = s.AddTotalDeformation()' + chr(10)
            + '    d.Mode = i' + chr(10)
            + '    d.Name = "Burkulma_Modu_" + str(i)' + chr(10)
            + '    added.append(d.Name)' + chr(10)
            + 'json.dumps({"mode_results": added})',
        )
        return json.loads(raw)

    def extract_buckling_factors(self) -> List[Dict[str, Any]]:
        """Read the real load multipliers off the solved buckling mode results."""
        raw = self._run(
            "buckling_factors",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'out = []' + chr(10)
            + 'for c in s.Children:' + chr(10)
            + '    if c.GetType().Name == "TotalDeformation" and c.Mode is not None:' + chr(10)
            + '        out.append({"mode": int(c.Mode),'
            + ' "load_multiplier": float(c.LoadMultiplier),'
            + ' "max_displacement_m": float(c.Maximum.Value)})' + chr(10)
            + 'out.sort(key=lambda r: r["mode"])' + chr(10)
            + 'json.dumps(out)',
        )
        modes = json.loads(raw)
        if not modes:
            raise LiveMechanicalError(
                "Burkulma yük çarpanı okunamadı. Analiz çözüldü mü ve mod sonuçları eklendi mi?"
            )
        return modes

    def add_steady_state_thermal(self) -> str:
        return self._add_analysis("add_thermal", "Model.AddSteadyStateThermalAnalysis()")

    def _add_analysis(self, tag: str, factory_expr: str) -> str:
        """
        Create an analysis system and point the session at it.

        The new system is always appended, so its index is the last one; every
        later call in this session then targets it rather than whatever system
        happened to be created first.
        """
        raw = self._run(
            tag,
            'import json' + chr(10) + _ASCII_PRELUDE
            + 'a = ' + factory_expr + chr(10)
            + 'a.Activate()' + chr(10)
            # Object identity is unreliable across the .NET bridge, so the
            # system is located by its persistent ObjectId instead.
            + 'idx = -1' + chr(10)
            + 'for i in range(Model.Analyses.Count):' + chr(10)
            + '    if Model.Analyses[i].ObjectId == a.ObjectId:' + chr(10)
            + '        idx = i' + chr(10)
            + 'json.dumps({"index": idx, "name": _a(a.Name)})',
        )
        info = json.loads(raw)
        if info["index"] < 0:
            raise LiveMechanicalError("Analiz sistemi oluşturuldu ama modelde bulunamadı.")
        self.analysis_index = int(info["index"])
        return info["name"]

    #: Boundary-condition types that need no face: they act on the whole model.
    GLOBAL_LOADS = (BoundaryConditionType.GRAVITY,)

    #: Mechanical object class names that carry a reaction force.
    SUPPORT_CLASSES = ("FixedSupport", "FrictionlessSupport", "Displacement", "CylindricalSupport",
                       "CompressionOnlySupport")

    def apply_boundary_condition(
        self, bc: BoundaryCondition, resolution: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply one real load or support to the resolved face."""
        name = _safe_name(bc.name)
        face_id = int(resolution["face_id"]) if resolution else None
        face_ids = [int(i) for i in (resolution.get("face_ids") or [face_id])] if resolution else []
        sel = ""
        if face_id is not None:
            sel = (
                'sel = ExtAPI.SelectionManager.CreateSelectionInfo('
                'Ansys.ACT.Interfaces.Common.SelectionTypeEnum.GeometryEntities)' + chr(10)
                + 'sel.Ids = [' + ", ".join(str(i) for i in face_ids) + ']' + chr(10)
            )
        components = 'Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components'

        def vector_load(factory: str, vec: List[float], unit: str, extra: str = "") -> str:
            return (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.' + factory + '()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.DefineBy = ' + components + chr(10)
                + extra
                + 'o.XComponent.Output.DiscreteValues = [Quantity(' + _fmt(vec[0]) + ', "' + unit + '")]' + chr(10)
                + 'o.YComponent.Output.DiscreteValues = [Quantity(' + _fmt(vec[1]) + ', "' + unit + '")]' + chr(10)
                + 'o.ZComponent.Output.DiscreteValues = [Quantity(' + _fmt(vec[2]) + ', "' + unit + '")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )

        def unit_direction(default: Optional[List[float]] = None) -> List[float]:
            d = bc.direction or default
            if not d:
                raise LiveMechanicalError(f"'{bc.name}' için yön tanımlı değil.")
            norm = math.sqrt(sum(c * c for c in d))
            if norm < 1.0e-9:
                raise LiveMechanicalError(f"'{bc.name}' yönü sıfır vektör.")
            return [c / norm for c in d]

        if bc.type == BoundaryConditionType.FIXED_SUPPORT:
            script = (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.AddFixedSupport()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.FRICTIONLESS_SUPPORT:
            script = (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.AddFrictionlessSupport()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.COMPRESSION_ONLY_SUPPORT:
            script = (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.AddCompressionOnlySupport()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.CYLINDRICAL_SUPPORT:
            dofs = bc.dofs or {}
            if set(dofs) != {'radial', 'axial', 'tangential'}:
                raise LiveMechanicalError(f"'{bc.name}' silindirik mesnedi için radial, axial ve tangential verilmelidir.")
            kind = 'Ansys.Mechanical.DataModel.Enums.FixedOrFree.'
            script = (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.AddCylindricalSupport()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Radial = ' + kind + dofs['radial'].capitalize() + chr(10)
                + 'o.Axial = ' + kind + dofs['axial'].capitalize() + chr(10)
                + 'o.Tangential = ' + kind + dofs['tangential'].capitalize() + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.DISPLACEMENT:
            comps = bc.components_mm
            if not comps or len(comps) != 3 or all(c is None for c in comps):
                raise LiveMechanicalError(
                    f"'{bc.name}' yer değiştirmesi için en az bir bileşen (ux, uy, uz) verilmelidir."
                )
            lines = (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.AddDisplacement()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.DefineBy = ' + components + chr(10)
            )
            # A component left out stays free; only the ones given are prescribed.
            for axis, value in zip("XYZ", comps):
                if value is not None:
                    lines += (
                        'o.' + axis + 'Component.Output.DiscreteValues = [Quantity('
                        + _fmt(value * 1.0e-3) + ', "m")]' + chr(10)
                    )
            script = (
                lines + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.FORCE:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' kuvvetinin büyüklüğü tanımsız.")
            newtons = UnitEngine.convert(bc.magnitude, "N").value
            d = unit_direction([0.0, -1.0, 0.0])
            script = vector_load("AddForce", [newtons * c for c in d], "N")
        elif bc.type == BoundaryConditionType.REMOTE_FORCE:
            if bc.magnitude is None or not bc.origin_mm or len(bc.origin_mm) != 3:
                raise LiveMechanicalError(
                    f"'{bc.name}' uzak kuvveti için büyüklük ve 'origin_mm' [x, y, z] gereklidir."
                )
            newtons = UnitEngine.convert(bc.magnitude, "N").value
            d = unit_direction()
            origin = (
                'o.XCoordinate = Quantity(' + _fmt(bc.origin_mm[0] * 1.0e-3) + ', "m")' + chr(10)
                + 'o.YCoordinate = Quantity(' + _fmt(bc.origin_mm[1] * 1.0e-3) + ', "m")' + chr(10)
                + 'o.ZCoordinate = Quantity(' + _fmt(bc.origin_mm[2] * 1.0e-3) + ', "m")' + chr(10)
            )
            script = vector_load("AddRemoteForce", [newtons * c for c in d], "N", origin)
        elif bc.type == BoundaryConditionType.MOMENT:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' momentinin büyüklüğü tanımsız.")
            n_m = UnitEngine.convert(bc.magnitude, "N*m").value
            d = unit_direction()
            script = vector_load("AddMoment", [n_m * c for c in d], "N m")
        elif bc.type == BoundaryConditionType.PRESSURE:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' basıncının büyüklüğü tanımsız.")
            pa = UnitEngine.to_si(bc.magnitude).value
            script = (
                'a = ' + self._an + chr(10) + sel
                + 'o = a.AddPressure()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Magnitude.Output.DiscreteValues = [Quantity(' + _fmt(pa) + ', "Pa")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.GRAVITY:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' yerçekimi ivmesi tanımsız.")
            g = UnitEngine.to_si(bc.magnitude).value
            d = unit_direction()
            # Mechanical's Acceleration load is the acceleration of the body; the
            # body force it produces points the opposite way. Gravity pulling
            # along ``d`` is therefore an acceleration of -g*d.
            acc = [-g * c for c in d]
            script = (
                'a = ' + self._an + chr(10)
                + 'o = a.AddAcceleration()' + chr(10)
                + 'o.DefineBy = ' + components + chr(10)
                + 'o.XComponent.Output.DiscreteValues = [Quantity(' + _fmt(acc[0]) + ', "m s^-1 s^-1")]' + chr(10)
                + 'o.YComponent.Output.DiscreteValues = [Quantity(' + _fmt(acc[1]) + ', "m s^-1 s^-1")]' + chr(10)
                + 'o.ZComponent.Output.DiscreteValues = [Quantity(' + _fmt(acc[2]) + ', "m s^-1 s^-1")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        else:
            raise LiveMechanicalError(
                f"'{bc.type.value}' canlı Mechanical sürücüsünde henüz desteklenmiyor."
            )

        state = self._run("bc_" + name, script)
        if "NoState" in state:
            # Right after the analysis system is created Mechanical can answer
            # "NoState" for an object that is in fact fully defined a moment
            # later.  Only that answer is re-read; UnderDefined is a real fault.
            for _ in range(6):
                time.sleep(0.5)
                state = self._run(
                    "bc_state_" + name,
                    'a = ' + self._an + chr(10)
                    + 'st = "NoState"' + chr(10)
                    + 'for c in a.Children:' + chr(10)
                    + '    if c.Name == "' + name + '":' + chr(10)
                    + '        st = str(c.ObjectState)' + chr(10)
                    + 'st',
                )
                if "NoState" not in state:
                    break
        if "FullyDefined" not in state:
            raise LiveMechanicalError(
                f"Sınır şartı '{bc.name}' tam tanımlı değil (durum: {state})."
            )
        return {
            "name": bc.name,
            "type": bc.type.value,
            "state": state,
            "face_id": face_id,
            "face_ids": face_ids,
            "confidence": resolution["confidence"] if resolution else None,
            "centroid_mm": resolution["centroid_mm"] if resolution else None,
        }

    def add_standard_results(self, safety_factor: bool = True) -> Dict[str, Any]:
        """Add stress, deformation, safety factor and reaction-force results.

        The stress tool and its safety factor are refused for layered sections, so a layered
        job passes ``safety_factor=False``.
        """
        raw = self._run(
            "add_results",
            'import json' + chr(10)
            + 'cat = Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'eqv = s.AddEquivalentStress()' + chr(10)
            + 'eqv.Name = "Esdeger_von_Mises_Gerilme"' + chr(10)
            + 'tot = s.AddTotalDeformation()' + chr(10)
            + 'tot.Name = "Toplam_Deplasman"' + chr(10)
            + ('st = s.AddStressTool()' + chr(10)
               + 'st.Name = "Gerilme_Araci"' + chr(10)
               + 'sf = st.AddSafetyFactor()' + chr(10)
               + 'sf.Name = "Emniyet_Katsayisi"' + chr(10) if safety_factor else '')
            + 'added = ["EquivalentStress", "TotalDeformation"' + (', "SafetyFactor"' if safety_factor else '') + ']' + chr(10)
            + 'n_react = 0' + chr(10)
            + 'for c in ' + self._an + '.Children:' + chr(10)
            + '    if c.GetType().Name in ' + repr(tuple(self.SUPPORT_CLASSES)) + ':' + chr(10)
            + '        fr = s.AddForceReaction()' + chr(10)
            + '        fr.BoundaryConditionSelection = c' + chr(10)
            + '        fr.Name = "Reaksiyon_" + c.Name' + chr(10)
            + '        n_react = n_react + 1' + chr(10)
            + 'if n_react > 0:' + chr(10)
            + '    added.append("ForceReaction x" + str(n_react))' + chr(10)
            + 'json.dumps({"results": added, "solution_state": str(s.ObjectState)})',
        )
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> Dict[str, Any]:
        """Run the real MAPDL solver and fail loudly if it does not converge."""
        t0 = time.time()
        raw_state = self._run(
            "solve",
            'import json' + chr(10)
            + 'a = ' + self._an + chr(10)
            + 'a.Solution.Activate()' + chr(10)
            + 'a.Solve(True)' + chr(10)
            + 'kids = [str(c.ObjectState) for c in a.Solution.Children if c.GetType().Name != "SolutionInformation"]' + chr(10)
            + 'json.dumps({"solution": str(a.Solution.ObjectState), "analysis": str(a.ObjectState), "results": kids})',
        )
        elapsed = round(time.time() - t0, 1)
        messages = self.get_messages()
        states = json.loads(raw_state)
        state = states["solution"]
        # A layered section leaves the Solution node UnderDefined although the analysis and every
        # result are Solved.  That is accepted only when the solver output itself holds no error.
        if (state != "Solved" and states["analysis"] == "Solved" and states["results"]
                and all(s == "Solved" for s in states["results"]) and not self._solver_errors()):
            state = "Solved"

        if state != "Solved":
            raise LiveMechanicalError(
                "ANSYS çözümü başarısız. Durum: %s. Solver mesajları: %s%s"
                % (state, messages or "yok", self._undefined_results() + self._solver_output_tail())
            )
        return {
            "status": "SOLVED",
            "solver": "ANSYS 2025 R1 Mechanical APDL",
            "solution_state": state,
            "solve_time_s": elapsed,
            "messages": messages,
            "working_dir": self.working_dir(),
        }

    def _solver_errors(self) -> List[str]:
        """Error lines of the solver output file (empty when there are none or the file is missing)."""
        try:
            path = os.path.join(self.working_dir(), "solve.out")
            if not os.path.exists(path):
                return ["solve.out yok"]
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return [l.strip() for l in handle if "*** ERROR ***" in l]
        except Exception:
            return ["solve.out okunamadı"]

    def _undefined_results(self) -> str:
        """Names of result objects that are not fully defined, with their invalid properties."""
        try:
            raw = self._run(
                "undefined_results",
                'import json' + chr(10)
                + 'out = []' + chr(10)
                + 'for c in ' + self._an + '.Solution.Children:' + chr(10)
                + '    if str(c.ObjectState) != "FullyDefined" and c.GetType().Name != "SolutionInformation":' + chr(10)
                + '        bad = [str(p.Name) + "=" + str(p.StringValue) for p in c.VisibleProperties if not p.IsValid]' + chr(10)
                + '        out.append(c.GetType().Name + ":" + str(c.ObjectState) + ":" + ",".join(bad))' + chr(10)
                + 'json.dumps(out)',
            )
            found = json.loads(raw)
            return (" | Tanımsız sonuçlar: %s" % "; ".join(found)) if found else ""
        except Exception:
            return ""

    def _solver_output_tail(self, lines: int = 25) -> str:
        """Last lines of the solver output file, so a failed solve explains itself."""
        try:
            wd = self.working_dir()
            for name in ("solve.out", "file.err"):
                path = os.path.join(wd, name)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="replace") as handle:
                        text = [l.rstrip() for l in handle.readlines()]
                    flagged = []
                    for i, l in enumerate(text):
                        if "*** ERROR ***" in l or "*** WARNING ***" in l:
                            flagged.extend(x.strip() for x in text[i:i + 5] if x.strip())
                    tail = flagged[:40] if flagged else [l for l in text[-lines:] if l.strip()]
                    if tail:
                        return " | Çözücü çıktısı (%s): %s" % (name, " / ".join(tail)[-1800:])
        except Exception:
            pass
        return ""

    def get_messages(self) -> List[str]:
        raw = self._run(
            "messages",
            'import json' + chr(10) + _ASCII_PRELUDE
            + 'out = []' + chr(10)
            + 'for m in ExtAPI.Application.Messages:' + chr(10)
            + '    out.append(_a(str(m.Severity) + ": " + m.DisplayString))' + chr(10)
            + 'json.dumps(out)',
        )
        return json.loads(raw)

    def working_dir(self) -> str:
        return self._run("working_dir", self._an + ".WorkingDir")

    def rst_path(self) -> Optional[str]:
        """Absolute path of the real .rst result file produced by the solver."""
        wd = self.working_dir()
        # A thermal solve writes .rth instead of .rst; both are real result files.
        for candidate in ("file.rst", "file0.rst", "file.rth"):
            p = os.path.join(wd, candidate)
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def extract_results(self) -> List[ResultMetric]:
        """Read the real solved field maxima straight out of the result objects."""
        raw = self._run(
            "extract_results",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'out = []' + chr(10)
            + 'rx = 0.0' + chr(10)
            + 'ry = 0.0' + chr(10)
            + 'rz = 0.0' + chr(10)
            + 'n_react = 0' + chr(10)
            + 'n_react_total = 0' + chr(10)
            + 'for c in s.Children:' + chr(10)
            + '    if c.GetType().Name == "ForceReaction":' + chr(10)
            + '        n_react_total = n_react_total + 1' + chr(10)
            + 'for c in s.Children:' + chr(10)
            + '    n = c.GetType().Name' + chr(10)
            + '    if n == "EquivalentStress":' + chr(10)
            + '        out.append(["maximum_von_mises_stress", c.Maximum.Value, "Pa"])' + chr(10)
            + '        out.append(["minimum_von_mises_stress", c.Minimum.Value, "Pa"])' + chr(10)
            + '    elif n == "TotalDeformation":' + chr(10)
            + '        out.append(["maximum_total_deformation", c.Maximum.Value, "m"])' + chr(10)
            + '    elif n == "ForceReaction":' + chr(10)
            + '        rx = rx + c.XAxis.Value' + chr(10)
            + '        ry = ry + c.YAxis.Value' + chr(10)
            + '        rz = rz + c.ZAxis.Value' + chr(10)
            + '        n_react = n_react + 1' + chr(10)
            + '        if n_react_total > 1:' + chr(10)
            + '            out.append(["reaction[" + c.Name + "]_x", c.XAxis.Value, "N"])' + chr(10)
            + '            out.append(["reaction[" + c.Name + "]_y", c.YAxis.Value, "N"])' + chr(10)
            + '            out.append(["reaction[" + c.Name + "]_z", c.ZAxis.Value, "N"])' + chr(10)
            + '    elif n == "StressTool":' + chr(10)
            + '        for k in c.Children:' + chr(10)
            + '            if k.GetType().Name == "StressSafetyFactor":' + chr(10)
            + '                out.append(["minimum_safety_factor", k.Minimum.Value, ""])' + chr(10)
            + 'if n_react > 0:' + chr(10)
            + '    out.append(["reaction_force_x", rx, "N"])' + chr(10)
            + '    out.append(["reaction_force_y", ry, "N"])' + chr(10)
            + '    out.append(["reaction_force_z", rz, "N"])' + chr(10)
            + '    out.append(["reaction_force_total", (rx*rx + ry*ry + rz*rz) ** 0.5, "N"])' + chr(10)
            + 'json.dumps(out)',
        )
        rows = json.loads(raw)

        metrics: List[ResultMetric] = []
        for name, value, unit in rows:
            if unit == "Pa":
                metrics.append(
                    ResultMetric(
                        metric_name=name,
                        value=round(value / 1.0e6, 4),
                        unit="MPa",
                        note="ANSYS 2025 R1 çözülmüş modelden okundu.",
                    )
                )
            elif unit == "m":
                metrics.append(
                    ResultMetric(
                        metric_name=name,
                        value=round(value * 1.0e3, 6),
                        unit="mm",
                        note="ANSYS 2025 R1 çözülmüş modelden okundu.",
                    )
                )
            else:
                metrics.append(
                    ResultMetric(
                        metric_name=name,
                        value=round(value, 6),
                        unit=unit,
                        note="ANSYS 2025 R1 çözülmüş modelden okundu.",
                    )
                )
        if not metrics:
            raise LiveMechanicalError("Çözülmüş modelden hiçbir sonuç okunamadı.")
        return metrics

    def export_result_images(self, out_dir: str) -> Dict[str, str]:
        """Export real contour PNGs rendered by Mechanical's own graphics engine."""
        os.makedirs(out_dir, exist_ok=True)
        safe = out_dir.replace("\\", "/")
        raw = self._run(
            "export_images",
            'import json, os' + chr(10)
            + 'enums = Ansys.Mechanical.DataModel.Enums' + chr(10)
            + 'g = ExtAPI.Graphics' + chr(10)
            + 'st = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()' + chr(10)
            + 'st.Resolution = enums.GraphicsResolutionType.EnhancedResolution' + chr(10)
            + 'st.Background = enums.GraphicsBackgroundType.White' + chr(10)
            + 'st.Width = 1600' + chr(10)
            + 'st.Height = 1000' + chr(10)
            + 'st.CurrentGraphicsDisplay = False' + chr(10)
            + 'fmt = enums.GraphicsImageExportFormat.PNG' + chr(10)
            + 'names = {"EquivalentStress": "von_mises_stress",'
            + ' "TotalDeformation": "total_deformation",'
            + ' "StressSafetyFactor": "safety_factor"}' + chr(10)
            + 'made = {}' + chr(10)
            + 'def shot(obj, key):' + chr(10)
            + '    obj.Activate()' + chr(10)
            + '    g.Camera.SetFit()' + chr(10)
            + '    p = os.path.join(r"' + safe + '", key + ".png")' + chr(10)
            + '    g.ExportImage(p, fmt, st)' + chr(10)
            + '    made[key] = p' + chr(10)
            + 'for c in ' + self._an + '.Solution.Children:' + chr(10)
            + '    n = c.GetType().Name' + chr(10)
            + '    if n in names:' + chr(10)
            + '        shot(c, names[n])' + chr(10)
            + '    for k in c.Children:' + chr(10)
            + '        kn = k.GetType().Name' + chr(10)
            + '        if kn in names:' + chr(10)
            + '            shot(k, names[kn])' + chr(10)
            + 'Model.Mesh.Activate()' + chr(10)
            + 'g.Camera.SetFit()' + chr(10)
            + 'mp = os.path.join(r"' + safe + '", "mesh.png")' + chr(10)
            + 'g.ExportImage(mp, fmt, st)' + chr(10)
            + 'made["mesh"] = mp' + chr(10)
            + 'json.dumps(made)',
        )
        images = json.loads(raw)
        missing = [k for k, p in images.items() if not os.path.exists(p)]
        if missing:
            raise LiveMechanicalError(f"Görsel dışa aktarılamadı: {missing}")
        return images

    def save_project(self, path: str) -> str:
        """Persist the real .mechdb so the user can reopen the exact model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safe = path.replace("\\", "/")
        return self._run(
            "save_project",
            'ExtAPI.DataModel.Project.SaveAs("' + safe + '", True)' + chr(10)
            + '"' + safe + '"',
        )

    def add_modal_mode_shapes(self, modes: int = 6) -> Dict[str, Any]:
        """Add one Total Deformation result per mode so each shape is evaluated."""
        raw = self._run(
            "add_mode_shapes",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'added = []' + chr(10)
            + 'for i in range(1, ' + str(int(modes)) + ' + 1):' + chr(10)
            + '    d = s.AddTotalDeformation()' + chr(10)
            + '    d.Mode = i' + chr(10)
            + '    d.Name = "Mod_" + str(i) + "_Sekli"' + chr(10)
            + '    added.append(d.Name)' + chr(10)
            + 'json.dumps({"mode_results": added})',
        )
        return json.loads(raw)

    def extract_modal_frequencies(self) -> List[Dict[str, Any]]:
        """Read real eigenfrequencies off the solved mode-shape results."""
        raw = self._run(
            "modal_frequencies",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'out = []' + chr(10)
            + 'for c in s.Children:' + chr(10)
            + '    if c.GetType().Name == "TotalDeformation" and c.Mode is not None:' + chr(10)
            + '        out.append({"mode": int(c.Mode),'
            + ' "frequency_hz": float(c.ReportedFrequency.Value),'
            + ' "max_amplitude_m": float(c.Maximum.Value)})' + chr(10)
            + 'out.sort(key=lambda r: r["mode"])' + chr(10)
            + 'json.dumps(out)',
        )
        modes = json.loads(raw)
        if not modes:
            raise LiveMechanicalError(
                "Modal frekans okunamadı. Modal analiz çözüldü mü ve mod şekilleri eklendi mi?"
            )
        return modes


    # ------------------------------------------------------------------
    # Steady-state thermal
    # ------------------------------------------------------------------

    def apply_thermal_boundary_condition(
        self, bc: BoundaryCondition, resolution: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply one real thermal load or constraint to the resolved face."""
        name = _safe_name(bc.name)
        sel = ""
        face_id = None
        if resolution is not None:
            face_id = int(resolution["face_id"])
            sel = (
                'sel = ExtAPI.SelectionManager.CreateSelectionInfo('
                'Ansys.ACT.Interfaces.Common.SelectionTypeEnum.GeometryEntities)' + chr(10)
                + 'sel.Ids = [' + str(face_id) + ']' + chr(10)
            )

        head = 'a = ' + self._an + chr(10) + sel

        if bc.type == BoundaryConditionType.TEMPERATURE:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' sıcaklığı tanımsız.")
            kelvin = UnitEngine.to_si(bc.magnitude).value
            script = (
                head
                + 'o = a.AddTemperature()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Magnitude.Output.DiscreteValues = [Quantity(' + _fmt(kelvin) + ', "K")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.CONVECTION:
            if bc.film_coefficient is None or bc.ambient_temperature is None:
                raise LiveMechanicalError(
                    f"'{bc.name}' taşınımı için film katsayısı ve ortam sıcaklığı gereklidir."
                )
            h = UnitEngine.to_si(bc.film_coefficient).value
            t_amb = UnitEngine.to_si(bc.ambient_temperature).value
            script = (
                head
                + 'o = a.AddConvection()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.FilmCoefficient.Output.DiscreteValues = '
                + '[Quantity(' + _fmt(h) + ', "W m^-1 m^-1 K^-1")]' + chr(10)
                + 'o.AmbientTemperature.Output.DiscreteValues = '
                + '[Quantity(' + _fmt(t_amb) + ', "K")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.HEAT_FLOW:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' ısı akışı tanımsız.")
            watts = UnitEngine.to_si(bc.magnitude).value
            script = (
                head
                + 'o = a.AddHeatFlow()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Magnitude.Output.DiscreteValues = [Quantity(' + _fmt(watts) + ', "W")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        elif bc.type == BoundaryConditionType.INTERNAL_HEAT_GENERATION:
            if bc.magnitude is None:
                raise LiveMechanicalError(f"'{bc.name}' iç ısı üretimi tanımsız.")
            w_per_m3 = UnitEngine.to_si(bc.magnitude).value
            script = (
                'a = ' + self._an + chr(10)
                + 'cat = Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory' + chr(10)
                + 'bodies = Model.Geometry.GetChildren(cat.Body, True)' + chr(10)
                + 'sel = ExtAPI.SelectionManager.CreateSelectionInfo('
                'Ansys.ACT.Interfaces.Common.SelectionTypeEnum.GeometryEntities)' + chr(10)
                + 'sel.Ids = [b.GetGeoBody().Id for b in bodies]' + chr(10)
                + 'o = a.AddInternalHeatGeneration()' + chr(10)
                + 'o.Location = sel' + chr(10)
                + 'o.Magnitude.Output.DiscreteValues = '
                + '[Quantity(' + _fmt(w_per_m3) + ', "W m^-3")]' + chr(10)
                + 'o.Name = "' + name + '"' + chr(10)
                + 'o.Activate()' + chr(10)
                + 'str(o.ObjectState)'
            )
        else:
            raise LiveMechanicalError(
                f"'{bc.type.value}' termal analizde desteklenmiyor."
            )

        state = self._run("thermal_bc_" + name, script)
        if "FullyDefined" not in state:
            raise LiveMechanicalError(
                f"Termal sınır şartı '{bc.name}' tam tanımlı değil (durum: {state})."
            )
        return {
            "name": bc.name,
            "type": bc.type.value,
            "state": state,
            "face_id": face_id,
            "confidence": resolution["confidence"] if resolution else None,
            "centroid_mm": resolution["centroid_mm"] if resolution else None,
        }

    def add_thermal_results(self) -> Dict[str, Any]:
        """Add temperature and total heat flux results to the thermal solution."""
        raw = self._run(
            "add_thermal_results",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 't = s.AddTemperature()' + chr(10)
            + 't.Name = "Sicaklik_Dagilimi"' + chr(10)
            + 'q = s.AddTotalHeatFlux()' + chr(10)
            + 'q.Name = "Toplam_Isi_Akisi"' + chr(10)
            + 'json.dumps({"results": ["Temperature", "TotalHeatFlux"],'
            + ' "solution_state": str(s.ObjectState)})',
        )
        return json.loads(raw)

    def extract_thermal_results(self) -> List[ResultMetric]:
        """
        Read solved temperature and heat-flux extremes off the result objects.

        Mechanical reports every result in the project's active unit system, so
        the unit string is read back with the value instead of being assumed.
        Assuming kelvin when the project is in celsius silently shifts every
        temperature by 273.15 K and the numbers still look plausible.
        """
        raw = self._run(
            "extract_thermal",
            'import json' + chr(10)
            + 's = ' + self._an + '.Solution' + chr(10)
            + 'out = []' + chr(10)
            + 'for c in s.Children:' + chr(10)
            + '    n = c.GetType().Name' + chr(10)
            + '    if n == "TemperatureResult":' + chr(10)
            + '        out.append(["maximum_temperature", c.Maximum.Value, str(c.Maximum.Unit)])' + chr(10)
            + '        out.append(["minimum_temperature", c.Minimum.Value, str(c.Minimum.Unit)])' + chr(10)
            + '        out.append(["average_temperature", c.Average.Value, str(c.Average.Unit)])' + chr(10)
            + '    elif n == "TotalHeatFlux":' + chr(10)
            + '        out.append(["maximum_heat_flux", c.Maximum.Value, str(c.Maximum.Unit)])' + chr(10)
            + '        out.append(["average_heat_flux", c.Average.Value, str(c.Average.Unit)])' + chr(10)
            + 'json.dumps(out)',
        )
        rows = json.loads(raw)
        if not rows:
            raise LiveMechanicalError("Çözülmüş termal modelden hiçbir sonuç okunamadı.")

        metrics: List[ResultMetric] = []
        for name, value, unit in rows:
            token = _normalize_mechanical_unit(unit)
            if UnitEngine.is_temperature(token):
                celsius = UnitEngine.convert(Quantity(value=value, unit=token), "C").value
                metrics.append(
                    ResultMetric(
                        metric_name=name,
                        value=round(celsius, 4),
                        unit="C",
                        note="ANSYS 2025 R1 çözülmüş termal modelden okundu (%s)." % unit,
                    )
                )
            else:
                si = UnitEngine.to_si(Quantity(value=value, unit=token))
                metrics.append(
                    ResultMetric(
                        metric_name=name,
                        value=round(si.value, 4),
                        unit=si.unit,
                        note="ANSYS 2025 R1 çözülmüş termal modelden okundu (%s)." % unit,
                    )
                )
        return metrics

    def export_named_result_images(self, out_dir: str, type_names: Dict[str, str]) -> Dict[str, str]:
        """
        Export one PNG per result object whose type appears in ``type_names``.

        ``type_names`` maps a Mechanical result class name to the file stem, so
        callers decide which fields to capture for their analysis type.
        """
        os.makedirs(out_dir, exist_ok=True)
        safe = out_dir.replace(chr(92), "/")
        mapping = json.dumps(type_names)
        raw = self._run(
            "export_images_named",
            'import json, os' + chr(10)
            + 'enums = Ansys.Mechanical.DataModel.Enums' + chr(10)
            + 'g = ExtAPI.Graphics' + chr(10)
            + 'st = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()' + chr(10)
            + 'st.Resolution = enums.GraphicsResolutionType.EnhancedResolution' + chr(10)
            + 'st.Background = enums.GraphicsBackgroundType.White' + chr(10)
            + 'st.Width = 1600' + chr(10)
            + 'st.Height = 1000' + chr(10)
            + 'st.CurrentGraphicsDisplay = False' + chr(10)
            + 'fmt = enums.GraphicsImageExportFormat.PNG' + chr(10)
            + 'names = json.loads(' + repr(mapping) + ')' + chr(10)
            + 'made = {}' + chr(10)
            + 'def shot(obj, key):' + chr(10)
            + '    obj.Activate()' + chr(10)
            + '    g.Camera.SetFit()' + chr(10)
            + '    p = os.path.join(r"' + safe + '", key + ".png")' + chr(10)
            + '    g.ExportImage(p, fmt, st)' + chr(10)
            + '    made[key] = p' + chr(10)
            + 'for c in ' + self._an + '.Solution.Children:' + chr(10)
            + '    n = c.GetType().Name' + chr(10)
            + '    if n in names:' + chr(10)
            + '        key = names[n]' + chr(10)
            + '        if hasattr(c, "Mode") and c.Mode is not None:' + chr(10)
            + '            key = key + "_" + str(int(c.Mode))' + chr(10)
            + '        shot(c, key)' + chr(10)
            + '    for k in c.Children:' + chr(10)
            + '        kn = k.GetType().Name' + chr(10)
            + '        if kn in names:' + chr(10)
            + '            shot(k, names[kn])' + chr(10)
            + 'json.dumps(made)',
        )
        images = json.loads(raw)
        missing = [k for k, p in images.items() if not os.path.exists(p)]
        if missing:
            raise LiveMechanicalError(f"Görsel dışa aktarılamadı: {missing}")
        return images

    # ------------------------------------------------------------------
    # Modal participation factors
    # ------------------------------------------------------------------

    def extract_participation_factors(self) -> Dict[str, Any]:
        """
        Read the effective mass table MAPDL itself printed for the modal solve.

        Mechanical does not expose participation factors through the scripting
        object model, but the solver writes the full table into ``solve.out``,
        so the real numbers are parsed from the solver's own output rather than
        assumed from mode ordering.
        """
        out_file = os.path.join(self.working_dir(), "solve.out")
        if not os.path.exists(out_file):
            raise LiveMechanicalError(
                "solve.out bulunamadı; katılım faktörleri okunamaz: %s" % out_file
            )
        return parse_participation_factors(out_file)


def stage_cad_ascii(cad_file_path: str, stage_dir: Optional[str] = None) -> str:
    """
    Copy a CAD file to an ASCII path.

    The ANSYS geometry reader raises a bare ``COM E_FAIL`` on paths containing
    non-ASCII characters, which is routine for Turkish project folders, so the
    file is staged rather than read in place.
    """
    import shutil
    import tempfile

    target_dir = stage_dir or os.path.join(tempfile.gettempdir(), "ampc_cad_stage")
    os.makedirs(target_dir, exist_ok=True)
    ascii_name = "".join(
        c if (c.isalnum() or c in "-_.") else "_" for c in os.path.basename(cad_file_path)
    )
    staged = os.path.join(target_dir, ascii_name)
    if os.path.abspath(staged) != os.path.abspath(cad_file_path):
        shutil.copy2(cad_file_path, staged)
    return staged


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text]
    return "".join(keep)[:60]


#: Substrings gRPC uses when the connection itself is gone, as opposed to the
#: remote script raising. Reusing the handle after one of these always fails.
_CHANNEL_LOSS_MARKERS = (
    "stream removed",
    "unavailable",
    "socket closed",
    "connection reset",
    "transport is closing",
    "failed to connect",
)


def _is_channel_loss(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CHANNEL_LOSS_MARKERS)


def _safe_name(text: str) -> str:
    """Mechanical object names must avoid quotes that would break the remote script."""
    return text.replace('"', "").replace("\\", "").replace(chr(10), " ")[:60]


# ----------------------------------------------------------------------
# Module-level singleton so the MCP server keeps one live GUI across calls
# ----------------------------------------------------------------------

_SESSION: Optional[LiveMechanicalSession] = None


def get_live_session(visible: bool = True, port: int = DEFAULT_GRPC_PORT) -> LiveMechanicalSession:
    """Return the process-wide live Mechanical session, creating it if needed."""
    global _SESSION
    if _SESSION is None or not _SESSION.is_live:
        _SESSION = LiveMechanicalSession(port=port, visible=visible)
        _SESSION.connect_or_launch()
    return _SESSION


def reset_live_session():
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
    _SESSION = None


#: Directions MAPDL prints a participation factor table for.
_PF_DIRECTIONS = ("X", "Y", "Z", "ROTX", "ROTY", "ROTZ")

#: Header of one participation-factor table, e.g.
#: ``***** PARTICIPATION FACTOR CALCULATION *****  Y  DIRECTION``.
_PF_HEADER = re.compile(
    r"PARTICIPATION FACTOR CALCULATION\s*\*+\s*(ROTX|ROTY|ROTZ|X|Y|Z)\s+DIRECTION"
)


#: Mechanical spells units in MAPDL style ("W m^-2", "C"). These are the
#: spellings the unit engine is given instead.
_MECHANICAL_UNIT_ALIASES = {
    "C": "C",
    "K": "K",
    "F": "F",
    "W m^-2": "W/m2",
    # Mechanical writes an area denominator as two length terms.
    "W m^-1 m^-1": "W/m2",
    "W m^-1 m^-1 K^-1": "W/m2K",
    "W m^-1 m^-1 C^-1": "W/m2K",
    "W m^-3": "W/m3",
    "W/m^2": "W/m2",
    "W m^-1 C^-1": "W/mK",
    "W m^-1 K^-1": "W/mK",
    "W": "W",
    "m": "m",
    "mm": "mm",
    "Hz": "Hz",
}


def _normalize_mechanical_unit(unit: str) -> str:
    """Translate a unit string Mechanical reported into a unit-engine token."""
    token = (unit or "").strip()
    return _MECHANICAL_UNIT_ALIASES.get(token, token)


def parse_participation_factors(solve_out_path: str) -> Dict[str, Any]:
    """
    Parse the ``PARTICIPATION FACTOR CALCULATION`` tables out of ``solve.out``.

    MAPDL prints one table per direction. Each data row is::

        MODE  FREQUENCY  PERIOD  PARTIC.FACTOR  RATIO  EFFECTIVE MASS  CUMULATIVE  RATIO EFF.MASS

    Only rows whose first field is an integer mode number are taken, so header
    and separator lines are ignored without relying on fixed line offsets.
    """
    with open(solve_out_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split(chr(10))

    per_direction: Dict[str, Dict[int, Dict[str, float]]] = {}
    current: Optional[str] = None

    for line in lines:
        upper = line.upper()
        if "PARTICIPATION FACTOR CALCULATION" in upper:
            # The direction label sits between the asterisks and the word
            # DIRECTION. It must be matched as a whole token: "X DIRECTION" is
            # a substring of "ROTX DIRECTION", so a plain containment test
            # silently files the rotational table under the translational axis.
            match = _PF_HEADER.search(upper)
            current = match.group(1) if match else None
            if current:
                per_direction.setdefault(current, {})
            continue

        if current is None:
            continue

        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            mode = int(fields[0])
            frequency = float(fields[1])
            factor = float(fields[3])
            effective_mass = float(fields[5])
            cumulative_fraction = float(fields[6])
            ratio_to_total = float(fields[7]) if len(fields) > 7 else 0.0
        except (ValueError, IndexError):
            continue

        per_direction[current][mode] = {
            "frequency_hz": frequency,
            "participation_factor": factor,
            "effective_mass": effective_mass,
            "cumulative_mass_fraction": cumulative_fraction,
            "effective_mass_ratio": ratio_to_total,
        }

    if not per_direction:
        raise LiveMechanicalError(
            "solve.out içinde katılım faktörü tablosu bulunamadı: %s" % solve_out_path
        )

    modes = sorted({m for rows in per_direction.values() for m in rows})
    per_mode: List[Dict[str, Any]] = []
    for mode in modes:
        entry: Dict[str, Any] = {"mode": mode}
        for direction in _PF_DIRECTIONS:
            row = per_direction.get(direction, {}).get(mode)
            if row is None:
                continue
            key = direction.lower()
            entry["frequency_hz"] = row["frequency_hz"]
            entry["effective_mass_" + key] = row["effective_mass"]
            entry["effective_mass_ratio_" + key] = row["effective_mass_ratio"]
            entry["cumulative_fraction_" + key] = row["cumulative_mass_fraction"]
        per_mode.append(entry)

    cumulative = {}
    for direction, rows in per_direction.items():
        if rows:
            last_mode = max(rows)
            cumulative[direction.lower()] = rows[last_mode]["cumulative_mass_fraction"]

    return {
        "source": solve_out_path,
        "directions": sorted(per_direction),
        "modes": per_mode,
        "cumulative_mass_fraction": cumulative,
    }
