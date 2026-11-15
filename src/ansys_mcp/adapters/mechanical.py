"""
PyMechanical Adapter (ANSYS 2025 R1 / v251)
--------------------------------------------
Drives a real ANSYS Mechanical session through :mod:`mechanical_live`.

Every method here mutates a live Mechanical application: the geometry lands in
the real tree, Engineering Data receives a real material, the real mesher and
the real MAPDL solver run, and results are read back off the solved model.
Nothing in this adapter fabricates a number — if ANSYS cannot deliver a value,
the call raises instead of substituting a placeholder.
"""

import math
import os
import shutil
from typing import Any, Dict, List, Optional

from .base import BaseSimulationAdapter
from .mechanical_live import (
    LiveMechanicalError,
    LiveMechanicalSession,
    get_live_session,
    parse_participation_factors,
    stage_cad_ascii,
)
from ..domain.models import (
    BoundaryCondition,
    BoundaryConditionType,
    Material,
    MeshConfig,
    ResultMetric,
)
from ..domain.units import UnitEngine

class MeshQualityError(RuntimeError):
    """The mesher produced elements the solver should not be trusted with."""


#: Formats the ANSYS geometry reader accepts as true solid B-Rep.
SOLID_CAD_FORMATS = (".step", ".stp", ".iges", ".igs", ".x_t", ".x_b", ".sat", ".sldprt", ".sldasm")


class MechanicalAdapter(BaseSimulationAdapter):
    """Adapter for ANSYS 2025 R1 Mechanical static structural and modal analysis."""

    def __init__(self, visible: bool = True, session: Optional[LiveMechanicalSession] = None):
        self.visible = visible
        self.job_id: str = ""
        self.work_dir: str = ""
        self.cad_file_path: str = ""
        self.staged_cad_path: str = ""
        self.material: Optional[Material] = None
        self.mesh_config: Optional[MeshConfig] = None
        self.boundary_conditions: List[BoundaryCondition] = []
        self.applied_load_n: float = 0.0
        self.geometry_info: Dict[str, Any] = {}
        self.layered_section_applied: bool = False
        self.mesh_info: Dict[str, Any] = {}
        self.bc_report: List[Dict[str, Any]] = []
        self._session = session

    # ------------------------------------------------------------------

    @property
    def session(self) -> LiveMechanicalSession:
        # A session cached from an earlier job may have died with its
        # Mechanical process; reusing it would fail on the next remote call.
        if self._session is not None and not self._session.is_live:
            self._session = None
        if self._session is None:
            self._session = get_live_session(visible=self.visible)
        return self._session

    @property
    def step_log(self) -> List[Dict[str, Any]]:
        return self.session.step_log

    # ------------------------------------------------------------------

    def initialize_session(self, job_id: str, work_dir: str) -> bool:
        self.job_id = job_id
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        # Touching .session already attaches to (or launches) the shared GUI.
        self.session.reset_project()
        self.session.set_unit_system_mks()
        return True

    def stage_cad(self, cad_file_path: str) -> str:
        """
        Copy the CAD next to the job with an ASCII name.

        The ANSYS geometry reader raises a bare ``COM E_FAIL`` on paths holding
        non-ASCII characters, which is common for Turkish project folders, so
        the file is staged rather than read in place.
        """
        if not os.path.exists(cad_file_path):
            raise LiveMechanicalError(f"CAD dosyası bulunamadı: {cad_file_path}")

        ext = os.path.splitext(cad_file_path)[1].lower()
        if ext not in SOLID_CAD_FORMATS:
            raise LiveMechanicalError(
                f"'{ext}' katı FEA geometrisi değil. Desteklenen: {', '.join(SOLID_CAD_FORMATS)}. "
                "STL/OBJ gibi yüzey ağları hacim mesh'lenemez."
            )

        return stage_cad_ascii(cad_file_path, os.path.join(self.work_dir, "cad"))

    def load_geometry(self, cad_file_path: str) -> Dict[str, Any]:
        self.cad_file_path = cad_file_path
        self.staged_cad_path = self.stage_cad(cad_file_path)
        self.geometry_info = self.session.import_geometry(self.staged_cad_path)
        self.geometry_info["source_file"] = cad_file_path
        self.geometry_info["staged_file"] = self.staged_cad_path
        self.geometry_info["faces"] = self.session.list_faces()
        self.geometry_info["shell_model"] = not (self.geometry_info.get("volume_m3") or 0.0) > 0.0
        if self.geometry_info["shell_model"]:
            self.geometry_info["edges"] = self.session.list_edges()
        return self.geometry_info

    def apply_material(self, material: Material) -> Dict[str, Any]:
        self.material = material
        return self.session.apply_material(material, self.work_dir)

    def apply_layered_section(self, plies: List[Dict[str, float]]) -> Dict[str, Any]:
        """Layered section on every surface body; the job material is the ply material."""
        if not self.geometry_info.get("shell_model"):
            raise LiveMechanicalError(
                "Katmanlı kesit yüzey (kabuk) gövdesi ister; bu geometri katı. "
                "Katmanlı kompozit için ince cidarın orta yüzeyini içeren bir STEP verin."
            )
        if self.material is None:
            raise LiveMechanicalError("Katmanlı kesitten önce ply malzemesi atanmalıdır.")
        ids = [f["id"] for f in self.geometry_info["faces"]]
        report = self.session.add_layered_section(ids, self.material.name, plies)
        self.layered_section_applied = True
        return report

    def generate_mesh(self, mesh_config: MeshConfig) -> Dict[str, Any]:
        self.mesh_config = mesh_config
        try:
            self.mesh_info = self.session.generate_mesh(mesh_config)
        except LiveMechanicalError as exc:
            if "parameterized and is read-only" not in str(exc):
                raise
            # A mesh control is locked by a design parameter that only a new
            # Mechanical process clears.  Nothing but the geometry and material
            # exists in the model at this point, so they are replayed.
            self.session.restart()
            self.session.reset_project()
            self.session.set_unit_system_mks()
            self.geometry_info = self.session.import_geometry(self.staged_cad_path)
            self.geometry_info["source_file"] = self.cad_file_path
            self.geometry_info["staged_file"] = self.staged_cad_path
            self.geometry_info["faces"] = self.session.list_faces()
            self.session.apply_material(self.material, self.work_dir)
            self.mesh_info = self.session.generate_mesh(mesh_config)
            self.mesh_info["restarted_mechanical"] = True
        self.mesh_info["quality_gate"] = self.check_mesh_quality(self.mesh_info, mesh_config)
        return self.mesh_info

    @staticmethod
    def check_mesh_quality(mesh_info: Dict[str, Any], mesh_config: MeshConfig) -> Dict[str, Any]:
        """Stop on a mesh the solver should not be trusted with; warn on a marginal one.

        Skewness above ``max_skewness`` or element quality below
        ``min_element_quality`` raise :class:`MeshQualityError`.  When the mesher
        reported no quality statistics the gate cannot judge, and says so rather
        than passing the mesh.
        """
        skew = mesh_info.get("max_skewness")
        quality = mesh_info.get("min_element_quality")
        if skew is None or quality is None:
            return {"passed": None, "warnings": ["Mesh kalite istatistikleri okunamadı; mesh kalitesi doğrulanamadı."]}
        problems = []
        if skew > mesh_config.max_skewness:
            problems.append("en büyük çarpıklık %.3f > izin verilen %.3f" % (skew, mesh_config.max_skewness))
        if quality < mesh_config.min_element_quality:
            problems.append("en düşük eleman kalitesi %.3f < izin verilen %.3f" % (quality, mesh_config.min_element_quality))
        if problems:
            raise MeshQualityError(
                "Mesh kalitesi yetersiz (%s). Eleman boyutunu küçültün ya da geometriyi sadeleştirin."
                % "; ".join(problems)
            )
        warnings = []
        if skew > 0.85:
            warnings.append("Çarpıklık yüksek (%.3f > 0.85); gerilme sonuçları mesh'e duyarlı olabilir." % skew)
        if quality < mesh_config.target_quality:
            warnings.append("En düşük eleman kalitesi hedefin altında (%.3f < %.2f)." % (quality, mesh_config.target_quality))
        return {"passed": True, "max_skewness": skew, "min_element_quality": quality, "warnings": warnings}

    # Equilibrium tolerance when a pressure is among the loads (see verify_equilibrium).
    PRESSURE_TOLERANCE_PCT = 1.5

    def _selection_entities(self) -> List[Dict[str, Any]]:
        """Faces of a solid model; edges of a surface (shell) model, where supports and loads sit."""
        if self.geometry_info.get("shell_model"):
            return self.geometry_info.get("edges") or self.session.list_edges()
        return self.geometry_info.get("faces") or self.session.list_faces()

    def apply_boundary_conditions(self, boundary_conditions: List[BoundaryCondition],
                                  analysis_factory=None, add_results: bool = True) -> Dict[str, Any]:
        """Resolve each semantic selector against the real topology, then apply it.

        ``analysis_factory`` creates the analysis system the conditions go into;
        a static structural system by default.
        """
        self.boundary_conditions = boundary_conditions
        (analysis_factory or self.session.add_static_structural)()

        # Surface bodies carry supports and loads on their edges, solids on their faces.
        faces = self._selection_entities()
        self.bc_report = []
        self.applied_load_n = 0.0
        # Net applied load as a vector, so equilibrium can be checked as a
        # vector sum. ``None`` once a load whose resultant cannot be stated
        # exactly (a pressure on a curved face) has been applied.
        self.applied_vector_n: Optional[List[float]] = [0.0, 0.0, 0.0]
        self.pressure_applied = False
        faces_by_id = {f["id"]: f for f in faces}

        for bc in boundary_conditions:
            # Gravity acts on the whole model and has no face to resolve.
            if bc.type in self.session.GLOBAL_LOADS:
                resolution = None
            else:
                resolution = self.session.resolve_selector(bc.target_selector, faces)
            applied = self.session.apply_boundary_condition(bc, resolution)
            self.bc_report.append(applied)
            if bc.type in (BoundaryConditionType.FORCE, BoundaryConditionType.REMOTE_FORCE) and bc.magnitude:
                newtons = UnitEngine.convert(bc.magnitude, "N").value
                self.applied_load_n += newtons
                d = bc.direction or [0.0, -1.0, 0.0]
                norm = math.sqrt(sum(c * c for c in d)) or 1.0
                if self.applied_vector_n is not None:
                    for i in range(3):
                        self.applied_vector_n[i] += newtons * d[i] / norm
            elif bc.type == BoundaryConditionType.GRAVITY and bc.magnitude:
                # Weight = mass * g along the gravity direction. Mass comes from
                # the imported volume and the assigned density.
                volume = self.geometry_info.get("volume_m3")
                if not volume or self.material is None or not self.material.density:
                    self.applied_vector_n = None
                elif self.applied_vector_n is not None:
                    self.pressure_applied = True   # the weight is estimated from tessellated geometry
                    rho = UnitEngine.to_si(self.material.density).value
                    g = UnitEngine.to_si(bc.magnitude).value
                    d = bc.direction or [0.0, 0.0, -1.0]
                    norm = math.sqrt(sum(c * c for c in d)) or 1.0
                    weight = volume * rho * g
                    self.applied_load_n += weight
                    for i in range(3):
                        self.applied_vector_n[i] += weight * d[i] / norm
            elif bc.type == BoundaryConditionType.PRESSURE and bc.magnitude:
                face_list = [faces_by_id.get(i) for i in (resolution.get("face_ids") or [resolution["face_id"]])]
                if any(f is None or not f.get("is_planar") for f in face_list):
                    # The resultant of a pressure over a curved face is the
                    # integral of p*n over the surface, not p*A*(mean normal).
                    self.applied_vector_n = None
                elif self.applied_vector_n is not None:
                    self.pressure_applied = True
                    pressure_pa = UnitEngine.to_si(bc.magnitude).value
                    for face in face_list:
                        area_m2 = face["area_mm2"] * 1.0e-6
                        # Mechanical's positive pressure presses into the surface,
                        # against the outward normal.
                        for i in range(3):
                            self.applied_vector_n[i] -= pressure_pa * area_m2 * face["mean_normal"][i]

        # Static stress, safety factor and reaction results do not exist for every
        # analysis type (a damped harmonic system refuses force reactions).
        results_added = (
            self.session.add_standard_results(safety_factor=not self.layered_section_applied)
            if add_results else {"results": []}
        )
        return {
            "boundary_conditions": self.bc_report,
            "results_requested": results_added,
            "total_applied_force_n": self.applied_load_n,
        }

    def solve(self) -> Dict[str, Any]:
        """Run the real MAPDL solve and verify Newton's first law against it."""
        solve_res = self.session.solve()
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["equilibrium"] = self.verify_equilibrium()
        solve_res["mesh"] = self.mesh_info
        return solve_res

    def archive_solver_files(self) -> Optional[str]:
        """
        Copy the solver's own output files into the job directory.

        Mechanical solves inside a temporary project folder that is relocated by
        a later ``SaveAs`` and eventually cleaned up, so the ``.rst`` is archived
        with the job while it is still guaranteed to be on disk.
        """

        source_rst = self.session.rst_path()
        if not source_rst:
            return None

        solver_dir = os.path.join(self.work_dir, "solver")
        os.makedirs(solver_dir, exist_ok=True)
        source_dir = os.path.dirname(source_rst)

        archived_rst = None
        for name in ("file.rst", "file0.rst", "file.rth", "solve.out", "file.err", "file.mntr"):
            src = os.path.join(source_dir, name)
            if os.path.exists(src):
                dst = os.path.join(solver_dir, name)
                shutil.copy2(src, dst)
                if name.endswith((".rst", ".rth")) and archived_rst is None:
                    archived_rst = dst
        return archived_rst or source_rst

    # ------------------------------------------------------------------
    # Modal
    # ------------------------------------------------------------------

    def setup_modal(self, boundary_conditions: List[BoundaryCondition], modes: int = 6) -> Dict[str, Any]:
        """
        Build a real modal system: supports only, then one result per mode.

        Loads are meaningless in a free-vibration eigenvalue problem, so any
        force or pressure in the list is rejected rather than silently ignored.
        """
        self.boundary_conditions = boundary_conditions
        self.session.add_modal(modes=modes)

        faces = self._selection_entities()
        self.bc_report = []
        for bc in boundary_conditions:
            if bc.type not in (
                BoundaryConditionType.FIXED_SUPPORT,
                BoundaryConditionType.DISPLACEMENT,
            ):
                raise LiveMechanicalError(
                    f"Modal analizde '{bc.type.value}' yükü kullanılamaz; "
                    "yalnızca mesnet tanımlanabilir."
                )
            resolution = self.session.resolve_selector(bc.target_selector, faces)
            self.bc_report.append(self.session.apply_boundary_condition(bc, resolution))

        shapes = self.session.add_modal_mode_shapes(modes=modes)
        return {
            "modes_requested": modes,
            "boundary_conditions": self.bc_report,
            "mode_results": shapes["mode_results"],
        }

    def setup_harmonic(self, boundary_conditions: List[BoundaryCondition], f_min_hz: float,
                       f_max_hz: float, intervals: int, damping_ratio: float,
                       method: str = "mode_superposition") -> Dict[str, Any]:
        """Supports and a harmonic force amplitude, then a mode-superposition frequency sweep."""
        report = self.apply_boundary_conditions(
            boundary_conditions,
            analysis_factory=lambda: self.session.add_harmonic(f_min_hz, f_max_hz, intervals, method, damping_ratio),
            add_results=False,
        )
        self.session.add_harmonic_response_result()
        report["frequency_range_hz"] = [f_min_hz, f_max_hz]
        report["solution_intervals"] = intervals
        report["damping_ratio"] = damping_ratio
        report["method"] = method
        return report

    def solve_harmonic(self) -> Dict[str, Any]:
        solve_res = self.session.solve()
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["mesh"] = self.mesh_info
        return solve_res

    def setup_transient(self, boundary_conditions: List[BoundaryCondition], end_time_s: float,
                        time_step_s: float, mass_coefficient: float,
                        stiffness_coefficient: float) -> Dict[str, Any]:
        """Supports and loads applied to a transient system with Rayleigh damping."""
        report = self.apply_boundary_conditions(
            boundary_conditions,
            analysis_factory=lambda: self.session.add_transient(
                end_time_s, time_step_s, mass_coefficient, stiffness_coefficient),
            add_results=False,
        )
        self.session.add_transient_result()
        report.update({"end_time_s": end_time_s, "time_step_s": time_step_s,
                       "mass_coefficient": mass_coefficient, "stiffness_coefficient": stiffness_coefficient})
        return report

    def solve_transient(self) -> Dict[str, Any]:
        solve_res = self.session.solve()
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["mesh"] = self.mesh_info
        return solve_res

    def setup_random_vibration(self, boundary_conditions: List[BoundaryCondition], axis: str,
                               frequencies_hz: List[float], asd_m2_s4_hz: List[float],
                               damping_ratio: float, modes: int) -> Dict[str, Any]:
        """Supports on a modal system, then a random-vibration system driven by a base-acceleration PSD."""
        self.boundary_conditions = boundary_conditions
        self.session.add_modal(modes=modes)
        modal_index = self.session.analysis_index
        faces = self._selection_entities()
        self.bc_report = []
        for bc in boundary_conditions:
            if bc.type != BoundaryConditionType.FIXED_SUPPORT:
                raise LiveMechanicalError(
                    f"Rastgele titreşimde '{bc.type.value}' kullanılamaz; yalnızca sabit mesnet tanımlanabilir."
                )
            resolution = self.session.resolve_selector(bc.target_selector, faces)
            self.bc_report.append(self.session.apply_boundary_condition(bc, resolution))
        self.session.add_modal_mode_shapes(modes=modes)
        self._rv_modal_index = modal_index
        self.session.add_random_vibration(modal_index, axis, frequencies_hz, asd_m2_s4_hz, damping_ratio)
        self.session.add_random_vibration_results(axis)
        return {"boundary_conditions": self.bc_report, "axis": axis, "psd_points": len(frequencies_hz),
                "damping_ratio": damping_ratio, "modes": modes}

    def solve_random_vibration(self) -> Dict[str, Any]:
        solve_res = self.session.solve_after(self._rv_modal_index)
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["mesh"] = self.mesh_info
        rv_index = self.session.analysis_index
        self.session.analysis_index = self._rv_modal_index
        try:
            solve_res["modal_frequencies"] = self.session.extract_modal_frequencies()
        finally:
            self.session.analysis_index = rv_index
        solve_res["results_si"] = self.session.extract_random_vibration_results()
        return solve_res

    def setup_buckling(self, boundary_conditions: List[BoundaryCondition], modes: int = 3) -> Dict[str, Any]:
        """Static pre-stress with the reference loads, then a linked buckling system."""
        report = self.apply_boundary_conditions(boundary_conditions)
        self.session.add_buckling(modes=modes)
        shapes = self.session.add_buckling_mode_results(modes=modes)
        report["buckling_modes_requested"] = modes
        report["mode_results"] = shapes["mode_results"]
        return report

    def solve_buckling(self) -> Dict[str, Any]:
        solve_res = self.session.solve()
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["mesh"] = self.mesh_info
        return solve_res

    def solve_modal(self) -> Dict[str, Any]:
        solve_res = self.session.solve()
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["mesh"] = self.mesh_info
        return solve_res

    def extract_buckling(self) -> Dict[str, Any]:
        """Real load multipliers from the solved model, scaled by the reference load."""
        factors = self.session.extract_buckling_factors()
        vec = getattr(self, "applied_vector_n", None)
        reference_n = None
        if vec is not None:
            reference_n = math.sqrt(sum(v * v for v in vec))
        modes = []
        for row in factors:
            critical = row["load_multiplier"] * reference_n if reference_n else None
            modes.append({
                "mode": row["mode"],
                "load_multiplier": row["load_multiplier"],
                "critical_load_n": critical,
                "max_displacement_m": row["max_displacement_m"],
            })
        return {"modes": modes, "reference_load_n": reference_n}

    def extract_modal(self) -> Dict[str, Any]:
        """Read real eigenfrequencies and the solver's own participation table."""
        frequencies = self.session.extract_modal_frequencies()
        participation = self._participation_from_archive()
        return {"frequencies": frequencies, "participation": participation}

    def _participation_from_archive(self) -> Dict[str, Any]:
        """
        Parse participation factors from the archived solver output.

        ``archive_solver_files`` already copied ``solve.out`` next to the job,
        so the table survives the project relocation that follows a save.
        """
        archived = os.path.join(self.work_dir, "solver", "solve.out")
        if os.path.exists(archived):
            return parse_participation_factors(archived)
        return self.session.extract_participation_factors()

    def export_modal_images(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        target = out_dir or os.path.join(self.work_dir, "plots")
        return self.session.export_named_result_images(target, {"TotalDeformation": "mode_shape"})

    # ------------------------------------------------------------------
    # Steady-state thermal
    # ------------------------------------------------------------------

    def setup_thermal(self, boundary_conditions: List[BoundaryCondition]) -> Dict[str, Any]:
        """Build a real steady-state thermal system with real thermal loads."""
        self.boundary_conditions = boundary_conditions
        self.session.add_steady_state_thermal()

        faces = self.geometry_info.get("faces") or self.session.list_faces()
        self.bc_report = []
        for bc in boundary_conditions:
            # Internal heat generation is a body load, so it needs no face.
            resolution = None
            if bc.type != BoundaryConditionType.INTERNAL_HEAT_GENERATION:
                resolution = self.session.resolve_selector(bc.target_selector, faces)
            self.bc_report.append(
                self.session.apply_thermal_boundary_condition(bc, resolution)
            )

        added = self.session.add_thermal_results()
        return {"boundary_conditions": self.bc_report, "results_requested": added}

    def solve_thermal(self) -> Dict[str, Any]:
        solve_res = self.session.solve()
        solve_res["rst_file"] = self.archive_solver_files()
        solve_res["mesh"] = self.mesh_info
        return solve_res

    def extract_thermal_results(self) -> List[ResultMetric]:
        return self.session.extract_thermal_results()

    def export_thermal_images(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        target = out_dir or os.path.join(self.work_dir, "plots")
        return self.session.export_named_result_images(
            target,
            {"TemperatureResult": "temperature_contour", "TotalHeatFlux": "heat_flux"},
        )

    def verify_equilibrium(self, tolerance_pct: float = 0.5) -> Dict[str, Any]:
        """
        Compare the applied load with the reaction force ANSYS actually computed.

        This is a real check: the reaction comes from the solved model, not from
        echoing the applied load back.
        """
        metrics = {m.metric_name: m for m in self.extract_results()}
        reaction = metrics.get("reaction_force_total")
        if reaction is None:
            return {
                "status": "UNAVAILABLE",
                "message": "Reaksiyon kuvveti sonucu modelde yok (sabit mesnet tanımlı mı?).",
            }

        applied_vec = getattr(self, "applied_vector_n", None)
        if applied_vec is None:
            return {
                "status": "UNVERIFIED",
                "passed": False,
                "message": "Eğri bir yüzeye uygulanan basıncın bileşkesi kesin olarak bilinmiyor; "
                "denge kontrolü yapılamadı. Reaksiyon: %.6f N." % reaction.value,
            }
        applied = math.sqrt(sum(c * c for c in applied_vec))

        components = [metrics.get("reaction_force_" + axis) for axis in ("x", "y", "z")]
        if any(c is None for c in components):
            return {"status": "UNAVAILABLE", "message": "Reaksiyon bileşenleri modelde yok."}
        reaction_vec = [c.value for c in components]

        if applied <= 0.0:
            # No external force. With several supports (a prescribed
            # displacement against a fixed end, say) the reactions must still
            # cancel; the net is judged against the largest single reaction.
            per_support = {}
            for key, metric in metrics.items():
                if key.startswith("reaction[") and key[-2:] in ("_x", "_y", "_z"):
                    per_support.setdefault(key[:-2], [0.0, 0.0, 0.0])["xyz".index(key[-1])] = metric.value
            if len(per_support) < 2:
                return {
                    "status": "UNAVAILABLE",
                    "passed": False,
                    "message": "Uygulanan net kuvvet sıfır ve tek mesnet var; denge kontrolü anlamsız.",
                }
            scale = max(math.sqrt(sum(c * c for c in v)) for v in per_support.values())
            net = math.sqrt(sum(c * c for c in reaction_vec))
            if scale <= 0.0:
                return {"status": "UNAVAILABLE", "passed": False, "message": "Mesnet reaksiyonları sıfır."}
            pct = net / scale * 100.0
            passed = pct <= tolerance_pct
            return {
                "status": "BALANCED" if passed else "IMBALANCED",
                "passed": passed,
                "applied_force_n": 0.0,
                "largest_support_reaction_n": round(scale, 6),
                "net_reaction_n": round(net, 6),
                "residual_pct": round(pct, 6),
                "tolerance_pct": tolerance_pct,
                "law": "Dış kuvvet yok: mesnet reaksiyonlarının vektör toplamı sıfır olmalıdır.",
                "source": "ANSYS çözülmüş modelinden okunan reaksiyon kuvvetleri.",
            }
        # Newton I: applied + reaction = 0, summed as vectors.
        residual_vec = [a + r for a, r in zip(applied_vec, reaction_vec)]
        residual = math.sqrt(sum(c * c for c in residual_vec))
        residual_pct = residual / applied * 100.0

        # A pressure's resultant is p times the face area, and a self-weight is
        # density times the body volume. Mechanical reports both from the
        # tessellated geometry, whose round edges are inscribed polygons, so the
        # estimated load is a little low -- 0.65 % on the 16/14 mm tube -- while
        # the reaction the solver returns is exact. The tolerance is widened to
        # cover that and the report says why.
        tolerance_note = None
        if getattr(self, "pressure_applied", False) and tolerance_pct < self.PRESSURE_TOLERANCE_PCT:
            tolerance_pct = self.PRESSURE_TOLERANCE_PCT
            tolerance_note = (
                "Basınç veya yerçekimi içeren yükte uygulanan kuvvet, Mechanical'ın tessellation tabanlı "
                "yüz alanı/hacminden tahmin edilir (eğri kenarlarda hafif düşük); tolerans %.1f olarak "
                "alındı." % tolerance_pct
            )
        passed = residual_pct <= tolerance_pct

        return {
            "status": "BALANCED" if passed else "IMBALANCED",
            "passed": passed,
            "applied_force_n": round(applied, 6),
            "reaction_force_n": round(reaction.value, 6),
            "applied_vector_n": [round(c, 6) for c in applied_vec],
            "reaction_vector_n": [round(c, 6) for c in reaction_vec],
            "residual_n": round(residual, 6),
            "residual_pct": round(residual_pct, 6),
            "tolerance_pct": tolerance_pct,
            "tolerance_note": tolerance_note,
            "law": "Newton I: sum(F_applied) + sum(F_reaction) = 0 (vector sum)",
            "source": "ANSYS çözülmüş modelinden okunan reaksiyon kuvveti.",
        }

    def extract_results(self) -> List[ResultMetric]:
        return self.session.extract_results()

    def export_result_images(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        target = out_dir or os.path.join(self.work_dir, "plots")
        return self.session.export_result_images(target)

    def save_project(self) -> str:
        return self.session.save_project(
            os.path.join(self.work_dir, f"{self.job_id}.mechdb")
        )

    def export_solve_script(self) -> str:
        """
        Write the exact remote-script transcript that produced this analysis.

        Unlike a speculative template, every line here was actually executed
        against the running Mechanical session.
        """
        path = os.path.join(self.work_dir, "executed_mechanical_steps.log")
        lines = [
            "# ======================================================================",
            f"# Executed PyMechanical transcript for job: {self.job_id}",
            "# Target: ANSYS 2025 R1 (v251) Mechanical, live gRPC session",
            "# Every step below ran against the real application.",
            "# ======================================================================",
        ]
        for entry in self.session.step_log:
            lines.append(
                "[{ts}] {step:<24} {elapsed:>6.2f}s  ok={ok}  {payload}".format(
                    ts=entry.get("timestamp", ""),
                    step=entry.get("step", ""),
                    elapsed=entry.get("elapsed_s", 0.0),
                    ok=entry.get("ok", True),
                    payload=str(entry.get("result", entry.get("error", "")))[:300],
                )
            )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return path

    def close(self):
        # The GUI session is deliberately left running so the user can keep
        # inspecting the model; reset_live_session() tears it down explicitly.
        self._session = None
