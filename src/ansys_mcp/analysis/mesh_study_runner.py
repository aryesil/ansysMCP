"""
Mesh Convergence Study Runner
-----------------------------
Runs a three-level mesh refinement study (coarse -> medium -> fine) as three
real ANSYS solves and compiles the ASME V&V 20 / Roache GCI report plus the
singularity diagnosis from what the solver actually returned.

Every grid level here is a separate solve of the same model at a different
element size. Nothing is extrapolated in place of a run: a GCI computed from
invented grid values reports mesh independence that was never demonstrated,
which is the single most misleading number a FEA report can carry.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.mechanical import MechanicalAdapter
from ..domain.models import BoundaryCondition, JobState, Material, MeshConfig, Quantity
from ..orchestrator.storage import JobStorage
from .gci_engine import GciEngine, GridLevelResult
from .singularity_detector import SingularityDetector


class MeshStudyError(RuntimeError):
    """Raised when a grid level could not be solved."""


class MeshStudyRunner:
    """Orchestrates three real refinement solves and evaluates convergence."""

    #: Metric the GCI is computed on, and the companion metric the singularity
    #: detector needs. Stress diverges at a singularity while displacement
    #: converges, which is exactly how the detector tells them apart.
    STRESS_METRIC = "maximum_von_mises_stress"
    DISPLACEMENT_METRIC = "maximum_total_deformation"

    @classmethod
    def run_mesh_study(
        cls,
        storage: JobStorage,
        job_id: str,
        refinement_ratio: float = 1.5,
        target_gci_pct: float = 3.0,
        show_gui: bool = False,
    ) -> Dict[str, Any]:
        job = storage.get_job(job_id)
        if not job or not job["config_json"]:
            return {"status": "error", "message": "Analiz bulunamadı veya konfigüre edilmedi."}

        if refinement_ratio <= 1.0:
            return {
                "status": "error",
                "message": "Sıklaştırma oranı 1.0'dan büyük olmalıdır; aksi halde üç ağ birbirinden ayrışmaz.",
            }

        cfg = json.loads(job["config_json"])
        if "material" not in cfg or "boundary_conditions" not in cfg:
            return {
                "status": "error",
                "message": "Mesh yakınsama çalışması statik yapısal konfigürasyon gerektirir.",
            }

        material = Material(**cfg["material"])
        bcs = [BoundaryCondition(**b) for b in cfg["boundary_conditions"]]
        base_size = float(cfg["mesh_config"]["global_size"]["value"])

        levels = [
            ("Coarse (h3)", round(base_size * refinement_ratio, 3)),
            ("Medium (h2)", round(base_size, 3)),
            ("Fine (h1)", round(base_size / refinement_ratio, 3)),
        ]

        job_dir = storage.get_job_dir(job_id)
        storage.update_job_state(job_id, JobState.SOLVING)

        solved: List[Tuple[str, float, Dict[str, Any]]] = []
        try:
            for level_name, size_mm in levels:
                solved.append((level_name, size_mm, cls._solve_level(
                    job=job,
                    job_dir=job_dir,
                    level_name=level_name,
                    size_mm=size_mm,
                    material=material,
                    boundary_conditions=bcs,
                    show_gui=show_gui,
                )))
        except Exception as exc:
            storage.update_job_state(job_id, JobState.FAILED, error_message=str(exc))
            return {
                "status": "error",
                "job_id": job_id,
                "message": f"Mesh yakınsama çalışması başarısız: {exc}",
            }

        storage.update_job_state(job_id, JobState.POST_PROCESSING)

        def grid(level_index: int, metric: str) -> GridLevelResult:
            level_name, size_mm, data = solved[level_index]
            if metric not in data["metrics"]:
                raise MeshStudyError(
                    f"'{level_name}' seviyesinde '{metric}' sonucu okunamadı."
                )
            return GridLevelResult(
                level_name=level_name,
                element_size_mm=size_mm,
                element_count=int(data["elements"]),
                result_value=float(data["metrics"][metric]),
            )

        try:
            stress_levels = [grid(i, cls.STRESS_METRIC) for i in (2, 1, 0)]
            disp_levels = [grid(i, cls.DISPLACEMENT_METRIC) for i in (2, 1, 0)]
        except MeshStudyError as exc:
            return {"status": "error", "job_id": job_id, "message": str(exc)}

        gci_report = GciEngine.calculate_gci(
            metric_name=cls.STRESS_METRIC,
            h1_fine=stress_levels[0],
            h2_medium=stress_levels[1],
            h3_coarse=stress_levels[2],
            target_gci_pct=target_gci_pct,
        )

        singularity_report = SingularityDetector.evaluate_singularity(
            stress_levels=stress_levels,
            displacement_levels=disp_levels,
        )

        # Mark the study done only if the job already carries a result;
        # otherwise a job that never had a full solve would claim COMPLETED.
        if job.get("manifest_json"):
            storage.update_job_state(job_id, JobState.COMPLETED)

        report = {
            "status": "success",
            "job_id": job_id,
            "grid_levels": [
                {
                    "level_name": name,
                    "element_size_mm": size_mm,
                    "nodes": data["nodes"],
                    "elements": data["elements"],
                    "solve_time_s": data["solve_time_s"],
                    "rst_file": data["rst_file"],
                    "metrics": data["metrics"],
                }
                for name, size_mm, data in solved
            ],
            "gci_report": gci_report.model_dump(),
            "singularity_report": singularity_report.model_dump(),
            "verdict": (
                f"Mesh Bağımsızlığı Doğrulandı (GCI: %{gci_report.gci_fine_pct:.2f} <= %{target_gci_pct}). "
                f"Sonuçlar sonlu eleman boyutu değişiminden bağımsızdır."
                if gci_report.is_converged else
                f"Mesh Bağımsızlığı Henüz Sağlanamadı (GCI: %{gci_report.gci_fine_pct:.2f} > %{target_gci_pct}). "
                f"Daha ince mesh gereklidir."
            ),
        }
        # Keep what the three real solves produced, so a later read of the
        # report does not launch them again.
        with open(cls.stored_path(storage, job_id), "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report

    @staticmethod
    def stored_path(storage: Any, job_id: str) -> str:
        return os.path.join(storage.get_job_dir(job_id), "mesh_study.json")

    @classmethod
    def load_stored(cls, storage: Any, job_id: str) -> Optional[Dict[str, Any]]:
        """The report of an earlier study of this job, or ``None`` if none was run."""
        path = cls.stored_path(storage, job_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------

    @classmethod
    def _solve_level(
        cls,
        job: Any,
        job_dir: str,
        level_name: str,
        size_mm: float,
        material: Material,
        boundary_conditions: List[BoundaryCondition],
        show_gui: bool,
    ) -> Dict[str, Any]:
        """Solve one grid level for real and return its mesh size and results."""
        slug = level_name.split(" ")[0].lower()
        level_dir = os.path.join(job_dir, "mesh_study", slug)

        adapter = MechanicalAdapter(visible=show_gui)
        adapter.initialize_session(f"{job['job_id']}-{slug}", level_dir)
        adapter.load_geometry(job["cad_file_path"])
        adapter.apply_material(material)
        mesh_report = adapter.generate_mesh(
            MeshConfig(global_size=Quantity(value=size_mm, unit="mm"))
        )
        adapter.apply_boundary_conditions(boundary_conditions)
        solve_res = adapter.solve()
        results = adapter.extract_results()

        return {
            "nodes": mesh_report.get("nodes"),
            "elements": mesh_report.get("elements"),
            "solve_time_s": solve_res.get("solve_time_s"),
            "rst_file": solve_res.get("rst_file"),
            "metrics": {r.metric_name: r.value for r in results},
        }
