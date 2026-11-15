"""
PyFluent Adapter (ANSYS 2025 R1)
--------------------------------
Runs real external-aerodynamics CFD in ANSYS Fluent through PyFluent.

The chain is:

1. Fluent's own CAD reader facets the supplied STEP/IGES/Parasolid part.  Fluent
   is used for this instead of a Python tessellator because generic tessellators
   leave needle triangles on swept and revolved faces, and those make the volume
   mesher fail with "tet initialization failed".
2. :mod:`ansys_mcp.cfd.domain_mesh` wraps that faceted surface in a named wind
   tunnel and writes one Fluent boundary mesh.
3. The Watertight Geometry workflow fills the tunnel with a poly-hexcore volume
   mesh carrying prism inflation layers on the body.
4. The same session switches to the solver, sets k-omega SST (or the requested
   model), initialises and iterates.
5. Drag/lift coefficients and forces are read back from Fluent report
   definitions; residuals come from the solver's monitor stream.

Every number this adapter returns is produced by Fluent.  There are no
fallbacks: if Fluent is not installed or a step fails, the call raises.
"""

import math
import os
import time
from typing import Any, Dict, List, Optional

from .base import BaseSimulationAdapter
from ..cfd.domain_mesh import build_domain_mesh
from ..cfd.inflation import InflationEngine, InflationLayerConfig
from ..config import SystemConfig
from ..domain.models import BoundaryCondition, BoundaryConditionType, MeshConfig, Material, ResultMetric
from ..domain.units import UnitEngine
from ..geometry.cad_parser import CadParser

#: Sea-level air used unless the caller supplies a fluid material.
DEFAULT_DENSITY_KG_M3 = 1.225
DEFAULT_VISCOSITY_PA_S = 1.7894e-5

#: Iterations between force-coefficient checks during a solve.
FORCE_CHECK_CHUNK = 25

#: Fluent turbulence model names keyed by the project's identifiers.
TURBULENCE_MODELS = {
    "k-omega-sst": ("k-omega", "sst"),
    "k-omega": ("k-omega", "standard"),
    "k-epsilon": ("k-epsilon", None),
    "spalart-allmaras": ("spalart-allmaras", None),
    "laminar": ("laminar", None),
}


class FluentAdapter(BaseSimulationAdapter):
    """Adapter that drives a real ANSYS Fluent session for external aerodynamics."""

    def __init__(self):
        self.job_id: str = ""
        self.work_dir: str = ""
        self.cad_file_path: str = ""
        self.inlet_velocity_m_s: float = 30.0
        self.angle_of_attack_deg: float = 0.0
        self.sideslip_deg: float = 0.0
        self.turbulence_model: str = "k-omega-sst"
        self.fluid_type: str = "air"
        self.density_kg_m3: float = DEFAULT_DENSITY_KG_M3
        self.viscosity_pa_s: float = DEFAULT_VISCOSITY_PA_S
        self.inflation_cfg: Optional[InflationLayerConfig] = None

        self._solver = None
        self._meshing_session = None
        self._geometry: Dict[str, Any] = {}
        self._mesh_info: Dict[str, Any] = {}
        self._solve_info: Dict[str, Any] = {}
        self._zones: Dict[str, str] = {}
        self._body_walls: List[str] = []
        self._inlets: List[str] = []
        self._outlets: List[str] = []
        self._reference_area_m2: float = 0.0
        self._reference_length_m: float = 0.0
        self._wall_boundary_conditions: List[BoundaryCondition] = []
        self._area_override_m2: Optional[float] = None
        self._length_override_m: Optional[float] = None
        self._moment_center_mm: Optional[List[float]] = None
        self._body_center_mm: List[float] = [0.0, 0.0, 0.0]
        self._reference_area_source: str = "faceted_body_bounding_box"
        self._characteristic_length_m: Optional[float] = None
        self._y_plus: Dict[str, Any] = {}
        self._cm_value: Optional[float] = None

    # -------------------------------------------------------------- accessors

    @property
    def solver_session(self):
        """The live PyFluent solver session, once the mesh has been built."""
        return self._solver

    @property
    def body_wall_zones(self) -> List[str]:
        return list(self._body_walls)

    @property
    def inlet_zones(self) -> List[str]:
        return list(self._inlets)

    @property
    def mesh_info(self) -> Dict[str, Any]:
        return dict(self._mesh_info)

    # ------------------------------------------------------------------ setup

    def initialize_session(self, job_id: str, work_dir: str) -> bool:
        info = SystemConfig.detect_ansys_installation()
        if info is None or not info.fluent_exe:
            raise RuntimeError(
                "ANSYS Fluent kurulumu bulunamadı. CFD analizi çalıştırılamaz."
            )
        self.job_id = job_id
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        return True

    def load_geometry(self, cad_file_path: str) -> Dict[str, Any]:
        """Reads the CAD file and reports its real extents."""
        if not os.path.exists(cad_file_path):
            raise FileNotFoundError("CAD dosyası bulunamadı: %s" % cad_file_path)
        self.cad_file_path = cad_file_path
        summary = CadParser.parse_cad_file(cad_file_path)
        self._geometry = {
            "status": "LOADED",
            "file": os.path.basename(cad_file_path),
            "size_kb": round(os.path.getsize(cad_file_path) / 1024, 2),
            "bounding_box_mm": summary.bounding_box,
            "surface_area_mm2": summary.total_surface_area_mm2,
            "volume_mm3": summary.total_volume_mm3,
        }
        return self._geometry

    def apply_material(self, material: Material) -> bool:
        """Takes the working fluid's density from the supplied material."""
        self.fluid_type = material.name
        if material.density is not None:
            si = UnitEngine.to_si(material.density)
            if si.value <= 0.0:
                raise ValueError("Akışkan yoğunluğu pozitif olmalı: %s" % material.density)
            self.density_kg_m3 = float(si.value)
        return True

    def configure_aerodynamics(
        self,
        inlet_velocity_m_s: float = 30.0,
        angle_of_attack_deg: float = 0.0,
        turbulence_model: str = "k-omega-sst",
        target_y_plus: float = 1.0,
        sideslip_deg: float = 0.0,
        characteristic_length_m: Optional[float] = None,
        reference_area_m2: Optional[float] = None,
        reference_length_m: Optional[float] = None,
        moment_center_mm: Optional[List[float]] = None,
        fluid_density_kg_m3: Optional[float] = None,
        dynamic_viscosity_pa_s: Optional[float] = None,
    ):
        if turbulence_model not in TURBULENCE_MODELS:
            raise ValueError(
                "Desteklenmeyen türbülans modeli: %s (geçerli: %s)"
                % (turbulence_model, ", ".join(sorted(TURBULENCE_MODELS)))
            )
        self.inlet_velocity_m_s = inlet_velocity_m_s
        self.angle_of_attack_deg = angle_of_attack_deg
        self.sideslip_deg = float(sideslip_deg)
        self.turbulence_model = turbulence_model
        if fluid_density_kg_m3 is not None:
            self.density_kg_m3 = float(fluid_density_kg_m3)
        if dynamic_viscosity_pa_s is not None:
            self.viscosity_pa_s = float(dynamic_viscosity_pa_s)
        self._area_override_m2 = reference_area_m2
        self._length_override_m = reference_length_m
        self._moment_center_mm = list(moment_center_mm) if moment_center_mm is not None else None
        self._characteristic_length_m = characteristic_length_m
        self.inflation_cfg = InflationEngine.calculate_inflation(
            target_y_plus=target_y_plus,
            velocity_m_s=inlet_velocity_m_s,
            characteristic_length_m=characteristic_length_m or self._reference_length_m or 1.0,
            turbulence_model=turbulence_model,
        )

    # ------------------------------------------------------------------ mesh

    def generate_mesh(self, mesh_config: MeshConfig) -> Dict[str, Any]:
        """Facets the CAD in Fluent, builds the tunnel, and fills it."""
        if not self.cad_file_path:
            raise RuntimeError("Önce load_geometry() ile CAD dosyası yüklenmelidir.")

        max_size_mm = float(UnitEngine.convert(mesh_config.global_size, "mm").value)
        if mesh_config.min_size is not None:
            min_size_mm = float(UnitEngine.convert(mesh_config.min_size, "mm").value)
        else:
            min_size_mm = max_size_mm / 10.0

        surface_path = os.path.join(self.work_dir, "body_surface.msh.h5")
        domain_path = os.path.join(self.work_dir, "domain.msh")

        started = time.time()
        self._facet_cad(surface_path, min_size_mm, max_size_mm, mesh_config)
        domain = build_domain_mesh(surface_path, domain_path)
        self._store_reference_dimensions(domain["body_bounding_box_mm"])
        quality = self._fill_volume_mesh(domain_path, mesh_config)

        self._mesh_info = {
            "status": "MESHED",
            "mesh_type": "Poly-Hexcore with Prism Inflation",
            "cell_count": quality["cell_count"],
            "min_orthogonal_quality": quality["min_orthogonal_quality"],
            "average_orthogonal_quality": quality["average_orthogonal_quality"],
            "max_skewness": quality["max_skewness"],
            "body_facet_count": domain["body_face_count"],
            "domain_bounds_mm": domain["domain_bounds"].model_dump(),
            "reference_area_m2": self._reference_area_m2,
            "reference_area_source": self._reference_area_source,
            "reference_length_m": self._reference_length_m,
            "first_layer_height_mm": self.inflation_cfg.first_layer_height_mm if self.inflation_cfg else None,
            "inflation_layers": self._inflation_layer_count(mesh_config),
            "meshing_time_s": round(time.time() - started, 1),
            "surface_mesh_path": surface_path,
            "domain_mesh_path": domain_path,
        }
        if quality["min_orthogonal_quality"] < mesh_config.target_quality:
            self._mesh_info["quality_warning"] = (
                "Minimum ortogonal kalite (%.3f) hedefin (%.3f) altında."
                % (quality["min_orthogonal_quality"], mesh_config.target_quality)
            )
        return self._mesh_info

    def _launch(self, mode: str):
        import ansys.fluent.core as pyfluent

        return pyfluent.launch_fluent(
            mode=mode,
            precision="double",
            processor_count=4,
            ui_mode="no_gui",
            cwd=self.work_dir,
        )

    @staticmethod
    def _run_task(session, name: str, arguments: Optional[Dict[str, Any]] = None) -> None:
        task = session.workflow.TaskObject[name]
        if arguments:
            task.Arguments.set_state(arguments)
        task.Execute()
        state = str(task.State())
        if state not in ("Up-to-date", "Forced-up-to-date"):
            raise RuntimeError("Fluent görevi tamamlanamadı: %s (durum: %s)" % (name, state))

    def _facet_cad(
        self, surface_path: str, min_size_mm: float, max_size_mm: float, mesh_config: MeshConfig
    ) -> None:
        """Lets Fluent read the CAD natively and writes the body surface mesh."""
        session = self._launch("meshing")
        try:
            session.workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
            self._run_task(session, "Import Geometry", {
                "FileFormat": "CAD",
                "FileName": self.cad_file_path,
                "LengthUnit": "mm",
            })
            self._run_task(session, "Add Local Sizing", {"AddChild": "no"})
            self._run_task(session, "Generate the Surface Mesh", {
                "CFDSurfaceMeshControls": {
                    "MinSize": min_size_mm,
                    "MaxSize": max_size_mm,
                    "GrowthRate": mesh_config.growth_rate,
                    "SizeFunctions": "Curvature",
                    "CurvatureNormalAngle": mesh_config.curvature_normal_angle_deg,
                },
            })
            session.tui.file.write_mesh(surface_path)
        finally:
            try:
                session.exit()
            except Exception:
                pass
        if not os.path.exists(surface_path):
            raise RuntimeError("Fluent gövde yüzey ağını yazamadı: %s" % surface_path)

    def _inflation_layer_count(self, mesh_config: MeshConfig) -> int:
        if mesh_config.inflation_layers:
            return int(mesh_config.inflation_layers)
        if self.inflation_cfg is not None:
            return int(self.inflation_cfg.number_of_layers)
        return 6

    def _fill_volume_mesh(self, domain_path: str, mesh_config: MeshConfig) -> Dict[str, Any]:
        """Runs the Watertight workflow on the tunnel mesh and keeps the session."""
        session = self._launch("meshing")
        self._meshing_session = session
        session.workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
        self._run_task(session, "Import Geometry", {
            "FileFormat": "Mesh",
            "MeshFileName": domain_path,
            "MeshUnit": "mm",
        })
        # The tunnel surface is already the mesh we want; without this Fluent
        # silently remeshes it (RemeshImportedMesh defaults to "All").
        self._run_task(session, "Generate the Surface Mesh", {
            "CFDSurfaceMeshControls": {"RemeshImportedMesh": "None"},
        })
        self._run_task(session, "Describe Geometry", {
            "SetupType": "The geometry consists of both fluid and solid regions and/or voids",
        })
        self._run_task(session, "Update Boundaries")
        self._run_task(session, "Create Regions")
        self._run_task(session, "Update Regions")
        # The first cell height is what sets y+, so it is prescribed from the
        # target y+ instead of letting the mesher pick a transition on its own.
        if self.inflation_cfg is None:
            raise RuntimeError("Önce configure_aerodynamics() ile hedef y+ verilmelidir.")
        self._add_prism_layers(session, mesh_config)
        self._run_task(session, "Generate the Volume Mesh", {
            "VolumeFill": "poly-hexcore",
            "VolumeMeshPreferences": {"ShowVolumeMeshPreferences": False},
        })

        quality = self._read_mesh_quality(session)
        self._solver = session.switch_to_solver()
        self._meshing_session = None
        self._prepare_zones()
        return quality

    def _add_prism_layers(self, session, mesh_config: MeshConfig) -> None:
        """Grows the prism stack on the body walls with the prescribed first-cell height.

        The parent task only takes ``AddChild``; the real controls live on a
        child task that has to be created first.  Without that child the task
        reports "Up-to-date" and no prism layer is ever built, which leaves the
        wall cell size, and therefore y+, to chance.  The first height is what
        sets y+, so it comes from the target y+ rather than a mesher default.
        """
        if self.inflation_cfg is None:
            raise RuntimeError("Önce configure_aerodynamics() ile hedef y+ verilmelidir.")
        workflow = session.workflow
        before = set(str(n) for n in workflow.TaskObject.get_object_names())
        parent = workflow.TaskObject["Add Boundary Layers"]
        parent.AddChildToTask()
        parent.InsertCompoundChildTask()
        created = [str(n) for n in workflow.TaskObject.get_object_names() if str(n) not in before]
        if len(created) != 1:
            raise RuntimeError("Sınır tabaka görevi oluşturulamadı (yeni görevler: %s)." % created)
        child = workflow.TaskObject[created[0]]
        child.Arguments.set_state({
            "BLControlName": "aero-prisms",
            "OffsetMethodType": "uniform",
            "NumberOfLayers": self._inflation_layer_count(mesh_config),
            "FirstHeight": float(self.inflation_cfg.first_layer_height_mm),
            "Rate": mesh_config.growth_rate,
            "FaceScope": {"GrowOn": "only-walls"},
        })
        child.Execute()
        # Executing renames the child after its control name.
        state = str(workflow.TaskObject["aero-prisms"].State())
        if state not in ("Up-to-date", "Forced-up-to-date"):
            raise RuntimeError("Prizma katmanları oluşturulamadı (durum: %s)." % state)

    @staticmethod
    def _read_mesh_quality(session) -> Dict[str, Any]:
        """Reads cell count and quality limits from the meshing session.

        ``get_cell_quality_limits`` answers with
        ``[cell_count, minimum, maximum, average, ...]`` for the measure asked.
        """
        utilities = session.meshing_utilities
        ortho = utilities.get_cell_quality_limits(
            cell_zone_name_pattern="*", measure="Orthogonal Quality"
        )
        skew = utilities.get_cell_quality_limits(
            cell_zone_name_pattern="*", measure="Skewness"
        )
        if not ortho or len(ortho) < 4 or not skew or len(skew) < 4:
            raise RuntimeError(
                "Fluent ağ kalitesi okunamadı (ortogonal: %r, skewness: %r)" % (ortho, skew)
            )
        return {
            "cell_count": int(ortho[0]),
            "min_orthogonal_quality": round(float(ortho[1]), 4),
            "average_orthogonal_quality": round(float(ortho[3]), 4),
            "max_skewness": round(float(skew[2]), 4),
        }

    def _store_reference_dimensions(self, bounding_box_mm: Dict[str, float]) -> None:
        span_x = bounding_box_mm["max_x"] - bounding_box_mm["min_x"]
        span_y = bounding_box_mm["max_y"] - bounding_box_mm["min_y"]
        span_z = bounding_box_mm["max_z"] - bounding_box_mm["min_z"]
        # The flow runs along +X, so the frontal (projected) area is the YZ box.
        box_area_m2 = (span_y / 1000.0) * (span_z / 1000.0)
        if box_area_m2 <= 0.0:
            raise RuntimeError("Gövdenin ön kesit alanı sıfır; CAD dosyası düzlemsel olamaz.")
        if self._area_override_m2 is not None:
            self._reference_area_m2 = float(self._area_override_m2)
            self._reference_area_source = "user"
        else:
            self._reference_area_m2 = box_area_m2
            self._reference_area_source = "faceted_body_bounding_box"
        self._reference_length_m = (
            float(self._length_override_m) if self._length_override_m is not None else span_x / 1000.0
        )
        self._body_center_mm = [
            0.5 * (bounding_box_mm["min_x"] + bounding_box_mm["max_x"]),
            0.5 * (bounding_box_mm["min_y"] + bounding_box_mm["max_y"]),
            0.5 * (bounding_box_mm["min_z"] + bounding_box_mm["max_z"]),
        ]

    # ------------------------------------------------------------------ zones

    def _face_zones(self) -> Dict[str, str]:
        """Maps every boundary face zone to the type it currently carries."""
        conditions = self._solver.settings.setup.boundary_conditions
        found: Dict[str, str] = {}
        for kind in conditions.child_names:
            try:
                container = getattr(conditions, kind)
                for name in container.keys():
                    found[name] = kind
            except Exception:
                continue
        return found

    def _prepare_zones(self) -> None:
        """Renames the fluid cell zone and restores the tunnel zone types.

        A Fluent region is named after the alphabetically first label of its
        bounding faces, so the fluid cell zone steals one of the tunnel face
        zone names and that face zone gets a ``:1`` suffix.  Renaming the cell
        zone frees the original name.  The imported far-field patches also
        arrive as pressure-far-field, which the solver refuses to run with an
        incompressible ideal-gas-free setup, so they are switched to symmetry.
        """
        setup = self._solver.settings.setup
        # The command takes ``zone_name``/``new_name``; PyFluent only warns about
        # an unknown keyword and then ignores the whole call, so the names have
        # to match the schema exactly.
        modify_zones = self._solver.settings.mesh.modify_zones
        for cell_zone in sorted(setup.cell_zone_conditions.fluid.keys()):
            if cell_zone == "fluid-domain":
                continue
            modify_zones.zone_name(zone_name=cell_zone, new_name="fluid-domain")
            if cell_zone + ":1" in self._face_zones():
                modify_zones.zone_name(zone_name=cell_zone + ":1", new_name=cell_zone)

        cell_zones = sorted(setup.cell_zone_conditions.fluid.keys())
        if cell_zones != ["fluid-domain"]:
            raise RuntimeError(
                "Akışkan hücre bölgesi yeniden adlandırılamadı: %s" % cell_zones
            )

        self._zones = self._face_zones()
        farfield = sorted(n for n in self._zones if n.startswith("farfield"))
        self._inlets = sorted(n for n in self._zones if n.startswith("inlet"))
        self._outlets = sorted(n for n in self._zones if n.startswith("outlet"))
        # A closed body splits into a coupled wall pair; the fluid-side wall
        # (the one without the "-shadow" suffix) carries the aerodynamic load.
        self._body_walls = sorted(
            n for n in self._zones
            if n.startswith("body") and not n.endswith("-shadow")
        )
        if not self._body_walls:
            raise RuntimeError("Gövde duvar bölgesi bulunamadı: %s" % sorted(self._zones))
        if not self._inlets or not self._outlets:
            raise RuntimeError("Giriş/çıkış bölgesi bulunamadı: %s" % sorted(self._zones))

        for names, wanted in ((farfield, "symmetry"),
                              (self._inlets, "velocity-inlet"),
                              (self._outlets, "pressure-outlet")):
            todo = [n for n in names if self._zones[n] != wanted.replace("-", "_")]
            if todo:
                setup.boundary_conditions.set_zone_type(zone_list=todo, new_type=wanted)
        self._zones = self._face_zones()

    def apply_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]) -> bool:
        """Records wall conditions; inlet/outlet come from configure_aerodynamics."""
        for bc in boundary_conditions:
            if bc.type == BoundaryConditionType.VELOCITY_INLET and bc.magnitude is not None:
                self.inlet_velocity_m_s = float(UnitEngine.to_si(bc.magnitude).value)
            elif bc.type == BoundaryConditionType.WALL:
                self._wall_boundary_conditions.append(bc)
        return True

    # ----------------------------------------------------------------- solve

    def _apply_physics(self) -> None:
        """Turbulence model, fluid, outlet, reference values and the moment report."""
        if self._solver is None:
            raise RuntimeError("Önce generate_mesh() ile hacim ağı üretilmelidir.")

        setup = self._solver.settings.setup
        model, sub_model = TURBULENCE_MODELS[self.turbulence_model]
        setup.models.viscous.model = model
        if sub_model and model == "k-omega":
            setup.models.viscous.k_omega_model = sub_model

        air = setup.materials.fluid["air"]
        air.density.value = self.density_kg_m3
        air.viscosity.value = self.viscosity_pa_s

        for name in self._outlets:
            setup.boundary_conditions.pressure_outlet[name].momentum.gauge_pressure.value = 0.0

        reference = setup.reference_values
        reference.area = self._reference_area_m2
        reference.length = self._reference_length_m
        reference.density = self.density_kg_m3
        reference.velocity = self.inlet_velocity_m_s
        reference.viscosity = self.viscosity_pa_s

        # Pitching moment about the Z axis (flow +X, lift +Y), taken about the
        # moment centre the caller gave or, failing that, the body's box centre.
        reports = self._solver.settings.solution.report_definitions
        centre_mm = self._moment_center_mm if self._moment_center_mm is not None else self._body_center_mm
        reports.moment["cm-body"] = {}
        reports.moment["cm-body"].zones = self._body_walls
        reports.moment["cm-body"].mom_center = [v / 1000.0 for v in centre_mm]
        reports.moment["cm-body"].mom_axis = [0.0, 0.0, 1.0]
        reports.moment["cm-body"].report_output_type = "Moment Coefficient"
        self._moment_center_used_mm = list(centre_mm)

    def _use_fluid_side_walls(self) -> None:
        """Points the force and moment reports at the wall side that touches fluid cells.

        A closed body splits into a coupled wall pair, ``body:1`` and ``body:1-shadow``.
        Which of the two faces the fluid depends on the face orientation of the mesh, so
        it cannot be told from the names: the dead side has no cells behind it and reads
        density 0 on its faces.  A report on the dead side returns exactly zero forces.
        Needs an initialised solution (density is only defined from then on).
        """
        import numpy as np
        from ansys.fluent.core.fields.field_data_interfaces import ScalarFieldDataRequest

        field_data = self._solver.fields.field_data
        chosen: List[str] = []
        for zone in self._body_walls:
            pair = [zone]
            if zone + "-shadow" in self._zones:
                pair.append(zone + "-shadow")
            live = []
            for name in pair:
                got = field_data.get_field_data(ScalarFieldDataRequest(
                    field_name="density", surfaces=[name], node_value=False, boundary_value=True,
                ))
                values = np.asarray(got[name], dtype=float)
                if values.size and float(values.max()) > 0.0:
                    live.append(name)
            if len(live) != 1:
                raise RuntimeError(
                    "Gövde duvarının akışkana bakan yüzü belirlenemedi: %s için yoğunluğu sıfırdan büyük "
                    "olan yüz sayısı %d (1 olmalı)." % (pair, len(live))
                )
            chosen.append(live[0])
        self._body_walls = sorted(chosen)
        reports = self._solver.settings.solution.report_definitions
        reports.moment["cm-body"].zones = self._body_walls
        for key, container in (("cd-body", reports.drag), ("cl-body", reports.lift),
                               ("fd-body", reports.force), ("fs-body", reports.force)):
            container[key].zones = self._body_walls

    @staticmethod
    def wind_axes(angle_of_attack_deg: float, sideslip_deg: float) -> Dict[str, List[float]]:
        """Drag, lift and side directions of the free stream.

        The stream is tilted by the angle of attack toward +Y and by the sideslip
        toward +Z: ``d = (cos a cos b, sin a cos b, sin b)``.  Lift is the part of
        +Y perpendicular to ``d`` and the side direction is ``d x lift`` (+Z when
        both angles are zero).
        """
        a = math.radians(angle_of_attack_deg)
        b = math.radians(sideslip_deg)
        d = [math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), math.sin(b)]
        lift = [-d[0] * d[1], 1.0 - d[1] * d[1], -d[2] * d[1]]
        norm = math.sqrt(sum(v * v for v in lift))
        lift = [v / norm for v in lift]
        side = [d[1] * lift[2] - d[2] * lift[1], d[2] * lift[0] - d[0] * lift[2], d[0] * lift[1] - d[1] * lift[0]]
        return {"drag": d, "lift": lift, "side": side}

    def _set_flow_direction(self, angle_of_attack_deg: float) -> None:
        """Turns the inlet velocity and the force report vectors to the free-stream angles."""
        setup = self._solver.settings.setup
        axes = self.wind_axes(angle_of_attack_deg, self.sideslip_deg)
        velocity = [self.inlet_velocity_m_s * v for v in axes["drag"]]
        for name in self._inlets:
            momentum = setup.boundary_conditions.velocity_inlet[name].momentum
            momentum.velocity_specification_method = "Components"
            for i in range(3):
                momentum.velocity_components[i].value = velocity[i]

        # Drag, lift and side force are resolved in the free-stream frame, so the
        # report vectors turn with the angles.
        reports = self._solver.settings.solution.report_definitions
        for key, container, vector in (
            ("cd-body", reports.drag, axes["drag"]),
            ("cl-body", reports.lift, axes["lift"]),
            ("fd-body", reports.force, axes["drag"]),
            ("fs-body", reports.force, axes["side"]),
        ):
            container[key] = {}
            container[key].zones = self._body_walls
            container[key].force_vector = vector

    #: Residual level the polar sweep calls converged (Fluent's own default criterion).
    POLAR_RESIDUAL_LIMIT = 1.0e-3

    def _disable_residual_stop(self) -> None:
        """Stops Fluent from ending a run on its residual criteria.

        After the first angle the residuals are already under their criteria, so
        with the criteria active Fluent stops after one iteration at the new
        angle and the force has not moved yet.  The sweep then ends a point on
        force stability instead and judges the residuals itself.
        """
        equations = self._solver.settings.solution.monitor.residual.equations
        for name in equations.keys():
            equations[name].check_convergence = False

    def _iterate(self, max_iterations: int, until_force_stable: bool = False) -> Dict[str, Any]:
        """Iterates in chunks, watching the drag coefficient between them.

        A flat residual alone does not prove the force has settled.  With
        ``until_force_stable`` the run ends when the last three checks agree
        (residual stopping is off in that mode, see ``_disable_residual_stop``).
        """
        chunk = FORCE_CHECK_CHUNK
        start = self._iterations_completed(0)
        force_history: List[Dict[str, float]] = []
        residual_converged = False
        done = 0
        while done < max_iterations:
            step = min(chunk, max_iterations - done)
            before = self._iterations_completed(start + done)
            self._solver.settings.solution.run_calculation.iterate(iter_count=step)
            after = self._iterations_completed(before + step)
            done = after - start
            cd_now = self._flatten_values(
                self._solver.settings.solution.report_definitions.compute(report_defs=["cd-body"]),
                ("cd-body",),
            )["cd-body"]
            force_history.append({"iteration": float(done), "cd": cd_now})
            if until_force_stable:
                if len(force_history) >= 3 and self._force_stability(force_history)["stable"]:
                    break
            elif after - before < step:
                residual_converged = True
                break
        if until_force_stable:
            residuals = self._read_residuals()
            residual_converged = bool(residuals) and max(residuals.values()) < self.POLAR_RESIDUAL_LIMIT
        return {
            "iterations": done,
            "residual_converged": residual_converged,
            "force_history": force_history,
            "force_state": self._force_stability(force_history),
        }

    def solve(self, max_iterations: int = 250) -> Dict[str, Any]:
        """Applies the physics setup and iterates until Fluent stops."""
        self._apply_physics()
        self._set_flow_direction(self.angle_of_attack_deg)

        started = time.time()
        self._solver.settings.solution.initialization.hybrid_initialize()
        self._use_fluid_side_walls()
        run = self._iterate(max_iterations)
        elapsed = time.time() - started

        residuals = self._read_residuals()
        case_path = os.path.join(self.work_dir, "%s.cas.h5" % (self.job_id or "cfd"))
        self._solver.settings.file.write_case_data(file_name=case_path)

        worst_residual = max(residuals.values()) if residuals else None
        iterations_run = run["iterations"]
        residual_converged = run["residual_converged"]
        force_history = run["force_history"]
        force_state = run["force_state"]
        if worst_residual is not None:
            message = ("Fluent yakınsama kriterlerini %d. iterasyonda sağladı "
                       "(en büyük artık %.2e)." % (iterations_run, worst_residual)
                       if residual_converged else
                       "İterasyon limiti doldu; en büyük artık %.2e." % worst_residual)
        else:
            message = "Artık verisi okunamadı."

        both = residual_converged and force_state["stable"] is True
        self._solve_info = {
            "status": "CONVERGED" if both else "COMPLETED",
            "solver": "ANSYS Fluent (3ddp)",
            "turbulence_model": self.turbulence_model,
            "iterations_run": iterations_run,
            "iterations_requested": max_iterations,
            "final_residuals": residuals,
            "residual_convergence": message,
            "residuals_converged": residual_converged,
            "force_convergence": force_state,
            "force_history": force_history,
            "solve_time_s": round(elapsed, 1),
            "case_path": case_path,
        }
        return self._solve_info

    def solve_polar(self, angles_deg: List[float], max_iterations: int = 150) -> List[Dict[str, Any]]:
        """Solves several angles of attack on one mesh, each continuing from the last.

        The first angle starts from a hybrid initialisation; later ones start from
        the previous solution, which is what makes a sweep cheaper than separate
        runs.  Every point reports its own residual and force convergence.
        """
        self._apply_physics()
        self._disable_residual_stop()
        points = []
        for index, angle in enumerate(angles_deg):
            self._set_flow_direction(angle)
            started = time.time()
            if index == 0:
                self._solver.settings.solution.initialization.hybrid_initialize()
                self._use_fluid_side_walls()
            run = self._iterate(max_iterations, until_force_stable=True)
            reports = self._solver.settings.solution.report_definitions
            got = self._flatten_values(
                reports.compute(report_defs=["cd-body", "cl-body", "fd-body", "cm-body"]),
                ("cd-body", "cl-body", "fd-body", "cm-body"),
            )
            cd, cl = got["cd-body"], got["cl-body"]
            points.append({
                "angle_of_attack_deg": float(angle),
                "cd": cd,
                "cl": cl,
                "cm": got["cm-body"],
                "lift_to_drag": cl / cd if abs(cd) > 1e-9 else None,
                "drag_force_n": got["fd-body"],
                "iterations": run["iterations"],
                "residuals_converged": run["residual_converged"],
                "force_stable": run["force_state"]["stable"],
                "solve_time_s": round(time.time() - started, 1),
            })
        self._solve_info = {
            "status": "COMPLETED", "solver": "ANSYS Fluent (3ddp)",
            "turbulence_model": self.turbulence_model, "polar": points,
            "iterations_run": sum(p["iterations"] for p in points),
        }
        return points

    @staticmethod
    def _force_stability(history: List[Dict[str, float]]) -> Dict[str, Any]:
        """Is the drag coefficient flat over the last three checks?

        Stable means the spread of the last three values is under 0.5 % of their
        mean (or 1e-4 absolute, for a near-zero coefficient).  With fewer than
        three checks nothing can be said, and ``stable`` is ``None``.
        """
        if len(history) < 3:
            return {"stable": None, "checks": len(history),
                    "note": "Yakınsama değerlendirmesi için en az 3 kontrol noktası gerekir."}
        last = [h["cd"] for h in history[-3:]]
        mean = sum(last) / 3.0
        spread = max(last) - min(last)
        limit = max(1.0e-4, 0.005 * abs(mean))
        return {"stable": spread <= limit, "checks": len(history), "last_values": last,
                "spread": spread, "allowed_spread": limit}

    def _read_residuals(self) -> Dict[str, float]:
        try:
            _, series = self._solver.monitors.get_monitor_set_data(
                monitor_set_name="residual"
            )
        except Exception:
            return {}
        return {name: float(values[-1]) for name, values in series.items() if len(values)}

    def _iterations_completed(self, requested: int) -> int:
        try:
            iterations, _ = self._solver.monitors.get_monitor_set_data(
                monitor_set_name="residual"
            )
            if len(iterations):
                return int(iterations[-1])
        except Exception:
            pass
        return requested

    # --------------------------------------------------------------- results

    def extract_results(self) -> List[ResultMetric]:
        """Reads the aerodynamic coefficients Fluent computed."""
        if not self._solve_info:
            raise RuntimeError("Önce solve() ile CFD çözümü tamamlanmalıdır.")

        reports = self._solver.settings.solution.report_definitions
        computed = reports.compute(report_defs=["cd-body", "cl-body", "fd-body"])
        values = self._flatten_report(computed)
        cm = self._flatten_values(reports.compute(report_defs=["cm-body"]), ("cm-body",))["cm-body"]
        side_force = self._flatten_values(reports.compute(report_defs=["fs-body"]), ("fs-body",))["fs-body"]
        self._cm_value = cm
        self._y_plus = self._read_y_plus()
        cd = values["cd-body"]
        cl = values["cl-body"]
        drag_force_n = values["fd-body"]
        dynamic_pressure = 0.5 * self.density_kg_m3 * self.inlet_velocity_m_s ** 2
        reynolds = (self.density_kg_m3 * self.inlet_velocity_m_s
                    * self._reference_length_m / self.viscosity_pa_s)

        metrics = [
            ResultMetric(
                metric_name="drag_coefficient_Cd",
                value=round(cd, 6),
                unit="",
                note="Fluent basınç + viskoz sürükleme katsayısı (referans alan %.6f m^2)."
                     % self._reference_area_m2,
            ),
            ResultMetric(
                metric_name="lift_coefficient_Cl",
                value=round(cl, 6),
                unit="",
                note="Fluent taşıma katsayısı, %.2f derece hücum açısında."
                     % self.angle_of_attack_deg,
            ),
            ResultMetric(
                metric_name="total_drag_force",
                value=round(drag_force_n, 6),
                unit="N",
                note="%.1f m/s serbest akışta gövdeye etkiyen toplam sürükleme kuvveti."
                     % self.inlet_velocity_m_s,
            ),
            ResultMetric(
                metric_name="dynamic_pressure",
                value=round(dynamic_pressure, 4),
                unit="Pa",
                note="0.5 * rho * V^2 serbest akış dinamik basıncı.",
            ),
            ResultMetric(
                metric_name="reynolds_number",
                value=round(reynolds, 1),
                unit="",
                note="Gövde uzunluğu (%.4f m) referanslı Reynolds sayısı."
                     % self._reference_length_m,
            ),
        ]
        metrics.append(ResultMetric(
            metric_name="side_force_coefficient_Cy",
            value=round(side_force / (dynamic_pressure * self._reference_area_m2), 6),
            unit="",
            note="Yan kuvvet katsayısı (serbest akış çerçevesinde, %.2f derece yan kayma)." % self.sideslip_deg,
        ))
        metrics.append(ResultMetric(
            metric_name="pitching_moment_coefficient_Cm",
            value=round(cm, 6),
            unit="",
            note="Z ekseni etrafında, merkez %s mm, referans alan ve uzunlukla normalize."
                 % [round(v, 3) for v in getattr(self, "_moment_center_used_mm", self._body_center_mm)],
        ))
        for key, label in (("average", "ortalama"), ("maximum", "en büyük")):
            metrics.append(ResultMetric(
                metric_name="wall_y_plus_%s" % key,
                value=round(self._y_plus[key], 4),
                unit="",
                note="Gövde duvarında %s y+ (%s)." % (label, self._y_plus["assessment"]),
            ))
        if abs(cd) > 1e-9:
            metrics.insert(2, ResultMetric(
                metric_name="lift_to_drag_ratio",
                value=round(cl / cd, 6),
                unit="",
                note="Aerodinamik verimlilik oranı (L/D).",
            ))
        return metrics

    @staticmethod
    def _flatten_report(computed: Any) -> Dict[str, float]:
        """Normalises the nested list Fluent returns from ``compute``."""
        return FluentAdapter._flatten_values(computed, ("cd-body", "cl-body", "fd-body"))

    @staticmethod
    def _flatten_values(computed: Any, required: Any) -> Dict[str, float]:
        values: Dict[str, float] = {}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, (int, float)):
                        values[key] = float(value)
                    elif isinstance(value, (list, tuple)) and value:
                        first = value[0]
                        if isinstance(first, (int, float)):
                            values[key] = float(first)
                        else:
                            walk(first)
                    else:
                        walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)

        walk(computed)
        missing = [k for k in required if k not in values]
        if missing:
            raise RuntimeError(
                "Fluent rapor değerleri okunamadı: %s (dönen veri: %r)" % (missing, computed)
            )
        return values

    def _read_y_plus(self) -> Dict[str, Any]:
        """Wall y+ on the body, read per wall face from the solution.

        Fluent's surface-report averages blend node values and gave a
        wall-average several times larger than every individual face value, so
        the per-face data is read and the statistics are computed here, with
        each face weighted by its area.
        """
        import numpy as np
        from ansys.fluent.core.fields.field_data_interfaces import ScalarFieldDataRequest

        field_data = self._solver.fields.field_data
        values, areas = [], []
        for zone in self._body_walls:
            for name, sink in (("y-plus", values), ("face-area-magnitude", areas)):
                got = field_data.get_field_data(ScalarFieldDataRequest(
                    field_name=name, surfaces=[zone], node_value=False, boundary_value=True,
                ))
                sink.append(np.asarray(got[zone], dtype=float))
        y = np.concatenate(values)
        area = np.concatenate(areas)
        if y.size == 0 or y.size != area.size:
            raise RuntimeError(
                "Duvar y+ verisi okunamadı (y+: %d yüz, alan: %d yüz)." % (y.size, area.size)
            )
        average = float((y * area).sum() / area.sum())
        return {
            "average": average,
            "maximum": float(y.max()),
            "p95": float(np.percentile(y, 95)),
            "faces": int(y.size),
            "fraction_above_5": float((y > 5.0).mean()),
            "target": self.inflation_cfg.target_y_plus if self.inflation_cfg else None,
            "assessment": self._classify_y_plus(average, self.turbulence_model),
        }

    @staticmethod
    def _classify_y_plus(average: float, model: str) -> str:
        low_re_models = ("k-omega-sst", "k-omega", "spalart-allmaras")
        if average <= 5.0:
            return "duvar çözünürlüklü (y+ <= 5)"
        if average < 30.0:
            return "tampon bölge (5 < y+ < 30): %s için önerilmez" % model
        if average <= 300.0:
            if model in low_re_models:
                return "duvar fonksiyonu aralığı (30-300): düşük-Re modelde hedef y+ ~1 değil"
            return "duvar fonksiyonu aralığı (30-300)"
        return "çok yüksek (y+ > 300): sınır tabaka çözülmedi"

    def close(self):
        for session in (self._solver, self._meshing_session):
            if session is None:
                continue
            try:
                session.exit()
            except Exception:
                pass
        self._solver = None
        self._meshing_session = None
