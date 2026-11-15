"""
Domain Models for ANSYS AI-MCP Server
-------------------------------------
Defines all strictly typed contracts, quantities, states, and manifests.
"""

from enum import Enum
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator


class JobState(str, Enum):
    CREATED = "CREATED"
    CONFIGURED = "CONFIGURED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    IMPORTING = "IMPORTING"
    MESHING = "MESHING"
    SOLVING = "SOLVING"
    POST_PROCESSING = "POST_PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AnalysisType(str, Enum):
    STATIC_STRUCTURAL = "static_structural"
    MODAL = "modal"
    BUCKLING = "buckling"
    HARMONIC = "harmonic"
    TRANSIENT = "transient"
    RANDOM_VIBRATION = "random_vibration"
    CFD_STEADY = "cfd_steady"
    CFD_TRANSIENT = "cfd_transient"
    THERMAL_STEADY = "thermal_steady"
    THERMAL_TRANSIENT = "thermal_transient"


class Quantity(BaseModel):
    """Explicitly typed physical quantity to eliminate unitless values."""
    value: float
    unit: str

    def __repr__(self) -> str:
        return f"{self.value} {self.unit}"


class SemanticSelector(BaseModel):
    """
    Robust geometric face/edge selector based on geometric invariants
    instead of fragile volatile topological indices.
    """
    name: str = "custom_selection"
    normal_vector: Optional[List[float]] = Field(None, description="Expected normal vector [nx, ny, nz]")
    normal_tolerance_deg: float = Field(5.0, description="Angle tolerance in degrees")
    area_range_mm2: Optional[List[float]] = Field(None, description="[min_area, max_area] in mm²")
    centroid_hint: Optional[List[float]] = Field(None, description="Expected centroid [x, y, z] in mm")
    position_filter: Optional[Dict[str, Any]] = Field(None, description="e.g. {'axis': 'z', 'min': 0, 'max': 10}")
    confidence_threshold: float = Field(0.85, ge=0.0, le=1.0)
    either_direction: bool = Field(
        False,
        description="Match faces whose normal is parallel OR anti-parallel to normal_vector "
        "(for example both end faces of a bar with [1, 0, 0]).",
    )
    multi_face: bool = Field(
        False,
        description="Select every face that meets the confidence threshold instead of only the "
        "best one (hole patterns, split faces, several faces on one side).",
    )


class OrthotropicElasticity(BaseModel):
    """Nine independent elastic constants of an orthotropic material.

    Axes 1, 2, 3 are the model's global X, Y, Z. A tube laid along X with the
    fibres running along it has E1 as its fibre-direction modulus and E2 = E3 as
    the (much lower) transverse modulus.
    """
    ex: Quantity
    ey: Quantity
    ez: Quantity
    nu_xy: float
    nu_yz: float
    nu_xz: float
    gxy: Quantity
    gyz: Quantity
    gxz: Quantity


class PlyStrength(BaseModel):
    """Orthotropic strength limits of a ply: axis 1 is the fibre direction (positive magnitudes)."""
    xt: Quantity
    xc: Quantity
    yt: Quantity
    yc: Quantity
    s12: Quantity
    zt: Optional[Quantity] = None
    zc: Optional[Quantity] = None
    s23: Optional[Quantity] = None
    s13: Optional[Quantity] = None


class Material(BaseModel):
    """Standardized material model with mandatory mechanical properties."""
    name: str = Field(..., min_length=1, max_length=80)
    youngs_modulus: Quantity = Field(..., description="e.g. {'value': 200.0, 'unit': 'GPa'}")
    poissons_ratio: float = Field(..., ge=-1.0, le=1.0)
    yield_strength: Quantity = Field(..., description="e.g. {'value': 250.0, 'unit': 'MPa'}")
    ultimate_strength: Optional[Quantity] = None
    density: Optional[Quantity] = None
    thermal_conductivity: Optional[Quantity] = Field(
        None, description="Isotropic thermal conductivity, e.g. {'value': 205.0, 'unit': 'W/mK'}"
    )
    specific_heat: Optional[Quantity] = Field(
        None, description="Specific heat capacity, e.g. {'value': 900.0, 'unit': 'J/kgK'}"
    )
    thermal_expansion: Optional[Quantity] = Field(
        None, description="Secant coefficient of thermal expansion, e.g. {'value': 2.3e-05, 'unit': '1/K'}"
    )
    ply_strength: Optional[PlyStrength] = Field(
        None, description="Orthotropic stress limits of a composite ply (used by the composite failure tool)."
    )
    orthotropic: Optional[OrthotropicElasticity] = Field(
        None,
        description="When set, Engineering Data receives orthotropic elasticity instead of the "
        "isotropic E and nu (which then only describe the primary direction).",
    )


    @model_validator(mode="after")
    def _isotropic_poisson_range(self) -> "Material":
        """An isotropic Poisson ratio lies in [0, 0.5]; an orthotropic nu_xy may exceed it."""
        if self.orthotropic is None and not (0.0 <= self.poissons_ratio <= 0.5):
            raise ValueError("Poisson oranı 0.0 ile 0.5 arasında olmalıdır: %r" % (self.poissons_ratio,))
        return self

    @field_validator("name")
    @classmethod
    def _name_is_plain_text(cls, value: str) -> str:
        """The name is written into a generated script and an XML document.

        Rejecting anything but letters, digits and ``space _ - . ( ) ,`` keeps a
        name from breaking out of either; it is refused, not rewritten, so the
        name Engineering Data holds is always the name the caller gave.
        """
        bad = sorted({ch for ch in value if not (ch.isalnum() or ch in " _-.(),")})
        if bad:
            raise ValueError("Malzeme adında izin verilmeyen karakter(ler): %s" % "".join(bad))
        return value


class BoundaryConditionType(str, Enum):
    FIXED_SUPPORT = "fixed_support"
    FORCE = "force"
    PRESSURE = "pressure"
    DISPLACEMENT = "displacement"
    REMOTE_FORCE = "remote_force"
    GRAVITY = "gravity"
    MOMENT = "moment"
    FRICTIONLESS_SUPPORT = "frictionless_support"
    CYLINDRICAL_SUPPORT = "cylindrical_support"
    COMPRESSION_ONLY_SUPPORT = "compression_only_support"
    VELOCITY_INLET = "velocity_inlet"
    PRESSURE_OUTLET = "pressure_outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    TEMPERATURE = "temperature"
    CONVECTION = "convection"
    HEAT_FLOW = "heat_flow"
    INTERNAL_HEAT_GENERATION = "internal_heat_generation"


class BoundaryCondition(BaseModel):
    """Defines a constraint or load on a semantic entity."""
    name: str
    type: BoundaryConditionType
    target_selector: SemanticSelector
    magnitude: Optional[Quantity] = None
    direction: Optional[List[float]] = Field(None, description="Unit vector [dx, dy, dz]")
    components_mm: Optional[List[Optional[float]]] = Field(
        None,
        description="Prescribed displacement [ux, uy, uz] in mm for a displacement support; "
        "None in a slot leaves that component free.",
    )
    origin_mm: Optional[List[float]] = Field(
        None, description="Point [x, y, z] in mm a remote force acts from."
    )
    dofs: Optional[Dict[str, str]] = Field(
        None,
        description="Cylindrical support: {'radial'|'axial'|'tangential': 'fixed'|'free'}.",
    )
    film_coefficient: Optional[Quantity] = Field(
        None, description="Convection film coefficient, e.g. {'value': 25.0, 'unit': 'W/m^2/K'}"
    )
    ambient_temperature: Optional[Quantity] = Field(
        None, description="Bulk temperature the convecting fluid is at, e.g. {'value': 25.0, 'unit': 'C'}"
    )


class MeshConfig(BaseModel):
    """Deterministic mesh controls."""
    global_size: Quantity
    min_size: Optional[Quantity] = None
    growth_rate: float = Field(1.2, ge=1.05, le=2.0)
    curvature_normal_angle_deg: float = Field(18.0, ge=5.0, le=45.0)
    target_quality: float = Field(0.20, description="Minimum allowable Orthogonal Quality (0.0 to 1.0)")
    inflation_layers: Optional[int] = None
    inflation_first_height: Optional[Quantity] = None
    #: Structural mesh gate: a mesh outside these limits stops the run instead of solving on it.
    max_skewness: float = Field(0.95, gt=0.0, le=1.0, description="Largest allowed element skewness")
    min_element_quality: float = Field(0.05, ge=0.0, lt=1.0, description="Smallest allowed element quality")


class ValidationCheck(BaseModel):
    check_name: str
    passed: bool
    message: str


class ValidationReport(BaseModel):
    """Strict Gatekeeper report. Analysis CANNOT be solved if valid is False."""
    job_id: str
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checks: List[ValidationCheck] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResultMetric(BaseModel):
    metric_name: str
    value: float
    unit: str
    location_coordinates: Optional[List[float]] = None
    confidence_interval: Optional[List[float]] = None
    note: Optional[str] = None


class AnalysisManifest(BaseModel):
    """Immutable record of the simulation for complete reproducibility."""
    job_id: str
    analysis_type: AnalysisType
    created_at: str
    completed_at: Optional[str] = None
    cad_file_path: str
    cad_sha256: str
    material: Material
    mesh_config: MeshConfig
    boundary_conditions: List[BoundaryCondition]
    results: List[ResultMetric] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    ansys_version: str = "Unknown"
    pyansys_version: str = "Unknown"
    status: JobState = JobState.CREATED
    solver_report: Optional[Dict[str, Any]] = Field(
        None,
        description="Live solver evidence: solve state, .rst path, mesh statistics, "
                    "resolved face ids and the reaction force ANSYS actually computed.",
    )
