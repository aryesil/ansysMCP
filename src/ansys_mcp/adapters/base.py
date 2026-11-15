"""
Base ANSYS Adapter Interface
-----------------------------
Defines abstract contracts for all simulation solver adapters (Mechanical, Fluent, Meshing).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..domain.models import Material, BoundaryCondition, MeshConfig, ResultMetric


class BaseSimulationAdapter(ABC):
    """Abstract base class for all CAE solver integrations."""

    @abstractmethod
    def initialize_session(self, job_id: str, work_dir: str) -> bool:
        """Launches or connects to the solver session."""
        pass

    @abstractmethod
    def load_geometry(self, cad_file_path: str) -> Dict[str, Any]:
        """Imports CAD file into solver workspace."""
        pass

    @abstractmethod
    def apply_material(self, material: Material) -> bool:
        """Assigns material properties."""
        pass

    @abstractmethod
    def generate_mesh(self, mesh_config: MeshConfig) -> Dict[str, Any]:
        """Generates mesh and returns mesh quality metrics."""
        pass

    @abstractmethod
    def apply_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]) -> bool:
        """Applies supports, forces, pressures, or fluid boundary conditions."""
        pass

    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """Executes the simulation solver."""
        pass

    @abstractmethod
    def extract_results(self) -> List[ResultMetric]:
        """Extracts field results via PyDPF or native solver API."""
        pass

    @abstractmethod
    def close(self):
        """Terminates session and frees licenses."""
        pass
