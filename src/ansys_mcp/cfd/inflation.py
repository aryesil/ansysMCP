"""
CFD Boundary Layer & Inflation Mesh Engine
------------------------------------------
Calculates y+ compliant first cell heights, prism layer counts, and inflation ratios
for wall-resolved (k-omega SST) and wall-modeled (k-epsilon) turbulence models.
"""

import math
from typing import Dict, Any
from pydantic import BaseModel, Field
from ..domain.models import Quantity
from ..domain.units import UnitEngine
from ..domain.engineering import EngineeringCalculator


class InflationLayerConfig(BaseModel):
    target_y_plus: float
    turbulence_model: str
    first_layer_height_mm: float
    number_of_layers: int
    growth_rate: float
    total_inflation_thickness_mm: float
    transition_ratio: float
    recommended_wall_treatment: str


class InflationEngine:
    """Calculates boundary layer mesh parameters for Fluent Meshing."""

    @classmethod
    def calculate_inflation(
        cls,
        target_y_plus: float = 1.0,
        velocity_m_s: float = 30.0,
        characteristic_length_m: float = 1.0,
        turbulence_model: str = "k-omega-sst",
        growth_rate: float = 1.20,
        num_layers: int = 18
    ) -> InflationLayerConfig:
        """
        Computes the complete boundary layer prism stack parameters.
        """
        vel_q = Quantity(value=velocity_m_s, unit="m/s")
        len_q = Quantity(value=characteristic_length_m, unit="m")

        # Estimate y_H in mm
        y_first_q = EngineeringCalculator.estimate_first_layer_height(
            target_y_plus=target_y_plus,
            velocity=vel_q,
            characteristic_length=len_q
        )
        y_first_mm = y_first_q.value

        # Total thickness delta = y_1 * (r^N - 1) / (r - 1)
        r = growth_rate
        n = num_layers
        total_thickness_mm = y_first_mm * ((r ** n - 1.0) / (r - 1.0))

        if target_y_plus <= 2.0:
            wall_treatment = "Enhanced Wall Treatment / Low-Re Wall Resolved (Viscous Sublayer Captured)"
        elif target_y_plus <= 5.0:
            wall_treatment = "Buffer Layer Transition (Acceptable for k-omega SST with blending)"
        else:
            wall_treatment = "Standard Wall Functions (Log-law region y+ > 30)"

        return InflationLayerConfig(
            target_y_plus=target_y_plus,
            turbulence_model=turbulence_model,
            first_layer_height_mm=round(y_first_mm, 5),
            number_of_layers=num_layers,
            growth_rate=growth_rate,
            total_inflation_thickness_mm=round(total_thickness_mm, 2),
            transition_ratio=0.272,
            recommended_wall_treatment=wall_treatment
        )
