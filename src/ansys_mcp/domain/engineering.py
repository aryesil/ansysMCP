"""
Deterministic Engineering Calculator
-------------------------------------
Performs physics equations and engineering validations purely in Python.
Eliminates LLM hallucination in CFD parameters, FEA safety margins, and mesh convergence.
"""

import math
from typing import Dict, Any, List
from .models import Quantity
from .units import UnitEngine


class EngineeringCalculator:
    """
    Pure mathematical and physics calculations for simulation setups.
    """

    @staticmethod
    def calculate_reynolds_number(
        velocity: Quantity,
        characteristic_length: Quantity,
        density: Quantity = Quantity(value=1.225, unit="kg/m3"),
        dynamic_viscosity: Quantity = Quantity(value=1.789e-5, unit="Pa")
    ) -> float:
        """
        Re = (rho * v * L) / mu
        """
        v_si = UnitEngine.to_si(velocity).value
        l_si = UnitEngine.to_si(characteristic_length).value
        rho_si = UnitEngine.to_si(density).value
        mu_si = UnitEngine.to_si(dynamic_viscosity).value

        if mu_si <= 0 or l_si <= 0 or rho_si <= 0 or v_si < 0:
            raise ValueError("Fiziksel parametreler sıfırdan büyük olmalıdır.")

        re = (rho_si * v_si * l_si) / mu_si
        return re

    @staticmethod
    def estimate_first_layer_height(
        target_y_plus: float,
        velocity: Quantity,
        characteristic_length: Quantity,
        density: Quantity = Quantity(value=1.225, unit="kg/m3"),
        dynamic_viscosity: Quantity = Quantity(value=1.789e-5, unit="Pa")
    ) -> Quantity:
        """
        Estimates the inflation first layer height (in mm) for a boundary layer mesh.
        Skin friction coefficient Cf ~ 0.058 * Re^(-0.2) (Turbulent flat plate approximation)
        Wall shear stress tau_w = 0.5 * Cf * rho * v^2
        Friction velocity u_tau = sqrt(tau_w / rho)
        y = (y_plus * mu) / (rho * u_tau)
        """
        re = EngineeringCalculator.calculate_reynolds_number(
            velocity, characteristic_length, density, dynamic_viscosity
        )
        v_si = UnitEngine.to_si(velocity).value
        rho_si = UnitEngine.to_si(density).value
        mu_si = UnitEngine.to_si(dynamic_viscosity).value

        # Turbulent boundary layer skin friction estimate
        cf = 0.058 * (re ** -0.2) if re > 1e4 else 0.664 / math.sqrt(re)
        tau_w = 0.5 * cf * rho_si * (v_si ** 2)
        u_tau = math.sqrt(tau_w / rho_si)

        y_first_m = (target_y_plus * mu_si) / (rho_si * u_tau)
        y_first_mm = y_first_m * 1000.0
        return Quantity(value=round(y_first_mm, 5), unit="mm")

    @staticmethod
    def calculate_safety_factor(
        max_von_mises_stress: Quantity,
        yield_strength: Quantity
    ) -> Dict[str, Any]:
        """
        Safety Factor = Yield Strength / Max Von Mises Stress
        Margin of Safety = Safety Factor - 1
        """
        s_max_pa = UnitEngine.to_si(max_von_mises_stress).value
        s_yield_pa = UnitEngine.to_si(yield_strength).value

        if s_max_pa <= 0:
            return {"safety_factor": float("inf"), "margin_of_safety": float("inf"), "status": "NO_STRESS"}

        sf = s_yield_pa / s_max_pa
        ms = sf - 1.0

        is_yield_exceeded = sf < 1.0

        return {
            "safety_factor": round(sf, 3),
            "margin_of_safety": round(ms, 3),
            "yield_exceeded": is_yield_exceeded,
            "max_stress": UnitEngine.convert(max_von_mises_stress, "MPa").value,
            "yield_strength": UnitEngine.convert(yield_strength, "MPa").value,
            "unit": "MPa"
        }

    @staticmethod
    def evaluate_mesh_convergence(
        mesh_sizes: List[float],
        results: List[float],
        tolerance_percentage: float = 3.0
    ) -> Dict[str, Any]:
        """
        Evaluates convergence across mesh refinements (coarse -> medium -> fine).
        Checks if relative difference between the two finest meshes is within tolerance.
        """
        if len(results) < 2:
            return {"converged": False, "reason": "En az 2 farklı mesh sonucu gereklidir."}

        # Compare last two refinements
        val_previous = results[-2]
        val_latest = results[-1]

        if val_previous == 0:
            rel_diff = 0.0
        else:
            rel_diff = abs((val_latest - val_previous) / val_previous) * 100.0

        is_converged = rel_diff <= tolerance_percentage

        return {
            "converged": is_converged,
            "relative_difference_pct": round(rel_diff, 2),
            "tolerance_pct": tolerance_percentage,
            "finest_value": val_latest,
            "previous_value": val_previous
        }
