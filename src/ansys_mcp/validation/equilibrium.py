"""
Static Equilibrium & Reaction Force Validator
---------------------------------------------
Verifies Newton's First Law of Motion (sum of external forces + reactions = 0)
to ensure the FEA solver has achieved true static equilibrium without artificial singularity.
"""

from typing import Dict, Any, List
from ..domain.models import BoundaryCondition, BoundaryConditionType
from ..domain.units import UnitEngine, Quantity


class StaticEquilibriumValidator:
    """Verifies sum(F_applied) + sum(F_reaction) == 0."""

    @classmethod
    def verify_equilibrium(
        cls,
        applied_bcs: List[BoundaryCondition],
        total_reaction_force_n: float,
        tolerance_pct: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validates static force balance.
        """
        # Sum of applied force magnitudes in Newtons
        total_applied_n = 0.0
        for bc in applied_bcs:
            if bc.type == BoundaryConditionType.FORCE and bc.magnitude:
                n_val = UnitEngine.convert(bc.magnitude, "N").value
                total_applied_n += n_val

        if total_applied_n == 0:
            return {
                "equilibrium_satisfied": True,
                "applied_force_N": 0.0,
                "reaction_force_N": total_reaction_force_n,
                "imbalance_pct": 0.0,
                "status": "PASS (No active force load)"
            }

        force_diff = abs(total_applied_n - total_reaction_force_n)
        imbalance_pct = (force_diff / total_applied_n) * 100.0

        is_satisfied = imbalance_pct <= tolerance_pct

        return {
            "equilibrium_satisfied": is_satisfied,
            "applied_force_N": round(total_applied_n, 2),
            "reaction_force_N": round(total_reaction_force_n, 2),
            "force_difference_N": round(force_diff, 2),
            "imbalance_pct": round(imbalance_pct, 3),
            "tolerance_pct": tolerance_pct,
            "status": "PASS" if is_satisfied else "FAIL (Force imbalance exceeds tolerance)"
        }
