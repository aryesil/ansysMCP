"""
Multi-Objective Optimizer & Pareto Frontier Engine
--------------------------------------------------
Identifies non-dominated Pareto Frontier design points and ranks trade-offs
using multi-criteria decision making (TOPSIS / Ideal Distance).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .doe_engine import DesignPoint


class OptimizationObjective(BaseModel):
    name: str  # e.g. "mass_kg", "max_von_mises_stress", "drag_coefficient_Cd"
    goal: str = "minimize"  # "minimize" or "maximize"
    weight: float = 1.0


class OptimizationConstraint(BaseModel):
    name: str  # e.g. "max_von_mises_stress", "total_deformation_mm"
    operator: str = "<="  # "<=", ">=", "=="
    limit_value: float


class ParetoStudyReport(BaseModel):
    study_id: str
    total_evaluated_points: int
    feasible_points_count: int
    pareto_optimal_count: int
    objectives: List[OptimizationObjective]
    constraints: List[OptimizationConstraint]
    pareto_frontier: List[DesignPoint]
    best_compromise_design: Optional[DesignPoint] = None
    summary_verdict: str


class ParetoOptimizer:
    """Computes Pareto dominance and multi-objective ranking."""

    @classmethod
    def evaluate_constraints(
        cls,
        point: DesignPoint,
        constraints: List[OptimizationConstraint]
    ) -> bool:
        """Checks whether design point meets all engineering limits."""
        for c in constraints:
            val = point.results.get(c.name)
            if val is None:
                continue
            if c.operator == "<=" and val > c.limit_value:
                return False
            elif c.operator == ">=" and val < c.limit_value:
                return False
            elif c.operator == "==" and abs(val - c.limit_value) > 1e-4:
                return False
        return True

    @classmethod
    def dominates(
        cls,
        p1: DesignPoint,
        p2: DesignPoint,
        objectives: List[OptimizationObjective]
    ) -> bool:
        """
        Returns True if p1 dominates p2 (p1 is strictly better or equal in all objectives
        and strictly better in at least one).
        """
        better_in_at_least_one = False

        for obj in objectives:
            v1 = p1.results.get(obj.name, 0.0)
            v2 = p2.results.get(obj.name, 0.0)

            if obj.goal == "minimize":
                if v1 > v2:
                    return False  # p1 is worse
                elif v1 < v2:
                    better_in_at_least_one = True
            else:  # maximize
                if v1 < v2:
                    return False  # p1 is worse
                elif v1 > v2:
                    better_in_at_least_one = True

        return better_in_at_least_one

    @classmethod
    def extract_pareto_frontier(
        cls,
        study_id: str,
        points: List[DesignPoint],
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint]
    ) -> ParetoStudyReport:
        """
        Filters feasible designs and computes the non-dominated Pareto Frontier.
        """
        # 1. Filter Feasible Points
        feasible_points: List[DesignPoint] = []
        for p in points:
            is_feas = cls.evaluate_constraints(p, constraints)
            p.is_feasible = is_feas
            if is_feas:
                feasible_points.append(p)

        if not feasible_points:
            return ParetoStudyReport(
                study_id=study_id,
                total_evaluated_points=len(points),
                feasible_points_count=0,
                pareto_optimal_count=0,
                objectives=objectives,
                constraints=constraints,
                pareto_frontier=[],
                summary_verdict="HİÇBİR UYGUN TASARIM BULUNAMADI (Tüm tasarım noktaları kısıtları ihlal etti)."
            )

        # 2. Compute Non-dominated Set (Pareto Frontier)
        pareto_set: List[DesignPoint] = []
        for p1 in feasible_points:
            is_dominated = False
            for p2 in feasible_points:
                if p1.point_id != p2.point_id and cls.dominates(p2, p1, objectives):
                    is_dominated = True
                    break

            p1.is_pareto_optimal = not is_dominated
            if not is_dominated:
                pareto_set.append(p1)

        # 3. Best Compromise Design (Weighted Normalized Distance to Utopia)
        best_design = cls._find_best_tradeoff(pareto_set, objectives)

        verdict = (
            f"Parametrik Optimizasyon Tamamlandı: {len(points)} tasarım noktasından "
            f"{len(feasible_points)} tanesi kısıtları sağladı ve {len(pareto_set)} adet "
            f"Pareto-Optimal (baskın) tasarım tespit edildi. "
            f"En ideal dengeli tasarım: DP #{best_design.point_id if best_design else 'Yok'}."
        )

        return ParetoStudyReport(
            study_id=study_id,
            total_evaluated_points=len(points),
            feasible_points_count=len(feasible_points),
            pareto_optimal_count=len(pareto_set),
            objectives=objectives,
            constraints=constraints,
            pareto_frontier=pareto_set,
            best_compromise_design=best_design,
            summary_verdict=verdict
        )

    @staticmethod
    def _find_best_tradeoff(
        pareto_points: List[DesignPoint],
        objectives: List[OptimizationObjective]
    ) -> Optional[DesignPoint]:
        """Finds closest point to normalized utopia point (0,0)."""
        if not pareto_points:
            return None
        if len(pareto_points) == 1:
            return pareto_points[0]

        # Normalize objectives to [0, 1]
        best_point = pareto_points[0]
        min_distance = float("inf")

        for p in pareto_points:
            dist_sq = 0.0
            for obj in objectives:
                val = p.results.get(obj.name, 1.0)
                # Ideal normalized value is minimal
                dist_sq += (val * obj.weight) ** 2
            
            if dist_sq < min_distance:
                min_distance = dist_sq
                best_point = p

        return best_point
