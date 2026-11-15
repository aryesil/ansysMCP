"""
Design of Experiments (DoE) & Parameter Space Engine
---------------------------------------------------
Generates multi-dimensional design parameter exploration matrices using
Latin Hypercube Sampling (LHS), Full Factorial, and Sobol-like quasi-random sampling.
"""

import math
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DesignParameter(BaseModel):
    name: str  # e.g. "thickness_mm", "angle_of_attack_deg", "fillet_radius_mm"
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    default_value: float


class DesignPoint(BaseModel):
    point_id: int
    parameters: Dict[str, float]
    evaluated: bool = False
    results: Dict[str, float] = Field(default_factory=dict)
    is_feasible: bool = True
    is_pareto_optimal: bool = False


class DoeEngine:
    """Generates parametric design exploration points."""

    @classmethod
    def generate_latin_hypercube_samples(
        cls,
        parameters: List[DesignParameter],
        num_samples: int = 10,
        seed: Optional[int] = 42
    ) -> List[DesignPoint]:
        """
        Generates Latin Hypercube Sample (LHS) points across n-dimensional parameter space.
        """
        if seed is not None:
            random.seed(seed)

        n_params = len(parameters)
        points: List[DesignPoint] = []

        # Generate stratified bins for each parameter
        param_bins: Dict[str, List[float]] = {}
        for p in parameters:
            delta = (p.max_value - p.min_value) / num_samples
            bins = [p.min_value + (i + random.random()) * delta for i in range(num_samples)]
            random.shuffle(bins)
            param_bins[p.name] = bins

        # Assemble design points
        for i in range(num_samples):
            p_dict: Dict[str, float] = {}
            for p in parameters:
                val = param_bins[p.name][i]
                if p.step_size and p.step_size > 0:
                    val = round(round(val / p.step_size) * p.step_size, 3)
                else:
                    val = round(val, 3)
                p_dict[p.name] = val

            points.append(DesignPoint(point_id=i + 1, parameters=p_dict))

        return points

    @classmethod
    def generate_grid_sweep(
        cls,
        parameters: List[DesignParameter],
        points_per_param: int = 3
    ) -> List[DesignPoint]:
        """
        Generates Full Factorial grid sweep points.
        """
        from itertools import product
        param_ranges = []
        param_names = [p.name for p in parameters]

        for p in parameters:
            delta = (p.max_value - p.min_value) / (points_per_param - 1) if points_per_param > 1 else 0
            vals = [round(p.min_value + i * delta, 3) for i in range(points_per_param)]
            param_ranges.append(vals)

        combinations = list(product(*param_ranges))
        points: List[DesignPoint] = []
        for idx, comb in enumerate(combinations):
            p_dict = {param_names[j]: comb[j] for j in range(len(param_names))}
            points.append(DesignPoint(point_id=idx + 1, parameters=p_dict))

        return points
