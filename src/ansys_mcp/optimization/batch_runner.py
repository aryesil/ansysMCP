"""
Parametric Optimization Batch Runner
------------------------------------
Generates design-of-experiments points, stores the response values that real
solver runs produced for them, computes the Pareto frontier, and renders the
trade-off scatter plot.

The runner never invents a response.  A design point only carries numbers once
a solve has been run for it and those numbers have been recorded, either by
passing an ``evaluate`` callable to :meth:`OptimizationBatchRunner.run_study`
or by calling :meth:`OptimizationBatchRunner.record_result` per point.
"""

import json
import os
import math
from typing import Callable, Dict, Any, List, Optional
from .doe_engine import DoeEngine, DesignParameter, DesignPoint
from .optimizer import ParetoOptimizer, OptimizationObjective, OptimizationConstraint, ParetoStudyReport
from ..orchestrator.storage import JobStorage


class OptimizationBatchRunner:
    """Orchestrates parametric optimization studies across simulation jobs."""

    @classmethod
    def run_study(
        cls,
        storage: JobStorage,
        study_id: str,
        parameters: List[DesignParameter],
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint],
        evaluate: Callable[[DesignPoint], Dict[str, float]],
        num_samples: int = 8,
        sampling_method: str = "lhs"
    ) -> Dict[str, Any]:
        """Executes a parametric exploration and Pareto optimization sweep.

        ``evaluate`` is called once per design point and must return the real
        response values for that point, normally by running a solve.  There is
        no built-in surrogate: a study without a solver has nothing to report.
        """
        points = cls._sample(parameters, num_samples, sampling_method)

        # 2. Evaluate Design Points with the caller-supplied solver
        for p in points:
            results = evaluate(p)
            if not isinstance(results, dict) or not results:
                raise RuntimeError(
                    "Tasarım noktası değerlendirilemedi (%s): çözücü sonuç döndürmedi."
                    % p.point_id
                )
            p.results = {k: float(v) for k, v in results.items()}
            p.evaluated = True

        return cls._finalize(storage, study_id, points, objectives, constraints)

    @staticmethod
    def _sample(
        parameters: List[DesignParameter],
        num_samples: int,
        sampling_method: str
    ) -> List[DesignPoint]:
        if sampling_method == "grid":
            return DoeEngine.generate_grid_sweep(parameters, points_per_param=3)
        return DoeEngine.generate_latin_hypercube_samples(parameters, num_samples=num_samples)

    @classmethod
    def _finalize(
        cls,
        storage: JobStorage,
        study_id: str,
        points: List[DesignPoint],
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint]
    ) -> Dict[str, Any]:
        pareto_report = ParetoOptimizer.extract_pareto_frontier(
            study_id=study_id,
            points=points,
            objectives=objectives,
            constraints=constraints
        )
        plot_path = cls._render_pareto_chart(
            storage, study_id, points, pareto_report.pareto_frontier, objectives
        )
        return {
            "status": "success",
            "study_id": study_id,
            "report": pareto_report.model_dump(),
            "pareto_plot": plot_path,
            "all_evaluated_points": [p.model_dump() for p in points]
        }

    @classmethod
    def _render_pareto_chart(
        cls,
        storage: JobStorage,
        study_id: str,
        all_points: List[DesignPoint],
        pareto_points: List[DesignPoint],
        objectives: List[OptimizationObjective]
    ) -> str:
        """Renders the 2D trade-off scatter plot of the first two objectives."""
        if len(objectives) < 2:
            raise RuntimeError(
                "Pareto grafiği için en az iki amaç fonksiyonu gereklidir (%s)." % study_id
            )
        x_name = objectives[0].name
        y_name = objectives[1].name
        out_png = os.path.join(cls.study_dir(storage, study_id), "pareto_tradeoff.png")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

            # Feasible vs Dominated points
            feasible = [p for p in all_points if p.is_feasible]
            x_all = [p.results[x_name] for p in feasible]
            y_all = [p.results[y_name] for p in feasible]

            x_pareto = [p.results[x_name] for p in pareto_points]
            y_pareto = [p.results[y_name] for p in pareto_points]

            ax.scatter(x_all, y_all, color="#90a4ae", s=60, alpha=0.7, label="Dominated Design Points")
            ax.scatter(x_pareto, y_pareto, color="#e53935", s=120, zorder=5, label="Pareto-Optimal Frontier (Best Trade-offs)")
            
            # Connect Pareto frontier line
            sorted_pareto = sorted(zip(x_pareto, y_pareto))
            if len(sorted_pareto) > 1:
                px, py = zip(*sorted_pareto)
                ax.plot(px, py, "--", color="#e53935", alpha=0.8)

            ax.set_title(f"Optimization Study: {study_id} - Pareto Trade-off", fontsize=12, fontweight="bold")
            ax.set_xlabel("%s (%s)" % (x_name, objectives[0].goal), fontsize=10)
            ax.set_ylabel("%s (%s)" % (y_name, objectives[1].goal), fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()

            return out_png

        except Exception as exc:
            raise RuntimeError(
                "Pareto grafiği oluşturulamadı (%s): %s" % (study_id, exc)
            ) from exc

    # ------------------------------------------------------------ persistence

    @staticmethod
    def study_dir(storage: JobStorage, study_id: str) -> str:
        """Returns (and creates) the directory holding a study's state."""
        path = os.path.join(storage.workspace_root, "optimizations", study_id)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def _state_path(cls, storage: JobStorage, study_id: str) -> str:
        return os.path.join(cls.study_dir(storage, study_id), "study.json")

    @classmethod
    def load_study(cls, storage: JobStorage, study_id: str) -> Dict[str, Any]:
        path = cls._state_path(storage, study_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Optimizasyon çalışması bulunamadı: %s. Önce "
                "generate_optimization_design_points ile başlatın." % study_id
            )
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _save_study(cls, storage: JobStorage, study_id: str, state: Dict[str, Any]) -> None:
        with open(cls._state_path(storage, study_id), "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)

    @classmethod
    def create_study(
        cls,
        storage: JobStorage,
        study_id: str,
        parameters: List[DesignParameter],
        num_samples: int = 8,
        sampling_method: str = "lhs"
    ) -> Dict[str, Any]:
        """Generates the design points that still have to be solved."""
        points = cls._sample(parameters, num_samples, sampling_method)
        state = {
            "study_id": study_id,
            "sampling_method": sampling_method,
            "parameters": [p.model_dump() for p in parameters],
            "points": [p.model_dump() for p in points],
            "report": None,
        }
        cls._save_study(storage, study_id, state)
        return state

    @classmethod
    def record_result(
        cls,
        storage: JobStorage,
        study_id: str,
        point_id: int,
        results: Dict[str, float]
    ) -> Dict[str, Any]:
        """Stores the solver responses measured for one design point."""
        if not results:
            raise ValueError("Tasarım noktası için en az bir sonuç değeri gereklidir.")
        state = cls.load_study(storage, study_id)
        for entry in state["points"]:
            if entry["point_id"] == point_id:
                entry["results"] = {k: float(v) for k, v in results.items()}
                entry["evaluated"] = True
                state["report"] = None  # the frontier is stale until recomputed
                cls._save_study(storage, study_id, state)
                return entry
        raise KeyError("Tasarım noktası bulunamadı: %s / %s" % (study_id, point_id))

    @classmethod
    def evaluate_study(
        cls,
        storage: JobStorage,
        study_id: str,
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint]
    ) -> Dict[str, Any]:
        """Computes the Pareto frontier over the recorded design points."""
        state = cls.load_study(storage, study_id)
        points = [DesignPoint(**entry) for entry in state["points"]]
        pending = [p.point_id for p in points if not p.evaluated]
        if pending:
            raise RuntimeError(
                "Şu tasarım noktaları henüz çözülmedi: %s. Her biri için "
                "submit_design_point_results çağırın." % pending
            )
        report = cls._finalize(storage, study_id, points, objectives, constraints)
        state["report"] = report
        cls._save_study(storage, study_id, state)
        return report

    @classmethod
    def get_report(cls, storage: JobStorage, study_id: str) -> Dict[str, Any]:
        """Returns the stored Pareto report, or explains what is still missing."""
        state = cls.load_study(storage, study_id)
        if state.get("report"):
            return state["report"]
        pending = [p["point_id"] for p in state["points"] if not p.get("evaluated")]
        if pending:
            raise RuntimeError(
                "Çalışma henüz değerlendirilmedi (%d nokta bekliyor: %s)."
                % (len(pending), pending)
            )
        raise RuntimeError(
            "Çalışma değerlendirilmedi. evaluate_optimization_study çağırın."
        )
