"""
CFD Aerodynamic Post-Processing & Flow Visualizer
-------------------------------------------------
Turns a solved Fluent session into engineering numbers and pictures: aerodynamic
force balance, surface pressure-coefficient contours, velocity pathlines, and
the residual convergence history.

The contours and pathlines are rendered by Fluent's own graphics engine from the
converged field data, and the residual plot is drawn from the solver's monitor
stream.  Nothing here invents a field: if the session has no solution, the call
raises instead of producing a placeholder image.
"""

import os
from typing import Any, Dict, List, Optional

from ..domain.models import ResultMetric


class CfdPostProcessingAdapter:
    """Calculates aerodynamic forces and renders CFD flow field plots."""

    @staticmethod
    def calculate_aerodynamic_performance(
        drag_coeff_cd: float,
        lift_coeff_cl: float,
        velocity_m_s: float = 30.0,
        reference_area_m2: float = 0.50,
        fluid_density: float = 1.225
    ) -> Dict[str, Any]:
        """
        Computes drag force, lift force, and aerodynamic efficiency L/D.
        F_D = 0.5 * rho * v^2 * A_ref * Cd
        F_L = 0.5 * rho * v^2 * A_ref * Cl
        """
        q_inf = 0.5 * fluid_density * (velocity_m_s ** 2)  # Dynamic pressure (Pa)
        drag_force_n = q_inf * reference_area_m2 * drag_coeff_cd
        lift_force_n = q_inf * reference_area_m2 * lift_coeff_cl
        l_over_d = (lift_coeff_cl / drag_coeff_cd) if drag_coeff_cd > 0 else 0.0

        return {
            "drag_coefficient_Cd": round(drag_coeff_cd, 4),
            "lift_coefficient_Cl": round(lift_coeff_cl, 4),
            "lift_to_drag_ratio_LD": round(l_over_d, 3),
            "dynamic_pressure_Pa": round(q_inf, 1),
            "drag_force_N": round(drag_force_n, 2),
            "lift_force_N": round(lift_force_n, 2),
            "inlet_velocity_m_s": velocity_m_s,
            "reference_area_m2": reference_area_m2,
            "evaluation": (
                f"Sürükleme Katsayısı Cd: {drag_coeff_cd:.4f}, Taşıma Katsayısı Cl: {lift_coeff_cl:.4f}. "
                f"Aerodinamik Verimlilik (L/D): {l_over_d:.2f}. "
                f"{velocity_m_s:.1f} m/s hızda toplam aerodinamik direnç kuvveti "
                f"{drag_force_n:.1f} N olarak hesaplanmıştır."
            )
        }

    @classmethod
    def render_cfd_plots(
        cls,
        job_dir: str,
        job_id: str,
        solver,
        body_surfaces: List[str],
        inlet_surfaces: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Renders pressure contour, pathlines, and residuals from a solved session.

        ``solver`` is the live PyFluent solver session that produced the
        solution -- :class:`~ansys_mcp.adapters.fluent.FluentAdapter` exposes it
        after :meth:`solve`.
        """
        if solver is None:
            raise RuntimeError(
                "CFD grafikleri için çözülmüş bir Fluent oturumu gereklidir."
            )
        if not body_surfaces:
            raise RuntimeError("Grafik için gövde yüzeyi listesi boş olamaz.")

        plot_dir = os.path.join(job_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        pressure_png = os.path.join(plot_dir, "pressure_contour.png")
        streamlines_png = os.path.join(plot_dir, "velocity_streamlines.png")
        residuals_png = os.path.join(plot_dir, "residuals_history.png")

        graphics = solver.settings.results.graphics
        picture = graphics.picture
        # Headless Fluent has no graphics window, so some of these settings are
        # inactive; applying the ones that exist still gives a sized PNG.
        cls._set_if_active(picture, "use_window_resolution", False)
        cls._set_if_active(picture, "x_resolution", 1200)
        cls._set_if_active(picture, "y_resolution", 900)
        cls._set_if_active(picture.driver_options, "hardcopy_format", "png")

        cls._render_contour(
            graphics, picture, "cp-body", "pressure-coefficient", body_surfaces, pressure_png
        )
        cls._render_pathlines(
            graphics, picture, "flow-pathlines",
            inlet_surfaces or [], body_surfaces, streamlines_png
        )
        cls._render_residuals(solver, job_id, residuals_png)

        return {
            "pressure_contour": pressure_png,
            "velocity_streamlines": streamlines_png,
            "residuals_history": residuals_png,
        }

    @staticmethod
    def _set_if_active(group, name: str, value) -> None:
        """Applies one graphics setting, skipping it when Fluent has it inactive."""
        child = getattr(group, name, None)
        if child is None:
            return
        try:
            if hasattr(child, "is_active") and not child.is_active():
                return
            child.set_state(value)
        except Exception:
            pass

    @staticmethod
    def _save(picture, path: str) -> None:
        picture.save_picture(file_name=path)
        if not os.path.exists(path):
            raise RuntimeError("Fluent görüntüyü yazamadı: %s" % path)

    @classmethod
    def _render_contour(cls, graphics, picture, name, field, surfaces, path) -> None:
        graphics.contour[name] = {}
        contour = graphics.contour[name]
        contour.field = field
        contour.surfaces_list = surfaces
        contour.filled = True
        contour.node_values = True
        contour.display()
        graphics.views.restore_view(view_name="isometric")
        graphics.views.auto_scale()
        cls._save(picture, path)

    @classmethod
    def _render_pathlines(cls, graphics, picture, name, release_surfaces,
                          body_surfaces, path) -> None:
        if not release_surfaces:
            # Without an inlet to release from, colour the body by velocity
            # instead of silently emitting an empty picture.
            cls._render_contour(
                graphics, picture, "velocity-body", "velocity-magnitude",
                body_surfaces, path
            )
            return
        graphics.pathline[name] = {}
        pathline = graphics.pathline[name]
        pathline.field = "velocity-magnitude"
        pathline.release_from_surfaces = release_surfaces
        pathline.skip = 200
        pathline.display()
        graphics.views.restore_view(view_name="isometric")
        graphics.views.auto_scale()
        cls._save(picture, path)

    @staticmethod
    def _render_residuals(solver, job_id: str, path: str) -> None:
        iterations, series = solver.monitors.get_monitor_set_data(
            monitor_set_name="residual"
        )
        if not len(iterations) or not series:
            raise RuntimeError("Fluent artık (residual) verisi bulunamadı.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(10, 6), dpi=120)
        for label, values in series.items():
            axes.semilogy(iterations[:len(values)], values, label=label, linewidth=1.2)
        axes.set_xlabel("İterasyon")
        axes.set_ylabel("Ölçeklenmiş artık (residual)")
        axes.set_title("Job %s | Fluent yakınsama geçmişi" % job_id)
        axes.grid(True, which="both", linewidth=0.3, alpha=0.6)
        axes.legend(loc="upper right", fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)

    @staticmethod
    def to_result_metrics(performance: Dict[str, Any]) -> List[ResultMetric]:
        """Wraps a performance dictionary as reportable metrics."""
        return [
            ResultMetric(metric_name="drag_force", value=performance["drag_force_N"],
                         unit="N", note="Serbest akış yönündeki toplam kuvvet."),
            ResultMetric(metric_name="lift_force", value=performance["lift_force_N"],
                         unit="N", note="Serbest akışa dik toplam kuvvet."),
            ResultMetric(metric_name="lift_to_drag_ratio",
                         value=performance["lift_to_drag_ratio_LD"], unit="",
                         note="Aerodinamik verimlilik oranı."),
        ]
