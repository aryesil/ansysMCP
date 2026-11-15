"""
Post-Processing & Visualization Adapter (v2.0)
----------------------------------------------
Generates 3D FEA stress, deformation, and safety factor contour maps (PNG)
using PyVista / Matplotlib and compiles standardized engineering compliance summaries.
"""

import os
from typing import Dict, Any, List, Optional
from ..domain.models import ResultMetric, Material, Quantity
from ..domain.engineering import EngineeringCalculator
from ..domain.units import UnitEngine


class PostProcessingAdapter:
    """Extracts field results, renders contour maps, and computes engineering margins."""

    @staticmethod
    def render_contour_plot(
        job_dir: str,
        job_id: str,
        metric_name: str = "von_mises_stress",
        rst_file_path: Optional[str] = None,
    ) -> str:
        """
        Render a contour map from the real solved field stored in the .rst file.

        The scalar field is read through PyDPF and painted onto the solver's own
        mesh. No field is ever synthesised: without a real result file this
        raises, because a plausible-looking plot of invented data is the most
        dangerous artifact an FEA pipeline can emit.
        """
        from .dpf_engine import DpfExtractionError, DpfResultEngine

        if not rst_file_path or not os.path.exists(rst_file_path):
            raise DpfExtractionError(
                "Kontur haritası için gerçek .rst sonuç dosyası gerekli. "
                "Verilen yol: %r" % rst_file_path
            )

        plot_dir = os.path.join(job_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        output_png = os.path.join(plot_dir, f"{metric_name}.png")

        field_key = "total_deformation" if "deformation" in metric_name else "von_mises"
        field, mesh_region = DpfResultEngine.extract_nodal_field(rst_file_path, field_key)

        if field_key == "von_mises":
            scalars = field.data / 1.0e6
            title = "Equivalent (von Mises) Stress [MPa]"
            cmap = "turbo"
        else:
            raw = field.data
            scalars = [(sum(c * c for c in node) ** 0.5) * 1.0e3 for node in raw]
            title = "Total Deformation [mm]"
            cmap = "plasma"

        import pyvista as pv

        pv.set_plot_theme("document")
        grid = mesh_region.grid
        grid[metric_name] = scalars

        plotter = pv.Plotter(off_screen=True, window_size=[1600, 1000])
        plotter.set_background("white")
        plotter.add_text(f"Job: {job_id} | {title}", font_size=12, color="black")
        plotter.add_mesh(
            grid,
            scalars=metric_name,
            cmap=cmap,
            smooth_shading=True,
            show_edges=False,
            scalar_bar_args={"title": title, "color": "black", "vertical": True},
        )
        plotter.camera_position = "iso"
        plotter.screenshot(output_png)
        plotter.close()

        return output_png

    @staticmethod
    def generate_engineering_summary(
        material: Material,
        results: List[ResultMetric]
    ) -> Dict[str, Any]:
        """
        Calculates safety factors and compliance without unverified AI approval.
        """
        max_stress_metric = next((r for r in results if "von_mises" in r.metric_name), None)
        max_disp_metric = next((r for r in results if "deformation" in r.metric_name), None)

        summary: Dict[str, Any] = {
            "results_overview": [r.model_dump() for r in results],
            "material_name": material.name,
            "yield_strength": f"{material.yield_strength.value} {material.yield_strength.unit}",
        }

        if max_stress_metric:
            stress_q = UnitEngine.convert(
                UnitEngine.to_si(Quantity(value=max_stress_metric.value, unit=max_stress_metric.unit)),
                "MPa"
            )
            sf_calc = EngineeringCalculator.calculate_safety_factor(
                max_von_mises_stress=stress_q,
                yield_strength=material.yield_strength
            )
            summary["safety_assessment"] = sf_calc
            summary["engineering_verdict"] = (
                f"Hesaplanan maksimum Von-Mises gerilmesi ({sf_calc['max_stress']} MPa), "
                f"malzemenin akma mukavemetinin ({sf_calc['yield_strength']} MPa) altındadır. "
                f"Güvenlik Faktörü: {sf_calc['safety_factor']} (Güvenlik Marjı: {sf_calc['margin_of_safety']})."
                if not sf_calc["yield_exceeded"] else
                f"DİKKAT: Hesaplanan maksimum gerilme ({sf_calc['max_stress']} MPa), "
                f"akma mukavemetini ({sf_calc['yield_strength']} MPa) aşmıştır! (Güvenlik Faktörü: {sf_calc['safety_factor']})."
            )

        if max_disp_metric:
            summary["maximum_deformation"] = f"{max_disp_metric.value} {max_disp_metric.unit}"

        caveats: List[str] = []
        if material.orthotropic is not None:
            caveats.append(
                "Malzeme ortotropiktir: Von-Mises tabanlı güvenlik faktörü ve akma karşılaştırması "
                "geçerli değildir. Yönlü dayanım için Tsai-Wu/Hashin gibi bir hasar kriteri gerekir; "
                "bu sürümde uygulanmamıştır. Deplasman ve rijitlik sonuçları geçerlidir."
            )
            summary["engineering_verdict"] = (
                "Ortotropik malzeme: dayanım değerlendirmesi yapılmadı (Von-Mises yönlü dayanım için "
                "geçerli değil). Deplasman sonuçları geçerlidir."
            )
            summary.pop("safety_assessment", None)
        if caveats:
            summary["caveats"] = caveats

        return summary
