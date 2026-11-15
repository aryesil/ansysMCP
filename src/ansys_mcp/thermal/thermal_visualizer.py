"""
Temperature & Heat Flux Contour Renderer
----------------------------------------
Renders thermal contours straight out of a solved ANSYS thermal result file.

The scalar at every node is the value DPF reads from the ``.rth``/``.rst``. No
field is interpolated between an assumed minimum and maximum, and there is no
placeholder image: without a real result file this module raises.
"""

import os
from typing import Dict, Optional


class ThermalRenderError(RuntimeError):
    """Raised when real thermal contours cannot be rendered."""


class ThermalVisualizer:
    """Renders real temperature and heat-flux contours using DPF and PyVista."""

    @staticmethod
    def render_thermal_plots(
        result_file_path: str,
        out_dir: str,
        job_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Render the temperature field and the total heat flux field.

        Args:
            result_file_path: Solved thermal result file (``.rth`` or ``.rst``).
            out_dir: Directory the PNGs are written to.
            job_id: Optional label drawn on the images.

        Returns:
            Mapping of plot name to PNG path. A field the solver did not write
            is simply absent from the mapping rather than being invented.
        """
        if not result_file_path or not os.path.exists(result_file_path):
            raise ThermalRenderError(
                "Termal sonuç dosyası bulunamadı: %r. Kontur üretilemez."
                % result_file_path
            )

        try:
            import ansys.dpf.core as dpf
            import numpy as np
            import pyvista as pv
        except ImportError as exc:
            raise ThermalRenderError(
                "Termal kontur için ansys-dpf-core, numpy ve pyvista gereklidir."
            ) from exc

        os.makedirs(out_dir, exist_ok=True)

        try:
            ds = dpf.DataSources(result_file_path)
            model = dpf.Model(ds)
            mesh = model.metadata.meshed_region
        except Exception as exc:
            raise ThermalRenderError(
                f"DPF, '{result_file_path}' termal sonucunu açamadı: {exc}"
            ) from exc

        grid = mesh.grid
        node_ids = list(mesh.nodes.scoping.ids)
        index_of = {nid: i for i, nid in enumerate(node_ids)}

        def nodal_scalars(field) -> "np.ndarray":
            """Map a possibly scoped DPF field onto every point of the grid."""
            data = np.asarray(field.data, dtype=float)
            if data.ndim > 1:
                data = np.linalg.norm(data, axis=1)
            if data.shape[0] == grid.n_points:
                return data
            full = np.full(grid.n_points, np.nan, dtype=float)
            for local, nid in enumerate(field.scoping.ids):
                target = index_of.get(nid)
                if target is not None:
                    full[target] = data[local]
            return full

        plots: Dict[str, str] = {}

        def render(name: str, scalars, title: str, bar: str, cmap: str) -> None:
            out_png = os.path.join(out_dir, name + ".png")
            painted = grid.copy()
            painted[bar] = scalars
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
            plotter.set_background("white")
            header = "%s | %s" % (job_id, title) if job_id else title
            plotter.add_text(header, font_size=12, color="black")
            plotter.add_mesh(
                painted,
                scalars=bar,
                cmap=cmap,
                smooth_shading=True,
                scalar_bar_args={"title": bar, "color": "black"},
            )
            plotter.camera_position = "iso"
            plotter.screenshot(out_png)
            plotter.close()
            plots[name] = out_png

        pv.set_plot_theme("document")

        try:
            temperature = model.results.temperature.eval()[0]
        except Exception as exc:
            raise ThermalRenderError(
                f"Sonuç dosyasında sıcaklık alanı okunamadı: {exc}"
            ) from exc

        # DPF reports temperature in kelvin; the contour is drawn in celsius.
        render(
            "temperature_contour",
            nodal_scalars(temperature) - 273.15,
            "Steady-State Temperature Distribution",
            "Temperature [C]",
            "inferno",
        )

        try:
            flux = model.results.thermal_flux_nodal_to_nodal_fc.eval()[0]
        except Exception:
            try:
                flux = model.results.thermal_flux.eval()[0]
            except Exception:
                flux = None

        if flux is not None:
            render(
                "heat_flux_contour",
                nodal_scalars(flux),
                "Total Heat Flux Distribution",
                "Total Heat Flux [W/m2]",
                "hot",
            )

        return plots
