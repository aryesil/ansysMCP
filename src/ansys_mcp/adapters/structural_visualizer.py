"""
Von Mises stress and total displacement contours from a solved structural result.

Every scalar is read from the solver's result file through DPF.  Nothing is
interpolated or invented; without a real result file this module raises.  The
maxima are returned next to the images so they can be compared with the
solver-reported metrics.
"""

import os
from typing import Any, Dict, Optional


class StructuralRenderError(RuntimeError):
    pass


class StructuralVisualizer:
    @staticmethod
    def render(result_file_path: str, out_dir: str, job_id: Optional[str] = None,
               deformation_scale: float = 0.0) -> Dict[str, Any]:
        """Renders von Mises stress [MPa] and total displacement [mm] on the mesh.

        ``deformation_scale`` > 0 draws the mesh moved by that multiple of the
        displacement; 0 draws it undeformed.
        """
        if not result_file_path or not os.path.exists(result_file_path):
            raise StructuralRenderError(
                "Yapısal sonuç dosyası bulunamadı: %r. Kontur üretilemez." % result_file_path
            )
        try:
            import ansys.dpf.core as dpf
            import ansys.dpf.core.operators as ops
            import numpy as np
            import pyvista as pv
        except ImportError as exc:
            raise StructuralRenderError("Kontur için ansys-dpf-core, numpy ve pyvista gereklidir.") from exc

        try:
            model = dpf.Model(dpf.DataSources(result_file_path))
            mesh = model.metadata.meshed_region
            stress = model.results.stress.on_location("Nodal").eval()
            von_mises = ops.invariant.von_mises_eqv_fc(fields_container=stress).eval()[0]
            displacement = model.results.displacement().eval()[0]
        except Exception as exc:
            raise StructuralRenderError(
                "DPF '%s' yapısal sonucunu okuyamadı: %s" % (result_file_path, exc)
            ) from exc

        grid = mesh.grid
        node_ids = list(mesh.nodes.scoping.ids)
        index_of = {nid: i for i, nid in enumerate(node_ids)}

        def on_grid(field) -> "np.ndarray":
            data = np.asarray(field.data, dtype=float)
            if data.shape[0] == grid.n_points:
                return data
            full = np.full((grid.n_points,) + data.shape[1:], np.nan)
            for local, nid in enumerate(field.scoping.ids):
                target = index_of.get(nid)
                if target is not None:
                    full[target] = data[local]
            return full

        vm_mpa = on_grid(von_mises) / 1.0e6
        disp_m = on_grid(displacement)
        disp_mm = np.linalg.norm(disp_m, axis=1) * 1.0e3

        os.makedirs(out_dir, exist_ok=True)
        pv.set_plot_theme("document")
        plots: Dict[str, str] = {}

        def draw(name: str, scalars, title: str, bar: str, cmap: str) -> None:
            painted = grid.copy()
            if deformation_scale > 0.0:
                painted.points = np.asarray(painted.points) + deformation_scale * disp_m
            painted[bar] = scalars
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
            plotter.set_background("white")
            header = "%s | %s" % (job_id, title) if job_id else title
            if deformation_scale > 0.0:
                header += " (deformasyon x%g)" % deformation_scale
            plotter.add_text(header, font_size=12, color="black")
            plotter.add_mesh(painted, scalars=bar, cmap=cmap, smooth_shading=True,
                             scalar_bar_args={"title": bar, "color": "black"})
            plotter.camera_position = "iso"
            path = os.path.join(out_dir, name + ".png")
            plotter.screenshot(path)
            plotter.close()
            plots[name] = path

        draw("von_mises_contour", vm_mpa, "Equivalent (von Mises) Stress", "von Mises [MPa]", "jet")
        draw("total_displacement_contour", disp_mm, "Total Displacement", "Total displacement [mm]", "viridis")
        return {
            "plots": plots,
            "field_maxima": {
                "von_mises_mpa": float(np.nanmax(vm_mpa)),
                "total_displacement_mm": float(np.nanmax(disp_mm)),
            },
        }
