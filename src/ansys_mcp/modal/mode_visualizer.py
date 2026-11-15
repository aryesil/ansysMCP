"""
Mode Shape Renderer
-------------------
Renders deformed mode shapes straight out of a solved ANSYS result file.

Every point of every picture comes from the eigenvector DPF reads off the
``.rst``. There is no synthetic displacement field and no placeholder image: if
the result file is missing or unreadable this module raises, because a plausible
looking mode shape that the solver never produced is worse than no picture.
"""

import os
from typing import Dict, List, Optional


class ModeShapeRenderError(RuntimeError):
    """Raised when real mode shapes cannot be rendered."""


class ModeShapeVisualizer:
    """Renders real mode shapes from a modal ``.rst`` using DPF and PyVista."""

    @staticmethod
    def render_mode_shapes(
        rst_file_path: str,
        out_dir: str,
        num_modes: int = 3,
        job_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Render one PNG per mode, warped by the real eigenvector.

        Args:
            rst_file_path: Solved modal result file.
            out_dir: Directory the PNGs are written to.
            num_modes: How many of the extracted modes to render.
            job_id: Optional label drawn on the image.

        Returns:
            Mapping of ``mode_<n>`` to the PNG path.
        """
        if not rst_file_path or not os.path.exists(rst_file_path):
            raise ModeShapeRenderError(
                "Modal sonuç dosyası (.rst) bulunamadı: %r. Mod şekli üretilemez."
                % rst_file_path
            )

        try:
            import ansys.dpf.core as dpf
            import numpy as np
            import pyvista as pv
        except ImportError as exc:
            raise ModeShapeRenderError(
                "Mod şekli çizimi için ansys-dpf-core, numpy ve pyvista gereklidir."
            ) from exc

        os.makedirs(out_dir, exist_ok=True)

        try:
            model = dpf.Model(dpf.DataSources(rst_file_path))
            frequencies: List[float] = list(
                model.metadata.time_freq_support.time_frequencies.data
            )
        except Exception as exc:
            raise ModeShapeRenderError(
                f"DPF, '{rst_file_path}' modal sonucunu açamadı: {exc}"
            ) from exc

        if not frequencies:
            raise ModeShapeRenderError(
                "Sonuç dosyasında hiç mod yok. Modal çözüm gerçekten tamamlandı mı?"
            )

        pv.set_plot_theme("document")
        grid = model.metadata.meshed_region.grid
        plots: Dict[str, str] = {}

        for mode_index in range(min(num_modes, len(frequencies))):
            mode_number = mode_index + 1
            freq_hz = float(frequencies[mode_index])

            disp = model.results.displacement.on_time_scoping([mode_number]).eval()[0]
            vectors = np.asarray(disp.data, dtype=float)
            if vectors.shape[0] != grid.n_points:
                # DPF scopes the field to the nodes that carry a result, so the
                # vectors are mapped back onto the full mesh by node id.
                full = np.zeros((grid.n_points, 3), dtype=float)
                node_ids = list(model.metadata.meshed_region.nodes.scoping.ids)
                index_of = {nid: i for i, nid in enumerate(node_ids)}
                for local, nid in enumerate(disp.scoping.ids):
                    target = index_of.get(nid)
                    if target is not None:
                        full[target] = vectors[local]
                vectors = full

            magnitude = np.linalg.norm(vectors, axis=1)
            peak = float(magnitude.max()) or 1.0

            # Eigenvectors are normalised to the mass matrix, so their absolute
            # size is arbitrary. Scaling to a fixed fraction of the model size
            # makes the shape visible without implying a real amplitude.
            model_size = float(np.linalg.norm(np.ptp(grid.points, axis=0)))
            warped = grid.copy()
            warped.points = grid.points + vectors * (0.15 * model_size / peak)
            warped["Relative_Modal_Amplitude"] = magnitude / peak

            out_png = os.path.join(out_dir, "mode_%d.png" % mode_number)
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
            plotter.set_background("white")
            title = "Mode %d - %.3f Hz" % (mode_number, freq_hz)
            if job_id:
                title = "%s | %s" % (job_id, title)
            plotter.add_text(title, font_size=12, color="black")
            plotter.add_mesh(
                warped,
                scalars="Relative_Modal_Amplitude",
                cmap="turbo",
                smooth_shading=True,
                scalar_bar_args={
                    "title": "Relative Modal Displacement [-]",
                    "color": "black",
                },
            )
            plotter.camera_position = "iso"
            plotter.screenshot(out_png)
            plotter.close()

            plots["mode_%d" % mode_number] = out_png

        return plots
