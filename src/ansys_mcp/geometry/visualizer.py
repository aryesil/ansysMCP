"""
3D CAD Preview Visualizer
-------------------------
Renders high-quality 3D preview snapshots of CAD models (STEP, STL) with
highlighted Named Selection faces in offscreen headless mode.
"""

import os
from typing import List, Optional, Dict, Any


class GeometryVisualizer:
    """Headless 3D snapshot renderer using PyVista / Trimesh."""

    @staticmethod
    def render_cad_preview(
        cad_file_path: str,
        output_png_path: str,
        highlight_face_normals: Optional[List[List[float]]] = None
    ) -> str:
        """
        Renders a 3D isometric snapshot of the CAD model.
        """
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

        try:
            import pyvista as pv

            from .cad_parser import CadParser

            pv.set_plot_theme("document")
            # CadParser knows how to tessellate BRep formats such as STEP/IGES,
            # which trimesh cannot read on its own.
            mesh_tri = CadParser.load_mesh(cad_file_path)
            
            # Convert trimesh to PyVista PolyData
            pv_mesh = pv.wrap(mesh_tri)

            plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
            plotter.set_background("white")
            plotter.add_mesh(
                pv_mesh,
                color="#78909c",
                smooth_shading=True,
                show_edges=True,
                edge_color="#37474f",
                opacity=0.92
            )
            plotter.add_axes()
            plotter.camera_position = "iso"
            plotter.screenshot(output_png_path)
            plotter.close()

            return output_png_path

        except Exception as exc:
            # A placeholder image here would be indistinguishable from a real
            # render, so the failure is reported instead.
            if os.path.exists(output_png_path):
                os.remove(output_png_path)
            raise RuntimeError(
                "Geometri görselleştirilemedi (%s): %s" % (cad_file_path, exc)
            ) from exc
