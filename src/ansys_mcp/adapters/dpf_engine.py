"""
PyDPF Result Extraction Engine
------------------------------
Reads mechanical field data (Von Mises stress, displacement, reaction forces)
out of a real ANSYS ``.rst`` result file using ansys-dpf-core.

There is no synthetic fallback. If the solver has not produced a result file,
or DPF cannot open it, this module raises. A simulation platform that invents
numbers when the solver is absent is worse than one that fails, because the
fabricated values are indistinguishable from real ones downstream.
"""

import os
from typing import Any, Dict, List, Optional

from ..domain.models import ResultMetric


class DpfExtractionError(RuntimeError):
    """Raised when real result data cannot be read from the .rst file."""


class DpfResultEngine:
    """Extracts field data, tensors, and scalar results from ANSYS result files."""

    @classmethod
    def extract_fea_results(
        cls,
        rst_file_path: str,
        applied_load_n: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Read peak Von Mises stress and total deformation from a solved ``.rst``.

        Raises
        ------
        DpfExtractionError
            If the result file is missing, or DPF cannot evaluate it.
        """
        if not rst_file_path or not os.path.exists(rst_file_path):
            raise DpfExtractionError(
                "Sonuç dosyası (.rst) bulunamadı: %r. Çözücü gerçekten çalıştırıldı mı? "
                "Bu motor sentetik değer üretmez." % rst_file_path
            )

        try:
            import ansys.dpf.core as dpf
        except ImportError as exc:
            raise DpfExtractionError(
                "ansys-dpf-core kurulu değil; .rst dosyası okunamaz."
            ) from exc

        try:
            ds = dpf.DataSources(rst_file_path)
            model = dpf.Model(ds)

            stress_op = dpf.operators.result.stress_von_mises(data_sources=ds)
            stress_fields = stress_op.eval()
            max_stress_pa = float(stress_fields[0].data.max())
            min_stress_pa = float(stress_fields[0].data.min())

            disp_op = dpf.operators.result.displacement(data_sources=ds)
            disp_fields = disp_op.eval()
            disp_data = disp_fields[0].data
            # Displacement is a vector field; the magnitude is the norm per node.
            max_disp_m = float(max((sum(c * c for c in node) ** 0.5) for node in disp_data))
        except Exception as exc:
            raise DpfExtractionError(
                f"DPF, '{rst_file_path}' dosyasını değerlendiremedi: {exc}"
            ) from exc

        results: List[ResultMetric] = [
            ResultMetric(
                metric_name="maximum_von_mises_stress",
                value=round(max_stress_pa / 1.0e6, 4),
                unit="MPa",
                note="PyDPF ile gerçek .rst dosyasından okundu.",
            ),
            ResultMetric(
                metric_name="minimum_von_mises_stress",
                value=round(min_stress_pa / 1.0e6, 4),
                unit="MPa",
                note="PyDPF ile gerçek .rst dosyasından okundu.",
            ),
            ResultMetric(
                metric_name="maximum_total_deformation",
                value=round(max_disp_m * 1.0e3, 6),
                unit="mm",
                note="PyDPF ile gerçek .rst dosyasından okundu.",
            ),
        ]

        return {
            "source": "PyDPF_Live",
            "rst_file": rst_file_path,
            "mesh_node_count": int(model.metadata.meshed_region.nodes.n_nodes),
            "mesh_element_count": int(model.metadata.meshed_region.elements.n_elements),
            "results": results,
            "applied_load_n": applied_load_n,
        }

    @classmethod
    def extract_nodal_field(cls, rst_file_path: str, field: str = "von_mises"):
        """
        Return the raw nodal scalar array plus the mesh, for real contour plots.

        Returns
        -------
        tuple
            ``(numpy array of nodal scalars, dpf MeshedRegion)``
        """
        if not rst_file_path or not os.path.exists(rst_file_path):
            raise DpfExtractionError(f"Sonuç dosyası (.rst) bulunamadı: {rst_file_path}")

        import ansys.dpf.core as dpf

        ds = dpf.DataSources(rst_file_path)
        model = dpf.Model(ds)

        if field == "von_mises":
            op = dpf.operators.result.stress_von_mises(data_sources=ds)
        elif field == "total_deformation":
            op = dpf.operators.result.displacement(data_sources=ds)
        else:
            raise DpfExtractionError(f"Bilinmeyen alan: {field}")

        to_nodal = dpf.operators.averaging.to_nodal_fc(fields_container=op.outputs.fields_container())
        fields = to_nodal.eval()
        return fields[0], model.metadata.meshed_region
