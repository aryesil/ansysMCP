"""
Reads the displacement history of a solved transient analysis from the result file.

The history is taken from the solver's ``.rst`` through DPF, not from the Mechanical
session: evaluating dozens of result times through the session was seen to drop the
connection after a long solve.
"""

import os
from typing import Any, Dict, List, Optional


class TransientReadError(RuntimeError):
    pass


def read_displacement_history(result_file_path: str, max_samples: int = 40) -> List[Dict[str, Any]]:
    """Largest displacement magnitude (m) of the model at each solved time, thinned to ``max_samples``."""
    if not result_file_path or not os.path.exists(result_file_path):
        raise TransientReadError("Geçici analiz sonuç dosyası bulunamadı: %r." % result_file_path)
    try:
        import ansys.dpf.core as dpf
        import numpy as np
    except ImportError as exc:
        raise TransientReadError("Geçmiş okumak için ansys-dpf-core ve numpy gereklidir.") from exc

    try:
        model = dpf.Model(dpf.DataSources(result_file_path))
        times = [float(t) for t in model.metadata.time_freq_support.time_frequencies.data]
        count = len(times)
        if count == 0:
            raise TransientReadError("Sonuç dosyasında zaman kümesi yok.")
        stride = max(1, -(-count // max_samples))
        chosen = list(range(stride - 1, count, stride))
        if chosen[-1] != count - 1:
            chosen.append(count - 1)
        rows: List[Dict[str, Any]] = []
        for index in chosen:
            fields = model.results.displacement.on_time_scoping([index + 1]).eval()
            data = np.asarray(fields[0].data, dtype=float)
            rows.append({
                "time_s": times[index],
                "reported_time_s": times[index],
                "max_displacement_m": float(np.linalg.norm(data, axis=1).max()),
            })
        return rows
    except TransientReadError:
        raise
    except Exception as exc:
        raise TransientReadError("DPF geçici sonucu okuyamadı: %s" % exc) from exc
