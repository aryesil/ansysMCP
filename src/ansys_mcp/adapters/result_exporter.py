"""
Exports a solved result field to CSV (and VTU) straight from the solver's result file.

Values are read by DPF from the ``.rst`` the solver wrote; nothing is interpolated or
invented.  Node coordinates are reported in mm, displacements in mm, stresses in MPa.
"""

import csv
import os
from typing import Any, Dict, List, Optional


class ResultExportError(RuntimeError):
    pass


#: result name -> (DPF result attribute, column names, scale to report units, unit)
FIELD_RESULTS = {
    "displacement": ("displacement", ["ux", "uy", "uz"], 1.0e3, "mm"),
    "stress": ("stress", ["sxx", "syy", "szz", "sxy", "syz", "sxz"], 1.0e-6, "MPa"),
    "elastic_strain": ("elastic_strain", ["exx", "eyy", "ezz", "exy", "eyz", "exz"], 1.0, ""),
    "reaction_force": ("reaction_force", ["fx", "fy", "fz"], 1.0, "N"),
    "temperature": ("temperature", ["temperature"], 1.0, "K"),
}
DERIVED_RESULTS = ("von_mises", "principal_stress", "displacement_magnitude")
SUPPORTED = sorted(set(FIELD_RESULTS) | set(DERIVED_RESULTS))


def list_time_sets(result_file_path: str) -> List[float]:
    """Times (transient) or frequencies/load factors (other types) of every solved set."""
    model = _model(result_file_path)
    support = model.metadata.time_freq_support
    return [float(t) for t in support.time_frequencies.data]


def _model(path: str):
    if not path or not os.path.exists(path):
        raise ResultExportError("Sonuç dosyası bulunamadı: %r." % path)
    try:
        import ansys.dpf.core as dpf
    except ImportError as exc:
        raise ResultExportError("Dışa aktarma için ansys-dpf-core gereklidir.") from exc
    try:
        return dpf.Model(dpf.DataSources(path))
    except Exception as exc:
        raise ResultExportError("DPF sonuç dosyasını açamadı: %s" % exc) from exc


def export_field(result_file_path: str, result: str, out_dir: str, set_number: Optional[int] = None,
                 write_vtu: bool = True) -> Dict[str, Any]:
    """Write one result at one solved set (default: the last) and return a summary with real extrema."""
    if result not in SUPPORTED:
        raise ResultExportError("Bilinmeyen sonuç %r. Desteklenenler: %s." % (result, SUPPORTED))
    try:
        import numpy as np
        import ansys.dpf.core.operators as ops
    except ImportError as exc:
        raise ResultExportError("Dışa aktarma için numpy ve ansys-dpf-core gereklidir.") from exc

    model = _model(result_file_path)
    times = [float(t) for t in model.metadata.time_freq_support.time_frequencies.data]
    count = len(times)
    if count == 0:
        raise ResultExportError("Sonuç dosyasında çözülmüş set yok.")
    chosen = count if set_number is None else set_number
    if isinstance(chosen, bool) or not isinstance(chosen, int) or not (1 <= chosen <= count):
        raise ResultExportError("set_number 1 ile %d arasında bir tam sayı olmalıdır." % count)

    try:
        mesh = model.metadata.meshed_region
        node_ids = [int(i) for i in mesh.nodes.scoping.ids]
        coords = np.asarray(mesh.nodes.coordinates_field.data, dtype=float) * 1.0e3
        index_of = {nid: i for i, nid in enumerate(node_ids)}

        base = {"von_mises": "stress", "principal_stress": "stress",
                "displacement_magnitude": "displacement"}.get(result, result)
        attr, columns, scale, unit = FIELD_RESULTS[base]
        op = getattr(model.results, attr)
        if base in ("stress", "elastic_strain"):
            op = op.on_location("Nodal")
        fields = op.on_time_scoping([chosen]).eval()

        if result == "von_mises":
            field = ops.invariant.von_mises_eqv_fc(fields_container=fields).eval()[0]
            columns, scale, unit = ["von_mises"], 1.0e-6, "MPa"
        elif result == "principal_stress":
            inv = ops.invariant.principal_invariants_fc(fields_container=fields)
            fields_eig = [inv.outputs.fields_eig_1()[0], inv.outputs.fields_eig_2()[0],
                          inv.outputs.fields_eig_3()[0]]
            field = fields_eig[0]
            columns, scale, unit = ["s1", "s2", "s3"], 1.0e-6, "MPa"
        else:
            field = fields[0]
        ids = [int(i) for i in field.scoping.ids]
        if result == "principal_stress":
            data = np.column_stack([np.asarray(f.data, dtype=float).reshape(len(ids), -1)[:, 0] for f in fields_eig])
        else:
            data = np.asarray(field.data, dtype=float).reshape(len(ids), -1)
        data = data * scale
        if result == "displacement_magnitude":
            data = np.linalg.norm(data, axis=1).reshape(-1, 1)
            columns = ["displacement_magnitude"]
    except ResultExportError:
        raise
    except Exception as exc:
        raise ResultExportError("DPF '%s' sonucunu okuyamadı: %s" % (result, exc)) from exc

    if data.shape[1] != len(columns):
        raise ResultExportError("Beklenmeyen bileşen sayısı: %d (beklenen %d)." % (data.shape[1], len(columns)))

    os.makedirs(out_dir, exist_ok=True)
    stem = "%s_set%d" % (result, chosen)
    csv_path = os.path.join(out_dir, stem + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "x_mm", "y_mm", "z_mm"] + columns)
        for row, nid in zip(data, ids):
            pos = index_of.get(nid)
            xyz = coords[pos] if pos is not None else (float("nan"),) * 3
            writer.writerow([nid, *("%.9g" % v for v in xyz), *("%.9g" % v for v in row)])

    vtu_path = None
    if write_vtu:
        try:
            grid = mesh.grid
            point_ids = list(mesh.nodes.scoping.ids)
            full = np.full((grid.n_points, data.shape[1]), np.nan)
            for local, nid in enumerate(ids):
                target = index_of.get(nid)
                if target is not None:
                    full[target] = data[local]
            for k, name in enumerate(columns):
                grid.point_data[name] = full[:, k]
            vtu_path = os.path.join(out_dir, stem + ".vtu")
            grid.save(vtu_path)
        except Exception:
            vtu_path = None            # CSV is the contract; VTU is a convenience.

    extrema = {name: {"min": float(np.nanmin(data[:, k])), "max": float(np.nanmax(data[:, k]))}
               for k, name in enumerate(columns)}
    return {
        "result": result, "set_number": chosen, "sets_available": count, "set_value": times[chosen - 1],
        "unit": unit, "columns": columns, "nodes": len(ids), "extrema": extrema,
        "csv": csv_path, "vtu": vtu_path,
    }
