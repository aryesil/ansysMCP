"""
Wind Tunnel Domain Mesh Builder
-------------------------------
Wraps a Fluent-faceted body surface in a named wind-tunnel box and writes a
single Fluent boundary mesh that the Watertight meshing workflow can fill.

Fluent performs the CAD faceting itself, because it reads STEP/IGES/CATIA/
Parasolid natively and never leaves the needle triangles that generic
tessellators produce on swept and revolved faces -- those wreck the volume
mesher.  This module reads that surface back, builds the tunnel walls around
it, and emits one mesh whose zones are named exactly as the solver setup
expects: ``inlet``, ``outlet``, ``farfield-*`` and ``body``.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .enclosure import FluidDomainBounds

#: Fluent boundary-condition type codes used in the ``(13 ...)`` face sections.
BC_TYPE_CODES = {
    "velocity-inlet": 10,
    "pressure-outlet": 5,
    "wall": 3,
    "symmetry": 7,
}

BODY_ZONE_NAME = "body"
INLET_ZONE_NAME = "inlet"
OUTLET_ZONE_NAME = "outlet"
FARFIELD_ZONE_NAMES = ("farfield-ymin", "farfield-ymax", "farfield-zmin", "farfield-zmax")


def _concat(group, field: str = None) -> np.ndarray:
    keys = sorted(group.keys(), key=lambda k: int(k) if k.isdigit() else k)
    return np.concatenate(
        [np.array(group[k] if field is None else group[k][field]) for k in keys], axis=0
    )


def read_wall_surface(surface_mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the wall face zones of a Fluent HDF5 surface mesh.

    Returns the node coordinates (in the file's own units, millimetres for a
    surface written by the meshing workflow) and the 1-based triangle table.
    """
    import h5py

    with h5py.File(surface_mesh_path, "r") as handle:
        mesh = handle["meshes"]["1"]
        coords = _concat(mesh["nodes"]["coords"])
        topology = mesh["faces"]["zoneTopology"]
        zone_types = np.array(topology["zoneType"])
        min_ids = np.array(topology["minId"])
        max_ids = np.array(topology["maxId"])
        node_counts = _concat(mesh["faces"]["nodes"], "nnodes")
        node_list = _concat(mesh["faces"]["nodes"], "nodes")

    offsets = np.concatenate([[0], np.cumsum(node_counts)])
    triangles: List[Tuple[int, int, int]] = []
    for zone in range(len(zone_types)):
        if int(zone_types[zone]) != BC_TYPE_CODES["wall"]:
            continue
        for face in range(int(min_ids[zone]) - 1, int(max_ids[zone])):
            face_nodes = node_list[offsets[face]:offsets[face + 1]]
            if len(face_nodes) == 3:
                triangles.append(tuple(int(n) for n in face_nodes))
            else:  # fan-triangulate polygons
                for k in range(1, len(face_nodes) - 1):
                    triangles.append(
                        (int(face_nodes[0]), int(face_nodes[k]), int(face_nodes[k + 1]))
                    )

    if not triangles:
        raise ValueError(
            "Fluent yüzey ağında duvar (wall) bölgesi bulunamadı: %s" % surface_mesh_path
        )
    return coords, np.array(triangles, dtype=np.int64)


class NodeTable:
    """Collects unique nodes so neighbouring patches share their boundary nodes."""

    def __init__(self, decimals: int = 6):
        self._index: Dict[Tuple[float, float, float], int] = {}
        self._decimals = decimals
        self.points: List[Tuple[float, float, float]] = []

    def add(self, point) -> int:
        key = (round(float(point[0]), self._decimals),
               round(float(point[1]), self._decimals),
               round(float(point[2]), self._decimals))
        index = self._index.get(key)
        if index is None:
            self.points.append(key)
            index = len(self.points)  # Fluent node ids are 1-based
            self._index[key] = index
        return index


def _grid_patch(nodes: NodeTable, corner, u_vec, v_vec, nu: int, nv: int):
    """Triangulates the planar rectangle spanned by ``u_vec`` and ``v_vec``."""
    corner = np.asarray(corner, dtype=float)
    u_vec = np.asarray(u_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    ids = [[nodes.add(corner + u_vec * (i / nu) + v_vec * (j / nv))
            for j in range(nv + 1)] for i in range(nu + 1)]
    triangles = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = ids[i][j], ids[i + 1][j], ids[i + 1][j + 1], ids[i][j + 1]
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return triangles


def _write_fluent_mesh(path: str, zones, points) -> None:
    total_faces = sum(len(t) for _, _, t in zones)
    with open(path, "w", encoding="ascii") as handle:
        handle.write('(0 "AMPC wind tunnel boundary mesh")\n')
        handle.write("(2 3)\n")
        handle.write("(10 (0 1 %x 0 3))\n" % len(points))
        handle.write("(10 (1 1 %x 1 3)(\n" % len(points))
        for point in points:
            handle.write("%.9e %.9e %.9e\n" % point)
        handle.write("))\n")
        handle.write("(13 (0 1 %x 0))\n" % total_faces)

        first = 1
        zone_id = 2
        metadata = []
        for name, bc_type, triangles in zones:
            last = first + len(triangles) - 1
            handle.write("(13 (%x %x %x %x 3)(\n"
                         % (zone_id, first, last, BC_TYPE_CODES[bc_type]))
            for a, b, c in triangles:
                handle.write("%x %x %x 0 0\n" % (a, b, c))
            handle.write("))\n")
            metadata.append((zone_id, bc_type, name))
            first = last + 1
            zone_id += 1

        handle.write("(45 (1 nodes node)())\n")
        for zone_id, bc_type, name in metadata:
            handle.write("(45 (%d %s %s)())\n" % (zone_id, bc_type, name))


def build_domain_mesh(
    surface_mesh_path: str,
    output_path: str,
    upstream_factor: float = 4.0,
    downstream_factor: float = 8.0,
    side_factor: float = 3.5,
    farfield_cells: int = 6,
) -> Dict[str, Any]:
    """Builds the wind tunnel around a faceted body and writes a Fluent mesh.

    ``farfield_cells`` divides the *narrowest* tunnel direction.  Sizing off the
    narrowest direction matters: a far-field facet wider than the tunnel itself
    leaves the volume mesher nothing to grow into and the fill fails.
    """
    coords, triangles = read_wall_surface(surface_mesh_path)
    used = np.unique(triangles) - 1
    low = coords[used].min(axis=0)
    high = coords[used].max(axis=0)
    size = np.maximum(high - low, 1e-9)

    x_min = low[0] - upstream_factor * size[0]
    x_max = high[0] + downstream_factor * size[0]
    y_min = low[1] - side_factor * size[1]
    y_max = high[1] + side_factor * size[1]
    z_min = low[2] - side_factor * size[2]
    z_max = high[2] + side_factor * size[2]
    len_x, len_y, len_z = x_max - x_min, y_max - y_min, z_max - z_min

    target = min(len_x, len_y, len_z) / max(farfield_cells, 1)
    n_x = max(2, int(round(len_x / target)))
    n_y = max(2, int(round(len_y / target)))
    n_z = max(2, int(round(len_z / target)))

    nodes = NodeTable()
    zones = [
        (INLET_ZONE_NAME, "velocity-inlet",
         _grid_patch(nodes, (x_min, y_min, z_min), (0, len_y, 0), (0, 0, len_z), n_y, n_z)),
        (OUTLET_ZONE_NAME, "pressure-outlet",
         _grid_patch(nodes, (x_max, y_min, z_min), (0, len_y, 0), (0, 0, len_z), n_y, n_z)),
        ("farfield-ymin", "symmetry",
         _grid_patch(nodes, (x_min, y_min, z_min), (len_x, 0, 0), (0, 0, len_z), n_x, n_z)),
        ("farfield-ymax", "symmetry",
         _grid_patch(nodes, (x_min, y_max, z_min), (len_x, 0, 0), (0, 0, len_z), n_x, n_z)),
        ("farfield-zmin", "symmetry",
         _grid_patch(nodes, (x_min, y_min, z_min), (len_x, 0, 0), (0, len_y, 0), n_x, n_y)),
        ("farfield-zmax", "symmetry",
         _grid_patch(nodes, (x_min, y_min, z_max), (len_x, 0, 0), (0, len_y, 0), n_x, n_y)),
    ]
    body_triangles = [tuple(nodes.add(coords[node - 1]) for node in triangle)
                      for triangle in triangles]
    zones.append((BODY_ZONE_NAME, "wall", body_triangles))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _write_fluent_mesh(output_path, zones, nodes.points)

    frontal_area = size[1] * size[2]
    tunnel_area = len_y * len_z
    bounds = FluidDomainBounds(
        inlet_x=round(float(x_min), 3),
        outlet_x=round(float(x_max), 3),
        bottom_y=round(float(y_min), 3),
        top_y=round(float(y_max), 3),
        left_z=round(float(z_min), 3),
        right_z=round(float(z_max), 3),
        blockage_ratio_pct=round(float(frontal_area / tunnel_area * 100.0), 3),
        recommended_mesh_size_farfield_mm=round(float(target), 3),
    )

    return {
        "path": output_path,
        "node_count": len(nodes.points),
        "face_count": sum(len(t) for _, _, t in zones),
        "body_face_count": len(body_triangles),
        "body_bounding_box_mm": {
            "min_x": float(low[0]), "max_x": float(high[0]),
            "min_y": float(low[1]), "max_y": float(high[1]),
            "min_z": float(low[2]), "max_z": float(high[2]),
        },
        "domain_bounds": bounds,
        "zones": [(name, bc, len(t)) for name, bc, t in zones],
    }
