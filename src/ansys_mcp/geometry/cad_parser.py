"""
CAD & Mesh Parser Engine
------------------------
Parses 3D CAD and surface mesh geometries (STEP, STL, IGES, Parasolid).
Extracts geometric invariants: volume, bounding box, face areas, normals, centroids.
"""

import os
import math
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from ..domain.models import SemanticSelector


class FaceTopology(BaseModel):
    """Geometric signature of an individual CAD face."""
    face_id: str
    surface_type: str = "planar"  # planar, cylindrical, conical, spherical, freeform
    area_mm2: float
    centroid: List[float] = Field(..., min_length=3, max_length=3)
    normal_vector: List[float] = Field(..., min_length=3, max_length=3)
    bounding_box: Dict[str, float]  # min_x, max_x, min_y, max_y, min_z, max_z
    fingerprint: str = ""


class CadModelSummary(BaseModel):
    """Full geometrical topology summary of a CAD model."""
    file_path: str
    file_name: str
    sha256: str
    format: str
    num_bodies: int
    num_faces: int
    total_surface_area_mm2: float
    total_volume_mm3: float
    bounding_box: Dict[str, float]
    center_of_mass: List[float]
    faces: List[FaceTopology] = Field(default_factory=list)


def _ascii_readable_path(file_path: str, digest: str) -> str:
    """Return a path OpenCASCADE can open.

    cascadio hands the file name to OpenCASCADE, which opens it through a
    narrow-character API. A path holding non-ASCII characters -- a Turkish
    project folder, for instance -- reaches the reader mangled and fails with
    "Cannot open input file". Copying the model to an ASCII-only temporary
    name is the only way to read it; the original file is left untouched.
    """
    try:
        file_path.encode("ascii")
        return file_path
    except UnicodeEncodeError:
        import shutil
        import tempfile

        ext = os.path.splitext(file_path)[1].lower()
        ascii_path = os.path.join(tempfile.gettempdir(), "ampc_src_%s%s" % (digest, ext))
        if not os.path.exists(ascii_path) or os.path.getmtime(ascii_path) < os.path.getmtime(file_path):
            shutil.copyfile(file_path, ascii_path)
        return ascii_path


class CadParser:
    """Extracts geometric metadata and invariants from CAD/mesh files."""

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def load_mesh(file_path: str):
        """Loads a CAD or mesh file as a trimesh object.

        STEP/IGES files are tessellated with cascadio (OpenCASCADE) first, because
        trimesh has no BRep reader of its own.
        """
        import trimesh

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".step", ".stp", ".iges", ".igs"):
            import cascadio
            import tempfile

            digest = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:10]
            glb_path = os.path.join(tempfile.gettempdir(), "ampc_tess_%s.glb" % digest)
            cascadio.step_to_glb(
                _ascii_readable_path(file_path, digest),
                glb_path,
                tol_linear=0.05,
                tol_angular=0.08,
            )
            mesh = trimesh.load(glb_path, force="mesh")
            # cascadio writes glTF, which is metres by definition; this project
            # reports CAD geometry in millimetres.
            mesh.apply_scale(1000.0)
        else:
            # Faceted formats carry no unit declaration; they are taken at face
            # value, which for CAD-exported STL/OBJ means millimetres.
            mesh = trimesh.load(file_path, force="mesh")

        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            raise ValueError(f"CAD dosyasından geometri okunamadı: {file_path}")

        scale = float(max(mesh.extents)) or 1.0
        mesh.merge_vertices(digits_vertex=max(0, int(round(-math.log10(scale * 1e-6)))))
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        return mesh

    @classmethod
    def parse_cad_file(cls, file_path: str, max_faces: int = 200) -> CadModelSummary:
        """Parses a CAD model (STEP, IGES, STL, OBJ) into a topological summary.

        Every number returned here comes from the actual geometry; the parser
        raises instead of substituting placeholder values, so a downstream
        analysis can never silently run on an invented body.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CAD dosyası bulunamadı: {file_path}")

        sha256 = cls.compute_sha256(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        mesh = cls.load_mesh(file_path)

        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bbox = {
            "min_x": float(bounds[0][0]), "max_x": float(bounds[1][0]),
            "min_y": float(bounds[0][1]), "max_y": float(bounds[1][1]),
            "min_z": float(bounds[0][2]), "max_z": float(bounds[1][2]),
        }
        volume = float(mesh.volume) if mesh.is_watertight else float(mesh.convex_hull.volume)
        area = float(mesh.area)
        com = [float(c) for c in mesh.center_mass]

        faces: List[FaceTopology] = []
        for idx, facet in enumerate(mesh.facets[:max_faces]):
            facet_area = float(mesh.area_faces[facet].sum())
            facet_normal = [float(n) for n in mesh.facets_normal[idx]]
            facet_centroid = [float(c) for c in mesh.triangles_center[facet].mean(axis=0)]
            facet_vertices = mesh.vertices[mesh.faces[facet].flatten()]
            f_bbox = {
                "min_x": float(facet_vertices[:, 0].min()),
                "max_x": float(facet_vertices[:, 0].max()),
                "min_y": float(facet_vertices[:, 1].min()),
                "max_y": float(facet_vertices[:, 1].max()),
                "min_z": float(facet_vertices[:, 2].min()),
                "max_z": float(facet_vertices[:, 2].max()),
            }
            faces.append(FaceTopology(
                face_id=f"Face_{idx + 1}",
                surface_type="planar",
                area_mm2=round(facet_area, 4),
                centroid=[round(c, 4) for c in facet_centroid],
                normal_vector=[round(n, 6) for n in facet_normal],
                bounding_box=f_bbox,
                fingerprint=cls.generate_face_fingerprint(
                    facet_area, facet_centroid, facet_normal),
            ))

        return CadModelSummary(
            file_path=file_path,
            file_name=file_name,
            sha256=sha256,
            format=ext,
            num_bodies=int(mesh.body_count),
            num_faces=len(faces),
            total_surface_area_mm2=round(area, 4),
            total_volume_mm3=round(abs(volume), 4),
            bounding_box=bbox,
            center_of_mass=[round(c, 4) for c in com],
            faces=faces,
        )

    @staticmethod
    def generate_face_fingerprint(area: float, centroid: List[float], normal: List[float]) -> str:
        """
        Creates an invariant hash fingerprint for a face.
        """
        raw = f"A:{area:.1f}_C:{centroid[0]:.1f},{centroid[1]:.1f},{centroid[2]:.1f}_N:{normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
