"""
Wind Tunnel Enclosure & Fluid Domain Engine
-------------------------------------------
Computes external aerodynamic computational fluid domain (virtual wind tunnel)
dimensions based on vehicle or airfoil bounding box to ensure blockage ratio < 3%.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FluidDomainBounds(BaseModel):
    inlet_x: float
    outlet_x: float
    bottom_y: float
    top_y: float
    left_z: float
    right_z: float
    blockage_ratio_pct: float
    recommended_mesh_size_farfield_mm: float


class WindTunnelEnclosure:
    """Sizes virtual wind tunnel and categorizes CFD boundary faces."""

    @classmethod
    def compute_domain_dimensions(
        cls,
        cad_bounding_box: Dict[str, float],
        upstream_factor: float = 4.0,   # 4L upstream
        downstream_factor: float = 8.0, # 8L wake downstream
        side_factor: float = 3.5,       # 3.5H / 3.5W sides
        ground_clearance_factor: float = 1.0
    ) -> FluidDomainBounds:
        """
        Computes the enclosure domain around a body [min_x..max_x, min_y..max_y, min_z..max_z].
        Assumes primary flow in +X direction.
        """
        # The box comes from the parsed CAD model.  A missing or degenerate
        # extent means the geometry was never read, and guessing a size here
        # would silently size the tunnel around a body that does not exist.
        required = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
        missing = [key for key in required if cad_bounding_box.get(key) is None]
        if missing:
            raise ValueError(
                "CAD sınır kutusu eksik: %s. Önce modeli okuyun." % ", ".join(missing)
            )

        min_x = float(cad_bounding_box["min_x"])
        max_x = float(cad_bounding_box["max_x"])
        min_y = float(cad_bounding_box["min_y"])
        max_y = float(cad_bounding_box["max_y"])
        min_z = float(cad_bounding_box["min_z"])
        max_z = float(cad_bounding_box["max_z"])

        length = max_x - min_x
        height = max_y - min_y
        width = max_z - min_z
        if min(length, height, width) <= 0.0:
            raise ValueError(
                "CAD sınır kutusu dejenere (uzunluk %.3f, yükseklik %.3f, genişlik %.3f mm)."
                % (length, height, width)
            )

        # Enclosure boundaries
        inlet_x = min_x - (upstream_factor * length)
        outlet_x = max_x + (downstream_factor * length)
        bottom_y = min_y if ground_clearance_factor == 0.0 else min_y - (side_factor * height)
        top_y = max_y + (side_factor * height)
        left_z = min_z - (side_factor * width)
        right_z = max_z + (side_factor * width)

        # Frontal area vs tunnel cross-section
        frontal_area = height * width
        tunnel_area = (top_y - bottom_y) * (right_z - left_z)
        blockage_pct = (frontal_area / tunnel_area) * 100.0 if tunnel_area > 0 else 0.0

        # Farfield mesh size recommendation ~ 10% of vehicle length
        farfield_mesh_size = round(length * 0.08, 1)

        return FluidDomainBounds(
            inlet_x=round(inlet_x, 1),
            outlet_x=round(outlet_x, 1),
            bottom_y=round(bottom_y, 1),
            top_y=round(top_y, 1),
            left_z=round(left_z, 1),
            right_z=round(right_z, 1),
            blockage_ratio_pct=round(blockage_pct, 2),
            recommended_mesh_size_farfield_mm=farfield_mesh_size
        )
