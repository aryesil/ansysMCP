"""
Semantic Geometry & Stable Identity Engine
------------------------------------------
Extracts geometric invariants (area, centroid, normal, bounding box) and matches
semantic selectors with confidence scoring, preventing fragile CAD face ID mismatch.
"""

import math
import hashlib
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .models import SemanticSelector


class GeometryEntity(BaseModel):
    """Immutable representation of a CAD face or body."""
    entity_id: str
    entity_type: str = "face"  # face, edge, body
    area_mm2: float
    centroid: List[float] = Field(..., min_length=3, max_length=3)
    normal_vector: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    bounding_box: Dict[str, float]  # x_min, x_max, y_min, y_max, z_min, z_max


class GeometryInspectionResult(BaseModel):
    cad_file_path: str
    cad_sha256: str
    num_bodies: int
    num_faces: int
    total_volume_mm3: float
    bounding_box: Dict[str, float]
    entities: List[GeometryEntity] = Field(default_factory=list)


class GeometryIntelligence:
    """Methods for CAD inspection, geometric hash calculation, and semantic selection."""

    @staticmethod
    def compute_file_sha256(file_path: str) -> str:
        """Calculates SHA-256 hash of the CAD file for immutable provenance."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CAD dosyası bulunamadı: {file_path}")
        
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_vector_angle_deg(v1: List[float], v2: List[float]) -> float:
        """Computes the angle in degrees between two 3D vectors."""
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        if mag1 == 0 or mag2 == 0:
            return 180.0

        cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_theta))

    @classmethod
    def match_semantic_selector(
        cls,
        selector: SemanticSelector,
        entities: List[GeometryEntity]
    ) -> List[Dict[str, Any]]:
        """
        Matches a semantic selector against a list of geometric entities.
        Computes a confidence score (0.0 - 1.0) based on normal, area, and location.
        """
        matches = []

        for entity in entities:
            score = 1.0
            reasons = []

            # 1. Normal Vector Check
            if selector.normal_vector and entity.normal_vector:
                angle_deg = cls.compute_vector_angle_deg(selector.normal_vector, entity.normal_vector)
                if angle_deg > selector.normal_tolerance_deg:
                    # Penalize based on deviation
                    penalty = min((angle_deg - selector.normal_tolerance_deg) / 90.0, 1.0)
                    score -= penalty * 0.6
                    reasons.append(f"Normal deviation: {angle_deg:.1f}°")
                else:
                    reasons.append("Normal aligned")

            # 2. Area Range Check
            if selector.area_range_mm2:
                min_area, max_area = selector.area_range_mm2
                if entity.area_mm2 < min_area or entity.area_mm2 > max_area:
                    score -= 0.3
                    reasons.append(f"Area {entity.area_mm2:.1f} mm² outside [{min_area}, {max_area}]")

            # 3. Position / Centroid Filter Check
            if selector.position_filter:
                axis = selector.position_filter.get("axis", "z").lower()
                axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
                p_val = entity.centroid[axis_idx]
                p_min = selector.position_filter.get("min", float("-inf"))
                p_max = selector.position_filter.get("max", float("inf"))

                if p_val < p_min or p_val > p_max:
                    score -= 0.4
                    reasons.append(f"Position {axis}={p_val:.1f} outside [{p_min}, {p_max}]")

            final_score = max(round(score, 3), 0.0)

            if final_score >= selector.confidence_threshold:
                matches.append({
                    "entity_id": entity.entity_id,
                    "confidence": final_score,
                    "reasons": reasons,
                    "entity": entity.model_dump()
                })

        # Sort matches by confidence descending
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        return matches
