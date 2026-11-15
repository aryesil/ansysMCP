"""
Advanced Semantic Matcher Engine
---------------------------------
Matches user/AI semantic intent against CAD topological faces using multi-criteria
geometric invariants (normal, area, centroid, spatial slicing) with confidence scoring.
"""

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .cad_parser import FaceTopology, CadModelSummary
from ..domain.models import SemanticSelector


class MatchCandidate(BaseModel):
    face_id: str
    fingerprint: str
    confidence: float
    is_ambiguous: bool
    reasons: List[str]
    face: FaceTopology


class SemanticMatcher:
    """Evaluates geometric criteria to find corresponding CAD faces with high confidence."""

    @staticmethod
    def vector_angle_deg(v1: List[float], v2: List[float]) -> float:
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        m1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        m2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if m1 == 0 or m2 == 0:
            return 180.0
        cos_theta = max(min(dot / (m1 * m2), 1.0), -1.0)
        return math.degrees(math.acos(cos_theta))

    @classmethod
    def match_selector(
        cls,
        selector: SemanticSelector,
        cad_summary: CadModelSummary
    ) -> List[MatchCandidate]:
        candidates: List[MatchCandidate] = []

        for face in cad_summary.faces:
            score = 1.0
            reasons = []

            # 1. Normal Vector Alignment
            if selector.normal_vector:
                angle = cls.vector_angle_deg(selector.normal_vector, face.normal_vector)
                if angle <= selector.normal_tolerance_deg:
                    reasons.append(f"Normal tam hizalı ({angle:.1f}°)")
                else:
                    penalty = min((angle - selector.normal_tolerance_deg) / 90.0, 1.0)
                    score -= penalty * 0.65
                    reasons.append(f"Normal sapması: {angle:.1f}°")

            # 2. Area Filtering
            if selector.area_range_mm2:
                min_a, max_a = selector.area_range_mm2
                if min_a <= face.area_mm2 <= max_a:
                    reasons.append(f"Alan uyumlu ({face.area_mm2:.1f} mm²)")
                else:
                    score -= 0.35
                    reasons.append(f"Alan aralık dışı ({face.area_mm2:.1f} mm² outside [{min_a}, {max_a}])")

            # 3. Position / Spatial Filter
            if selector.position_filter:
                axis = selector.position_filter.get("axis", "z").lower()
                axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
                p_val = face.centroid[axis_idx]
                p_min = selector.position_filter.get("min", float("-inf"))
                p_max = selector.position_filter.get("max", float("inf"))

                if p_min <= p_val <= p_max:
                    reasons.append(f"Konum {axis}={p_val:.1f} aralıkta [{p_min}, {p_max}]")
                else:
                    score -= 0.40
                    reasons.append(f"Konum {axis}={p_val:.1f} aralık dışı [{p_min}, {p_max}]")

            final_score = max(round(score, 3), 0.0)

            if final_score >= selector.confidence_threshold:
                candidates.append(MatchCandidate(
                    face_id=face.face_id,
                    fingerprint=face.fingerprint,
                    confidence=final_score,
                    is_ambiguous=False,
                    reasons=reasons,
                    face=face
                ))

        # Check for ambiguity: if multiple candidates have identical top score
        if len(candidates) > 1:
            top_score = candidates[0].confidence
            ties = [c for c in candidates if c.confidence == top_score]
            if len(ties) > 1 and top_score > 0.8:
                for c in ties:
                    c.is_ambiguous = True
                    c.reasons.append("Uyarı: Birden fazla simetrik/benzer yüzey eşleşti (Ambiguity).")

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
