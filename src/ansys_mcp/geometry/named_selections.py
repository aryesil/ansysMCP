"""
Named Selections Manager
------------------------
Groups matched semantic faces into ANSYS Named Selections (Component Groups).
Generates PyMechanical (NamedSelection) and PyFluent boundary zone scripts.
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field
from .semantic_matcher import MatchCandidate


class NamedSelectionGroup(BaseModel):
    name: str
    target_faces: List[str]  # Face IDs
    fingerprints: List[str]
    confidence_avg: float
    description: str = ""


class NamedSelectionsContainer(BaseModel):
    job_id: str
    groups: Dict[str, NamedSelectionGroup] = Field(default_factory=dict)

    def add_group(self, name: str, candidates: List[MatchCandidate], description: str = ""):
        if not candidates:
            return
        face_ids = [c.face_id for c in candidates]
        fps = [c.fingerprint for c in candidates]
        avg_conf = sum(c.confidence for c in candidates) / len(candidates)

        self.groups[name] = NamedSelectionGroup(
            name=name,
            target_faces=face_ids,
            fingerprints=fps,
            confidence_avg=round(avg_conf, 3),
            description=description
        )

    def export_mechanical_script(self) -> str:
        """Generates PyMechanical script to create Named Selections."""
        lines = ["# PyMechanical Named Selections Setup", "model = Model", "sel_group = model.NamedSelections"]
        for g_name, grp in self.groups.items():
            lines.append(f"# Group: {g_name} ({len(grp.target_faces)} faces)")
            lines.append(f"ns_{g_name} = sel_group.AddNamedSelection()")
            lines.append(f"ns_{g_name}.Name = '{g_name}'")
        return "\n".join(lines)
