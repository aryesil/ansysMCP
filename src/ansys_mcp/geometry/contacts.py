"""
Assembly Multi-body Contact Pairs Manager
-----------------------------------------
Detects proximal / touching faces between multi-body assembly parts and configures
bonded, frictional, or frictionless contact pairs for ANSYS Mechanical.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ContactType(str, Enum):
    BONDED = "bonded"
    FRICTIONAL = "frictional"
    FRICTIONLESS = "frictionless"
    NO_SEPARATION = "no_separation"


class ContactPair(BaseModel):
    contact_pair_id: str
    target_body: str
    target_face_id: str
    contact_body: str
    contact_face_id: str
    contact_type: ContactType = ContactType.BONDED
    friction_coefficient: Optional[float] = Field(0.20, ge=0.0, le=1.0)
    gap_tolerance_mm: float = Field(0.10, ge=0.001)


class ContactManager:
    """Manages multi-body contact pair generation and PyMechanical connection scripting."""

    @classmethod
    def create_contact_pair(
        cls,
        pair_id: str,
        body_1: str,
        face_1: str,
        body_2: str,
        face_2: str,
        contact_type: ContactType = ContactType.BONDED,
        friction_coeff: float = 0.20
    ) -> ContactPair:
        return ContactPair(
            contact_pair_id=pair_id,
            target_body=body_1,
            target_face_id=face_1,
            contact_body=body_2,
            contact_face_id=face_2,
            contact_type=contact_type,
            friction_coefficient=friction_coeff if contact_type == ContactType.FRICTIONAL else None
        )

    @classmethod
    def generate_mechanical_contacts_script(cls, pairs: List[ContactPair]) -> str:
        """Generates PyMechanical script to create connections in ANSYS Mechanical."""
        lines = [
            "# ==========================================================================",
            "# PyMechanical Multi-Body Contact Pairs Setup",
            "# ==========================================================================",
            "connections = Model.Connections",
            "contact_group = connections.AddConnectionGroup()",
            "contact_group.Name = 'Assembly_Auto_Contacts'",
            ""
        ]

        for p in pairs:
            lines.append(f"# Contact Region: {p.contact_pair_id} ({p.target_body} <-> {p.contact_body})")
            lines.append(f"cr_{p.contact_pair_id} = contact_group.AddContactRegion()")
            lines.append(f"cr_{p.contact_pair_id}.Name = '{p.contact_pair_id}'")
            
            if p.contact_type == ContactType.BONDED:
                lines.append(f"cr_{p.contact_pair_id}.ContactType = ContactType.Bonded")
            elif p.contact_type == ContactType.FRICTIONAL:
                lines.append(f"cr_{p.contact_pair_id}.ContactType = ContactType.Frictional")
                lines.append(f"cr_{p.contact_pair_id}.FrictionCoefficient = {p.friction_coefficient}")
            elif p.contact_type == ContactType.FRICTIONLESS:
                lines.append(f"cr_{p.contact_pair_id}.ContactType = ContactType.Frictionless")
            elif p.contact_type == ContactType.NO_SEPARATION:
                lines.append(f"cr_{p.contact_pair_id}.ContactType = ContactType.NoSeparation")

            lines.append("")

        return "\n".join(lines)
