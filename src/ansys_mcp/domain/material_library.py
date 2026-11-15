"""
Material library
----------------
Property cards for common engineering materials, so a client does not have to
guess E, density or yield from memory. The values are typical handbook figures
(room temperature) and every card says so; they are a starting point to be
checked against the supplier's datasheet, never a certified value.

Fibre composites are deliberately *not* given a card that looks authoritative.
A carbon-fibre tube is strongly orthotropic, and a single isotropic E describes
it only for loads along the fibres. The entry below carries ``fidelity`` text
that says so, and ``lookup_material`` repeats it in its answer.
"""

from typing import Any, Dict, List, Optional

SOURCE_NOTE = "Typical handbook values at room temperature; confirm against the supplier datasheet."

# Units: E in GPa, yield/ultimate in MPa, density in kg/m^3, k in W/m/K,
# cp in J/kg/K, alpha in 1/K.
_CARDS: Dict[str, Dict[str, Any]] = {
    "structural_steel": {
        "name": "Structural Steel",
        "aliases": ["steel", "s235", "a36", "mild steel", "yapı çeliği", "çelik"],
        "youngs_modulus_gpa": 200.0, "poissons_ratio": 0.30,
        "yield_strength_mpa": 250.0, "ultimate_strength_mpa": 460.0,
        "density_kg_m3": 7850.0,
        "thermal_conductivity_w_mk": 60.5, "specific_heat_j_kgk": 434.0,
        "thermal_expansion_per_k": 1.2e-5,
        "fidelity": "isotropic",
    },
    "stainless_304": {
        "name": "Stainless Steel 304",
        "aliases": ["304", "ss304", "paslanmaz", "stainless steel"],
        "youngs_modulus_gpa": 193.0, "poissons_ratio": 0.29,
        "yield_strength_mpa": 215.0, "ultimate_strength_mpa": 505.0,
        "density_kg_m3": 8000.0,
        "thermal_conductivity_w_mk": 16.2, "specific_heat_j_kgk": 500.0,
        "thermal_expansion_per_k": 1.7e-5,
        "fidelity": "isotropic",
    },
    "aluminum_6061_t6": {
        "name": "Aluminum 6061-T6",
        "aliases": ["6061", "6061-t6", "aluminium", "aluminum", "alüminyum"],
        "youngs_modulus_gpa": 68.9, "poissons_ratio": 0.33,
        "yield_strength_mpa": 276.0, "ultimate_strength_mpa": 310.0,
        "density_kg_m3": 2700.0,
        "thermal_conductivity_w_mk": 167.0, "specific_heat_j_kgk": 896.0,
        "thermal_expansion_per_k": 2.36e-5,
        "fidelity": "isotropic",
    },
    "aluminum_7075_t6": {
        "name": "Aluminum 7075-T6",
        "aliases": ["7075", "7075-t6"],
        "youngs_modulus_gpa": 71.7, "poissons_ratio": 0.33,
        "yield_strength_mpa": 503.0, "ultimate_strength_mpa": 572.0,
        "density_kg_m3": 2810.0,
        "thermal_conductivity_w_mk": 130.0, "specific_heat_j_kgk": 960.0,
        "thermal_expansion_per_k": 2.34e-5,
        "fidelity": "isotropic",
    },
    "titanium_ti6al4v": {
        "name": "Titanium Ti-6Al-4V",
        "aliases": ["ti-6al-4v", "ti6al4v", "titanyum", "titanium", "grade 5"],
        "youngs_modulus_gpa": 113.8, "poissons_ratio": 0.342,
        "yield_strength_mpa": 880.0, "ultimate_strength_mpa": 950.0,
        "density_kg_m3": 4430.0,
        "thermal_conductivity_w_mk": 6.7, "specific_heat_j_kgk": 526.3,
        "thermal_expansion_per_k": 8.6e-6,
        "fidelity": "isotropic",
    },
    "copper": {
        "name": "Copper (annealed)",
        "aliases": ["cu", "bakır"],
        "youngs_modulus_gpa": 110.0, "poissons_ratio": 0.34,
        "yield_strength_mpa": 70.0, "ultimate_strength_mpa": 220.0,
        "density_kg_m3": 8960.0,
        "thermal_conductivity_w_mk": 390.0, "specific_heat_j_kgk": 385.0,
        "thermal_expansion_per_k": 1.7e-5,
        "fidelity": "isotropic",
    },
    "pla": {
        "name": "PLA (3D print, bulk)",
        "aliases": ["pla"],
        "youngs_modulus_gpa": 3.5, "poissons_ratio": 0.36,
        "yield_strength_mpa": 50.0, "ultimate_strength_mpa": 60.0,
        "density_kg_m3": 1250.0,
        "thermal_conductivity_w_mk": 0.13, "specific_heat_j_kgk": 1800.0,
        "thermal_expansion_per_k": 6.8e-5,
        "fidelity": "isotropic idealisation; printed parts are anisotropic and weaker across layers",
    },
    "carbon_fiber_ud_orthotropic": {
        "name": "Carbon Fiber/Epoxy UD (orthotropic)",
        "aliases": ["carbon orthotropic", "karbon ortotropik", "cfrp orthotropic", "ud carbon"],
        "youngs_modulus_gpa": 121.0, "poissons_ratio": 0.27,
        "yield_strength_mpa": 1500.0, "ultimate_strength_mpa": 1500.0,
        "density_kg_m3": 1490.0,
        "thermal_conductivity_w_mk": None, "specific_heat_j_kgk": None,
        "thermal_expansion_per_k": None,
        # Axis 1 (X) is the fibre direction; the transverse plane is isotropic.
        "orthotropic": {
            "ex_gpa": 121.0, "ey_gpa": 8.6, "ez_gpa": 8.6,
            "nu_xy": 0.27, "nu_yz": 0.4, "nu_xz": 0.27,
            "gxy_gpa": 4.7, "gyz_gpa": 3.1, "gxz_gpa": 4.7,
        },
        "fidelity": (
            "ORTHOTROPIC, fibres along global X. Typical unidirectional carbon/epoxy values; ply "
            "orientation is not modelled and Von-Mises strength checks do not apply to it."
        ),
    },
    "carbon_fiber_ud_axial": {
        "name": "Carbon Fiber/Epoxy UD (axial idealisation)",
        "aliases": ["carbon", "carbon fiber", "karbon", "karbon fiber", "cfrp"],
        "youngs_modulus_gpa": 135.0, "poissons_ratio": 0.30,
        "yield_strength_mpa": 600.0, "ultimate_strength_mpa": 600.0,
        "density_kg_m3": 1600.0,
        "thermal_conductivity_w_mk": None, "specific_heat_j_kgk": None,
        "thermal_expansion_per_k": None,
        "fidelity": (
            "ORTHOTROPIC material shown as isotropic. Valid only for bending/tension along the fibres of "
            "a unidirectional tube. Shear, torsion, buckling and any off-axis load need orthotropic data "
            "and a layered section; do not trust those results from this card."
        ),
    },
}


def _normalise(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def list_materials() -> List[str]:
    return [card["name"] for card in _CARDS.values()]


def lookup(name: str) -> Optional[Dict[str, Any]]:
    """Find a card by key, display name or alias; ``None`` when unknown."""
    wanted = _normalise(name)
    for key, card in _CARDS.items():
        names = [key, card["name"]] + list(card.get("aliases", []))
        if wanted in (_normalise(n) for n in names):
            result = dict(card)
            result["key"] = key
            result["source"] = SOURCE_NOTE
            result.pop("aliases", None)
            return result
    return None
