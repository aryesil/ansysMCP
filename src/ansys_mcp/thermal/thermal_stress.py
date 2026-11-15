"""
Thermal-Structural Stress & Expansion Coupler
---------------------------------------------
Calculates thermal strain (epsilon = alpha * Delta_T) and coupled thermal stresses
(sigma_th = E * alpha * Delta_T) caused by constrained thermal expansion.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ThermalStressReport(BaseModel):
    max_temperature_c: float
    reference_temp_c: float
    temperature_differential_k: float
    coefficient_of_thermal_expansion: float  # 1/K
    thermal_expansion_strain_pct: float
    youngs_modulus_gpa: float
    yield_strength_mpa: float
    peak_thermal_stress_mpa: float
    thermal_safety_factor: Optional[float]  # None when there is no thermal stress to divide by
    is_yield_exceeded: bool
    structural_integrity_verdict: str


class ThermalStressCoupler:
    """Couples thermal temperature field with structural stress."""

    @classmethod
    def evaluate_thermal_stress(
        cls,
        max_temp_c: float,
        reference_temp_c: float,   # stress-free assembly temperature
        cte_per_k: float,
        youngs_modulus_gpa: float,
        yield_strength_mpa: float,
        constraint_factor: float,  # 1.0 = fully rigid constraint, 0.0 = free expansion
    ) -> ThermalStressReport:
        """
        Computes coupled thermal stress sigma = constraint_factor * E * alpha * |Delta_T|.

        Every material and boundary input is required: a stock aluminium card
        standing in for the real part would give a confident but meaningless
        safety factor. A part colder than its reference is stressed as much as
        one hotter by the same difference, so the magnitude of Delta_T is used.
        """
        delta_t = max_temp_c - reference_temp_c
        thermal_strain = cte_per_k * abs(delta_t)
        thermal_strain_pct = thermal_strain * 100.0

        # E in MPa = youngs_modulus_gpa * 1000
        e_mpa = youngs_modulus_gpa * 1000.0
        thermal_stress_mpa = constraint_factor * e_mpa * thermal_strain
        sf = yield_strength_mpa / thermal_stress_mpa if thermal_stress_mpa > 0 else None
        is_yield = thermal_stress_mpa >= yield_strength_mpa

        if sf is None:
            verdict = (
                "TERMAL GERİLME YOK: Sıcaklık farkı sıfır olduğundan termal genleşme gerilmesi oluşmaz; "
                "emniyet katsayısı tanımsızdır."
            )
        elif is_yield:
            verdict = (
                f"TERMAL AKMA HASARI: Termal genleşme gerilmesi ({thermal_stress_mpa:.1f} MPa) "
                f"akma mukavemetini ({yield_strength_mpa:.1f} MPa) aştı! Parça kalıcı plastik deformasyona uğrayacaktır."
            )
        elif sf < 1.5:
            verdict = (
                f"DÜŞÜK EMNİYET PAYI: Termal gerilme emniyet katsayısı ({sf:.2f} < 1.5). "
                f"Termal genleşme boşlukları (Thermal Expansion Joints) eklenmesi önerilir."
            )
        else:
            verdict = (
                f"TERMAL GÜVENLİ: Termal gerilme ({thermal_stress_mpa:.1f} MPa) emniyetli sınırlar içerisindedir "
                f"(Emniyet Faktörü: {sf:.2f} >= 1.5)."
            )

        return ThermalStressReport(
            max_temperature_c=round(max_temp_c, 2),
            reference_temp_c=round(reference_temp_c, 2),
            temperature_differential_k=round(delta_t, 2),
            coefficient_of_thermal_expansion=cte_per_k,
            thermal_expansion_strain_pct=round(thermal_strain_pct, 4),
            youngs_modulus_gpa=youngs_modulus_gpa,
            yield_strength_mpa=yield_strength_mpa,
            peak_thermal_stress_mpa=round(thermal_stress_mpa, 2),
            thermal_safety_factor=round(sf, 2) if sf is not None else None,
            is_yield_exceeded=is_yield,
            structural_integrity_verdict=verdict
        )
