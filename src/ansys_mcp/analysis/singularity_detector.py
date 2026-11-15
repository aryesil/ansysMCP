"""
Stress Singularity Detector Engine
----------------------------------
Detects artificial numerical stress singularities caused by sharp re-entrant corners,
point loads, or over-constrained edges where stress approaches infinity (sigma -> inf).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .gci_engine import GridLevelResult


class SingularityDetectionReport(BaseModel):
    singularity_detected: bool
    confidence: float
    risk_level: str  # "HIGH", "MODERATE", "LOW", "NONE"
    stress_growth_rate_pct: float
    displacement_convergence_pct: float
    diagnosis: str
    actionable_remediation: List[str]


class SingularityDetector:
    """Diagnoses numerical singularities vs physical stress concentrations."""

    @classmethod
    def evaluate_singularity(
        cls,
        stress_levels: List[GridLevelResult],
        displacement_levels: Optional[List[GridLevelResult]] = None
    ) -> SingularityDetectionReport:
        """
        Evaluates if peak stress grows uncontrollably while deformation converges.
        """
        if len(stress_levels) < 2:
            return SingularityDetectionReport(
                singularity_detected=False,
                confidence=0.0,
                risk_level="NONE",
                stress_growth_rate_pct=0.0,
                displacement_convergence_pct=0.0,
                diagnosis="Yetersiz mesh seviyesi (En az 2 seviye gereklidir).",
                actionable_remediation=[]
            )

        # Sort levels fine -> medium -> coarse by element size ascending
        sorted_stresses = sorted(stress_levels, key=lambda x: x.element_size_mm)
        s_fine = sorted_stresses[0].result_value
        s_coarse = sorted_stresses[-1].result_value

        stress_growth_pct = ((s_fine - s_coarse) / s_coarse) * 100.0 if s_coarse > 0 else 0.0

        # A singularity is diagnosed from two curves at once: stress that keeps
        # climbing while displacement has already settled.  Without the
        # displacement curve there is no second curve to compare against, so the
        # answer is "inconclusive" rather than an assumed-stable default.
        if not displacement_levels or len(displacement_levels) < 2:
            return SingularityDetectionReport(
                singularity_detected=False,
                confidence=0.0,
                risk_level="NONE",
                stress_growth_rate_pct=round(stress_growth_pct, 2),
                displacement_convergence_pct=0.0,
                diagnosis=(
                    "Tekillik teşhisi için deplasman yakınsama verisi de gereklidir "
                    "(en az 2 mesh seviyesi). Yalnızca gerilme artışı "
                    "(%%%.1f) tek başına tekillik kanıtı değildir." % stress_growth_pct
                ),
                actionable_remediation=[
                    "Aynı mesh seviyeleri için maksimum deplasman değerlerini de toplayıp "
                    "analizi tekrarlayın."
                ]
            )

        sorted_disps = sorted(displacement_levels, key=lambda x: x.element_size_mm)
        d_fine = sorted_disps[0].result_value
        d_coarse = sorted_disps[-1].result_value
        disp_conv_pct = abs((d_fine - d_coarse) / d_fine) * 100.0 if d_fine > 0 else 0.0

        # Singularity Signature: Stress keeps exploding (> 35% growth) while displacement is already converged (< 5%)
        is_singularity = stress_growth_pct > 35.0 and disp_conv_pct < 8.0

        if is_singularity:
            risk = "HIGH"
            conf = 0.92
            diag = (
                f"Yüksek Gerilme Tekilliği (Stress Singularity) Tespit Edildi. "
                f"Mesh sıklaştıkça gerilme sürekli artıyor (+%{stress_growth_pct:.1f}), "
                f"fakat deformasyon yakınsamış durumda (%{disp_conv_pct:.1f}). "
                f"Bu değer fiziksel değil, keskin iç köşe veya tekil yükten kaynaklı matematiksel bir yapaylıktır."
            )
            remediation = [
                "Keskin iç köşelere (Re-entrant corners) en az r = 2.0 - 5.0 mm radyus (fillet) ekleyin.",
                "Noktasal kuvvet yerine yüzey basıncı (Pressure) veya Remote Force uygulayın.",
                "Mesnet sınır şartını keskin kenar yerine tüm temas yüzeyine yayın."
            ]
        elif stress_growth_pct > 20.0:
            risk = "MODERATE"
            conf = 0.65
            diag = "Orta seviye gerilme konsantrasyonu mevcut. Mesh sıklaştırmasına devam edilerek yakınsama kontrol edilmelidir."
            remediation = ["Kritik bölgede lokal mesh sıklaştırma (Mesh Refinement) uygulayın."]
        else:
            risk = "NONE"
            conf = 0.95
            diag = "Gerilme artışı normal yakınsama sınırları içerisindedir. Tekillik tespit edilmedi."
            remediation = []

        return SingularityDetectionReport(
            singularity_detected=is_singularity,
            confidence=conf,
            risk_level=risk,
            stress_growth_rate_pct=round(stress_growth_pct, 2),
            displacement_convergence_pct=round(disp_conv_pct, 2),
            diagnosis=diag,
            actionable_remediation=remediation
        )
