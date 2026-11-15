"""
Resonance Risk & Excitation Spectrum Evaluator
----------------------------------------------
Evaluates resonance risk between structural natural frequencies (f_n) and
operational excitation frequencies (Motor RPM, blade passing, aerodynamic vortex shedding).
Enforces ISO / API standard 15% - 20% frequency separation margin.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .modal_engine import ModeShapeResult


class ResonanceCheckItem(BaseModel):
    mode_number: int
    natural_frequency_hz: float
    excitation_source: str
    excitation_frequency_hz: float
    separation_margin_pct: float
    risk_level: str  # "CRITICAL_RESONANCE", "WARNING", "SAFE"
    diagnosis: str


class ResonanceAssessmentReport(BaseModel):
    overall_resonance_risk: str  # "HIGH", "MODERATE", "LOW"
    minimum_separation_margin_pct: float
    safety_threshold_pct: float = 15.0
    critical_modes_count: int
    detailed_checks: List[ResonanceCheckItem]
    recommendation: str


class ResonanceEvaluator:
    """Evaluates resonance risk against excitation frequency bands."""

    @classmethod
    def evaluate_resonance(
        cls,
        modes: List[ModeShapeResult],
        operating_rpms: Optional[List[float]] = None,
        discrete_excitation_frequencies_hz: Optional[List[float]] = None,
        min_safe_margin_pct: float = 15.0
    ) -> ResonanceAssessmentReport:
        """
        Compares all natural frequencies against operational excitations.
        f_rpm = RPM / 60
        """
        excitations: List[Dict[str, Any]] = []

        # 1. RPM Frequencies (1X and 2X harmonics)
        if operating_rpms:
            for rpm in operating_rpms:
                f_1x = rpm / 60.0
                f_2x = f_1x * 2.0
                excitations.append({"name": f"Motor 1X RPM ({rpm:.0f} RPM)", "freq": f_1x})
                excitations.append({"name": f"Motor 2X Harmonik ({rpm:.0f} RPM)", "freq": f_2x})

        if discrete_excitation_frequencies_hz:
            for f_disc in discrete_excitation_frequencies_hz:
                excitations.append({"name": f"Harici Titreşim Kaynağı ({f_disc:.1f} Hz)", "freq": f_disc})

        # The margin is measured against the excitations this machine really
        # sees. Filling in a stock 50 Hz / 1800 RPM spectrum would produce a
        # pass or fail verdict about a machine nobody described.
        if not excitations:
            raise ValueError(
                "Rezonans denetimi için en az bir uyarma kaynağı gereklidir: "
                "operating_rpms ve/veya excitation_frequencies_hz verin."
            )

        checks: List[ResonanceCheckItem] = []
        min_margin = float("inf")
        critical_count = 0

        for mode in modes:
            fn = mode.frequency_hz
            for exc in excitations:
                f_exc = exc["freq"]
                if f_exc <= 0:
                    continue

                margin_pct = (abs(fn - f_exc) / f_exc) * 100.0
                min_margin = min(min_margin, margin_pct)

                if margin_pct < min_safe_margin_pct:
                    risk = "CRITICAL_RESONANCE"
                    critical_count += 1
                    diag = f"REZONE TEHLİKESİ: Mode {mode.mode_number} ({fn} Hz) ile {exc['name']} frekansı çakışmaktadır (Fark: %{margin_pct:.1f} < %{min_safe_margin_pct})."
                elif margin_pct < min_safe_margin_pct * 1.5:
                    risk = "WARNING"
                    diag = f"Düşük Güvenlik Marjı: Mode {mode.mode_number} ({fn} Hz) ile {exc['name']} yakınında (%{margin_pct:.1f} fark)."
                else:
                    risk = "SAFE"
                    diag = f"Güvenli Frekans Ayrımı (Fark: %{margin_pct:.1f} >= %{min_safe_margin_pct})."

                checks.append(ResonanceCheckItem(
                    mode_number=mode.mode_number,
                    natural_frequency_hz=fn,
                    excitation_source=exc["name"],
                    excitation_frequency_hz=round(f_exc, 2),
                    separation_margin_pct=round(margin_pct, 2),
                    risk_level=risk,
                    diagnosis=diag
                ))

        if critical_count > 0:
            overall_risk = "HIGH"
            rec = "KRİTİK UYARI: Rezonans riski tespit edildi! Parça rijitliğini artırarak (kalınlık artırma / feder ekleme) doğal frekansı çalışma bandının üzerine çekiniz."
        elif min_margin < min_safe_margin_pct * 1.5:
            overall_risk = "MODERATE"
            rec = "Frekans ayrımı kabul edilebilir sınıra yakın. Titreşim sönümleyici veya lokal takviye önerilir."
        else:
            overall_risk = "LOW"
            rec = "Tüm doğal frekanslar çalışma frekans bandından güvenli şekilde uzaktadır (Rezonans riski bulunmamaktadır)."

        return ResonanceAssessmentReport(
            overall_resonance_risk=overall_risk,
            minimum_separation_margin_pct=round(min_margin, 2),
            safety_threshold_pct=min_safe_margin_pct,
            critical_modes_count=critical_count,
            detailed_checks=checks,
            recommendation=rec
        )
