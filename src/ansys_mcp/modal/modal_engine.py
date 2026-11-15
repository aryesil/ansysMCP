"""
Modal Result Summariser
-----------------------
Turns the output of a real ANSYS modal solve into a structured summary.

This module does not compute eigenvalues. Natural frequencies come from the
solver, and effective mass ratios come from the participation-factor table
MAPDL writes into ``solve.out``. Mode classification is derived from those
ratios rather than assumed from mode ordering: which direction a mode actually
moves mass in is a property of the solved eigenvector, not of its index.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModeShapeResult(BaseModel):
    mode_number: int
    frequency_hz: float
    period_sec: float
    mode_type: str  # Derived from the dominant effective-mass direction
    effective_mass_ratio_x: float
    effective_mass_ratio_y: float
    effective_mass_ratio_z: float
    effective_mass_ratio_rotx: float = 0.0
    effective_mass_ratio_roty: float = 0.0
    effective_mass_ratio_rotz: float = 0.0
    max_relative_amplitude: float = 1.0


class ModalStudySummary(BaseModel):
    num_modes_extracted: int
    fundamental_frequency_hz: float
    modes: List[ModeShapeResult]
    cumulative_mass_participation: Dict[str, float]
    dynamic_stiffness_assessment: str
    source: str = "ANSYS 2025 R1 modal çözümü"


class ModalEngineError(RuntimeError):
    """Raised when there is no real modal result to summarise."""


#: Axis of the structure a dominant effective-mass direction implies. The
#: rotational entries are about the corresponding axis, so a mode dominated by
#: ROTX is a torsion mode about X.
_MODE_LABELS = {
    "x": "X Ekseninde Eğilme / Eksenel",
    "y": "Y Ekseninde Eğilme",
    "z": "Z Ekseninde Eğilme",
    "rotx": "X Ekseni Etrafında Burulma",
    "roty": "Y Ekseni Etrafında Burulma",
    "rotz": "Z Ekseni Etrafında Burulma",
}

#: Below this effective-mass ratio no direction is dominant enough to name.
_DOMINANCE_FLOOR = 0.01


class ModalEngine:
    """Summarises the modes a real modal solve produced."""

    @classmethod
    def summarize_solved_modes(
        cls,
        frequencies: List[Dict[str, Any]],
        participation: Optional[Dict[str, Any]] = None,
    ) -> ModalStudySummary:
        """
        Build the study summary from solver output.

        Args:
            frequencies: rows of ``{"mode", "frequency_hz", "max_amplitude_m"}``
                read off the solved mode-shape results.
            participation: the parsed ``solve.out`` participation table, or
                ``None`` when the solver did not emit one.
        """
        if not frequencies:
            raise ModalEngineError(
                "Özetlenecek modal sonuç yok. Modal analiz gerçekten çözüldü mü?"
            )

        by_mode: Dict[int, Dict[str, Any]] = {}
        if participation:
            for row in participation.get("modes", []):
                by_mode[int(row["mode"])] = row

        peak_amplitude = max(
            (abs(f.get("max_amplitude_m", 0.0)) for f in frequencies), default=0.0
        ) or 1.0

        modes: List[ModeShapeResult] = []
        for row in sorted(frequencies, key=lambda r: int(r["mode"])):
            mode_number = int(row["mode"])
            freq = float(row["frequency_hz"])
            pf = by_mode.get(mode_number, {})

            ratios = {
                key: float(pf.get("effective_mass_ratio_" + key, 0.0))
                for key in _MODE_LABELS
            }
            dominant = max(ratios, key=lambda k: ratios[k])
            if ratios[dominant] < _DOMINANCE_FLOOR:
                mode_type = "Baskın yönü olmayan lokal mod"
            else:
                mode_type = "%s (etkin kütle %%%.1f)" % (
                    _MODE_LABELS[dominant],
                    ratios[dominant] * 100.0,
                )

            modes.append(
                ModeShapeResult(
                    mode_number=mode_number,
                    frequency_hz=round(freq, 4),
                    period_sec=round(1.0 / freq, 6) if freq > 0 else 0.0,
                    mode_type=mode_type,
                    effective_mass_ratio_x=round(ratios["x"], 6),
                    effective_mass_ratio_y=round(ratios["y"], 6),
                    effective_mass_ratio_z=round(ratios["z"], 6),
                    effective_mass_ratio_rotx=round(ratios["rotx"], 6),
                    effective_mass_ratio_roty=round(ratios["roty"], 6),
                    effective_mass_ratio_rotz=round(ratios["rotz"], 6),
                    max_relative_amplitude=round(
                        abs(row.get("max_amplitude_m", 0.0)) / peak_amplitude, 4
                    ),
                )
            )

        cumulative = {}
        if participation:
            cumulative = {
                k: round(v, 6)
                for k, v in participation.get("cumulative_mass_fraction", {}).items()
            }

        fundamental = modes[0].frequency_hz
        if cumulative:
            covered = ", ".join(
                "%s: %%%.1f" % (k.upper(), v * 100.0) for k, v in sorted(cumulative.items())
            )
            assessment = (
                "Temel doğal frekans (Mod 1): %.2f Hz. "
                "Çıkarılan %d modun kümülatif etkin kütle oranı — %s."
                % (fundamental, len(modes), covered)
            )
        else:
            assessment = (
                "Temel doğal frekans (Mod 1): %.2f Hz. "
                "Katılım faktörü tablosu okunamadığı için etkin kütle kapsamı raporlanamıyor."
                % fundamental
            )

        return ModalStudySummary(
            num_modes_extracted=len(modes),
            fundamental_frequency_hz=fundamental,
            modes=modes,
            cumulative_mass_participation=cumulative,
            dynamic_stiffness_assessment=assessment,
        )
