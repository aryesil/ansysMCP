"""
Grid Convergence Index (GCI) & Richardson Extrapolation Engine
--------------------------------------------------------------
Implements ASME V&V 20-2009 / Roache Standard for numerical mesh independence.
Evaluates 3-grid refinement hierarchies (Coarse, Medium, Fine) deterministically.
"""

import math
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class GridLevelResult(BaseModel):
    level_name: str  # e.g. "Coarse (h3)", "Medium (h2)", "Fine (h1)"
    element_size_mm: float
    element_count: int
    result_value: float  # e.g. Max Von Mises Stress (MPa) or Drag Coefficient (Cd)


class GciReport(BaseModel):
    metric_name: str
    grid_refinement_ratio_r: float
    order_of_convergence_p: float
    extrapolated_value: float
    apparent_order: float
    gci_fine_pct: float
    gci_medium_pct: float
    asymptotic_ratio: float
    is_converged: bool
    is_in_asymptotic_range: bool
    #: False when the three grids do not move the metric in one direction. The
    #: observed order cannot be measured in that case, so p is assumed and the
    #: GCI must be read as an indication rather than an error bound.
    is_monotonic: bool = True
    status: str
    levels: List[GridLevelResult]
    recommendation: str


class GciEngine:
    """ASME V&V standard Grid Convergence Index calculator."""

    @classmethod
    def calculate_gci(
        cls,
        metric_name: str,
        h1_fine: GridLevelResult,
        h2_medium: GridLevelResult,
        h3_coarse: GridLevelResult,
        target_gci_pct: float = 3.0,
        safety_factor_Fs: float = 1.25
    ) -> GciReport:
        """
        Calculates GCI across 3 mesh levels (h1 < h2 < h3).
        r21 = h2 / h1
        r32 = h3 / h2
        """
        r21 = h2_medium.element_size_mm / h1_fine.element_size_mm
        r32 = h3_coarse.element_size_mm / h2_medium.element_size_mm

        if r21 <= 1.0 or r32 <= 1.0:
            raise ValueError(f"Mesh boyutları h1 < h2 < h3 sıralamasında olmalıdır. Verilen: {h1_fine.element_size_mm}, {h2_medium.element_size_mm}, {h3_coarse.element_size_mm}")

        r = round((r21 + r32) / 2.0, 3)

        f1 = h1_fine.result_value
        f2 = h2_medium.result_value
        f3 = h3_coarse.result_value

        e21 = f2 - f1
        e32 = f3 - f2

        # Check monotonicity
        is_monotonic = True
        if e21 == 0 or e32 == 0:
            p = 2.0
            is_monotonic = False
        else:
            ratio = e32 / e21
            if ratio <= 0:
                # Oscillatory convergence or singularity: the observed order
                # cannot be measured from these three grids, so first order is
                # assumed and the report says so.
                p = 1.0
                is_monotonic = False
            else:
                p = abs(math.log(abs(ratio)) / math.log(r))
                p = max(min(p, 4.0), 0.5)  # Constrain order to realistic 0.5 - 4.0 range

        # Richardson Extrapolation
        rp = r ** p
        if rp == 1.0:
            rp = 1.001
        f_extrap = f1 + (f1 - f2) / (rp - 1.0)

        # Fine Grid GCI (%)
        rel_diff_21 = abs((f1 - f2) / f1) if f1 != 0 else 0.0
        gci_fine = (safety_factor_Fs * rel_diff_21) / (rp - 1.0) * 100.0

        # Medium Grid GCI (%)
        rel_diff_32 = abs((f2 - f3) / f2) if f2 != 0 else 0.0
        gci_medium = (safety_factor_Fs * rel_diff_32) / (rp - 1.0) * 100.0

        # Check asymptotic range: GCI_med / (r^p * GCI_fine) ~ 1.0
        denom = rp * gci_fine if gci_fine > 0 else 1.0
        asymptotic_check = gci_medium / denom if denom > 0 else 1.0
        in_asymptotic_range = 0.90 <= asymptotic_check <= 1.10

        is_converged = gci_fine <= target_gci_pct and is_monotonic

        if is_converged:
            status = "CONVERGED (Mesh Bağımsızlığı Sağlandı)"
            recommendation = f"Fine mesh ({h1_fine.element_size_mm} mm) güvenilir sonuç vermektedir. Hata payı GCI: %{gci_fine:.2f}."
        elif gci_fine <= target_gci_pct * 2.0:
            status = "APPROACHING_CONVERGENCE"
            recommendation = f"Mesh yakınsama eğiliminde ancak GCI (%{gci_fine:.2f}) hedef sınırın (%{target_gci_pct}) üzerindedir. Bir kademe daha sıklaştırma önerilir."
        elif not is_monotonic:
            status = "NOT_CONVERGED_OSCILLATORY"
            recommendation = (
                f"Üç ağ metriği tek yönde değiştirmedi (salınımlı yakınsama), bu yüzden gözlenen "
                f"yakınsama mertebesi ölçülemedi ve p = {p:.1f} varsayıldı. GCI: %{gci_fine:.2f} "
                f"bir hata sınırı değil, gösterge olarak okunmalıdır. Sıklaştırma oranını artırın "
                f"veya kritik bölgede lokal refinement uygulayın."
            )
        else:
            status = "NOT_CONVERGED"
            recommendation = f"Mesh bağımsızlığı sağlanamadı. GCI: %{gci_fine:.2f}. Daha ince mesh veya lokal mesh refinement (Inflation) uygulayınız."

        return GciReport(
            metric_name=metric_name,
            grid_refinement_ratio_r=r,
            order_of_convergence_p=round(p, 3),
            extrapolated_value=round(f_extrap, 3),
            apparent_order=round(p, 2),
            gci_fine_pct=round(gci_fine, 2),
            gci_medium_pct=round(gci_medium, 2),
            asymptotic_ratio=round(asymptotic_check, 3),
            is_converged=is_converged,
            is_in_asymptotic_range=in_asymptotic_range,
            is_monotonic=is_monotonic,
            status=status,
            levels=[h1_fine, h2_medium, h3_coarse],
            recommendation=recommendation
        )
