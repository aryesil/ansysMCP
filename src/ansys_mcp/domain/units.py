"""
Deterministic Unit & Quantity Engine
------------------------------------
Handles unit validation, conversion to standard SI units, and prevents
any unit mismatches across the engineering simulation pipeline.
"""

from typing import Dict, Tuple
from .models import Quantity


# Standard SI Base Unit mappings
UNIT_CONVERSION_FACTORS: Dict[str, Tuple[str, float]] = {
    # Length -> meters (m)
    "m": ("m", 1.0),
    "meter": ("m", 1.0),
    "meters": ("m", 1.0),
    "mm": ("m", 1e-3),
    "millimeter": ("m", 1e-3),
    "cm": ("m", 1e-2),
    "in": ("m", 0.0254),
    "inch": ("m", 0.0254),
    "ft": ("m", 0.3048),
    "foot": ("m", 0.3048),

    # Force -> Newtons (N)
    "N": ("N", 1.0),
    "newton": ("N", 1.0),
    "kN": ("N", 1e3),
    "kilonewton": ("N", 1e3),
    "MN": ("N", 1e6),
    "lbf": ("N", 4.4482216152605),
    "kgf": ("N", 9.80665),

    # Pressure & Stress -> Pascals (Pa)
    "Pa": ("Pa", 1.0),
    "pascal": ("Pa", 1.0),
    "kPa": ("Pa", 1e3),
    "MPa": ("Pa", 1e6),
    "megapascal": ("Pa", 1e6),
    "GPa": ("Pa", 1e9),
    "gigapascal": ("Pa", 1e9),
    "bar": ("Pa", 1e5),
    "psi": ("Pa", 6894.757293168),
    "ksi": ("Pa", 6894757.293168),

    # Velocity -> m/s
    "m/s": ("m/s", 1.0),
    "km/h": ("m/s", 1.0 / 3.6),
    "ft/s": ("m/s", 0.3048),
    "knot": ("m/s", 0.514444),

    # Density -> kg/m^3
    "kg/m3": ("kg/m3", 1.0),
    "kg/m^3": ("kg/m3", 1.0),
    "g/cm3": ("kg/m3", 1000.0),
    "g/cm^3": ("kg/m3", 1000.0),

    # Angle -> radians (rad)
    "rad": ("rad", 1.0),
    "deg": ("rad", 0.017453292519943295),
    "degree": ("rad", 0.017453292519943295),

    # Power -> Watts (W)
    "W": ("W", 1.0),
    "watt": ("W", 1.0),
    "mW": ("W", 1e-3),
    "kW": ("W", 1e3),
    "MW": ("W", 1e6),

    # Heat flux -> W/m^2
    "W/m2": ("W/m2", 1.0),
    "W/m^2": ("W/m2", 1.0),
    "kW/m2": ("W/m2", 1e3),
    "kW/m^2": ("W/m2", 1e3),

    # Convection film coefficient -> W/(m^2 K)
    "W/m2K": ("W/m2K", 1.0),
    "W/m^2K": ("W/m2K", 1.0),
    "W/m^2/K": ("W/m2K", 1.0),
    "W/(m^2*K)": ("W/m2K", 1.0),
    "W/m2/K": ("W/m2K", 1.0),

    # Thermal conductivity -> W/(m K)
    "W/mK": ("W/mK", 1.0),
    "W/m/K": ("W/mK", 1.0),
    "W/m-K": ("W/mK", 1.0),
    "W/(m*K)": ("W/mK", 1.0),

    # Volumetric heat generation -> W/m^3
    "W/m3": ("W/m3", 1.0),
    "W/m^3": ("W/m3", 1.0),
    "MW/m3": ("W/m3", 1e6),
    "MW/m^3": ("W/m3", 1e6),

    # Mass -> kilograms (kg)
    "kg": ("kg", 1.0),
    "g": ("kg", 1e-3),
    "t": ("kg", 1e3),
    "lb": ("kg", 0.45359237),

    # Moment / torque -> N*m
    "N*m": ("N*m", 1.0),
    "Nm": ("N*m", 1.0),
    "N.m": ("N*m", 1.0),
    "kN*m": ("N*m", 1e3),
    "N*mm": ("N*m", 1e-3),
    "Nmm": ("N*m", 1e-3),

    # Acceleration -> m/s^2
    "m/s^2": ("m/s^2", 1.0),
    "m/s2": ("m/s^2", 1.0),
    "mm/s^2": ("m/s^2", 1e-3),

    # Area -> m^2
    "m2": ("m2", 1.0),
    "m^2": ("m2", 1.0),
    "mm2": ("m2", 1e-6),
    "mm^2": ("m2", 1e-6),
    "cm2": ("m2", 1e-4),
    "cm^2": ("m2", 1e-4),

    # Frequency -> Hz
    "Hz": ("Hz", 1.0),
    "kHz": ("Hz", 1e3),
    "rpm": ("Hz", 1.0 / 60.0),
    "RPM": ("Hz", 1.0 / 60.0),

    # Specific heat -> J/(kg K)
    "J/kgK": ("J/kgK", 1.0),
    "J/kg/K": ("J/kgK", 1.0),
    "J/kg-K": ("J/kgK", 1.0),
    "J/(kg*K)": ("J/kgK", 1.0),

    # Thermal expansion coefficient -> 1/K
    "1/K": ("1/K", 1.0),
    "K^-1": ("1/K", 1.0),
    "1/C": ("1/K", 1.0),
}

#: Temperature is affine, not a pure scaling, so it cannot live in the factor
#: table above. Each entry maps a unit onto kelvin as (scale, offset):
#: ``kelvin = value * scale + offset``.
TEMPERATURE_UNITS = {
    "K": (1.0, 0.0),
    "kelvin": (1.0, 0.0),
    "C": (1.0, 273.15),
    "degC": (1.0, 273.15),
    "°C": (1.0, 273.15),
    "celsius": (1.0, 273.15),
    "F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    "degF": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    "°F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
}


class UnitError(ValueError):
    """Raised when an unrecognized or mismatched unit is supplied."""
    pass


class UnitEngine:
    """Provides deterministic unit conversions and validations."""

    @staticmethod
    def normalize_unit(unit_str: str) -> str:
        return unit_str.strip()

    @staticmethod
    def is_temperature(unit_str: str) -> bool:
        return unit_str.strip() in TEMPERATURE_UNITS

    @classmethod
    def to_si(cls, quantity: Quantity) -> Quantity:
        """Converts any recognized quantity into its base SI equivalent."""
        u = cls.normalize_unit(quantity.unit)

        if u in TEMPERATURE_UNITS:
            scale, offset = TEMPERATURE_UNITS[u]
            return Quantity(value=quantity.value * scale + offset, unit="K")

        if u not in UNIT_CONVERSION_FACTORS:
            raise UnitError(f"Tanımlanamayan birim: '{quantity.unit}'. Lütfen standart SI birimi kullanın.")

        base_unit, factor = UNIT_CONVERSION_FACTORS[u]
        return Quantity(value=quantity.value * factor, unit=base_unit)

    @classmethod
    def convert(cls, quantity: Quantity, target_unit: str) -> Quantity:
        """Converts a quantity from its current unit to target_unit within the same physical dimension."""
        u_src = cls.normalize_unit(quantity.unit)
        u_tgt = cls.normalize_unit(target_unit)

        if u_src in TEMPERATURE_UNITS or u_tgt in TEMPERATURE_UNITS:
            if not (u_src in TEMPERATURE_UNITS and u_tgt in TEMPERATURE_UNITS):
                raise UnitError(
                    f"Boyutsal uyumsuzluk: '{quantity.unit}' birimi '{target_unit}' birimine dönüştürülemez."
                )
            kelvin = cls.to_si(quantity).value
            scale, offset = TEMPERATURE_UNITS[u_tgt]
            return Quantity(value=(kelvin - offset) / scale, unit=target_unit)

        if u_src not in UNIT_CONVERSION_FACTORS:
            raise UnitError(f"Kaynak birim bilinmiyor: '{quantity.unit}'")
        if u_tgt not in UNIT_CONVERSION_FACTORS:
            raise UnitError(f"Hedef birim bilinmiyor: '{target_unit}'")

        base_src, factor_src = UNIT_CONVERSION_FACTORS[u_src]
        base_tgt, factor_tgt = UNIT_CONVERSION_FACTORS[u_tgt]

        if base_src != base_tgt:
            raise UnitError(
                f"Boyutsal uyumsuzluk: '{quantity.unit}' ({base_src}) birimi '{target_unit}' ({base_tgt}) birimine dönüştürülemez."
            )

        si_val = quantity.value * factor_src
        target_val = si_val / factor_tgt
        return Quantity(value=target_val, unit=target_unit)

    @classmethod
    def format_quantity(cls, quantity: Quantity, precision: int = 2) -> str:
        """Standardized string representation."""
        return f"{quantity.value:.{precision}f} {quantity.unit}"
