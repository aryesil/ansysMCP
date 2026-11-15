"""
SolidWorks part/assembly to STEP conversion.

ANSYS Mechanical and Fluent read neutral B-Rep (STEP/IGES/Parasolid) reliably;
a native ``.sldprt`` only opens when the matching SolidWorks CAD reader is
configured.  This module converts through SolidWorks itself (COM automation) so
the exported solid is exactly what SolidWorks holds, with no re-modelling.
It needs SolidWorks installed; without it the call raises and nothing is faked.
"""

import os
from typing import Any, Dict, Optional

from .cad_parser import CadParser

SOLIDWORKS_FORMATS = {".sldprt": 1, ".sldasm": 2}

#: swOpenDocOptions_Silent | swOpenDocOptions_ReadOnly
_OPEN_OPTIONS = 1 | 2


class CadConversionError(RuntimeError):
    pass


def convert_to_step(source_path: str, output_dir: str, output_name: Optional[str] = None) -> Dict[str, Any]:
    """Exports a SolidWorks file as STEP and checks the result by reading it back."""
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in SOLIDWORKS_FORMATS:
        raise CadConversionError(
            "Yalnızca .sldprt ve .sldasm dönüştürülür (verilen: %r). STEP/IGES dosyaları zaten doğrudan okunur." % ext
        )
    if not os.path.isfile(source_path):
        raise CadConversionError("Kaynak dosya bulunamadı: %s" % source_path)

    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise CadConversionError("pywin32 kurulu değil; SolidWorks dönüştürmesi çalıştırılamaz (%s)." % exc)

    os.makedirs(output_dir, exist_ok=True)
    stem = output_name or os.path.splitext(os.path.basename(source_path))[0]
    target = os.path.join(output_dir, stem + ".step")

    pythoncom.CoInitialize()
    try:
        try:
            app = win32com.client.Dispatch("SldWorks.Application")
        except Exception as exc:
            raise CadConversionError("SolidWorks başlatılamadı veya kurulu değil: %s" % exc)
        errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        model = app.OpenDoc6(source_path, SOLIDWORKS_FORMATS[ext], _OPEN_OPTIONS, "", errors, warnings)
        if model is None:
            raise CadConversionError(
                "SolidWorks dosyayı açamadı (hata kodu %s): %s" % (errors.value, source_path)
            )
        try:
            saved = model.Extension.SaveAs(target, 0, 1, None, errors, warnings)
        finally:
            app.CloseDoc(model.GetTitle())
    except CadConversionError:
        raise
    except Exception as exc:
        raise CadConversionError("SolidWorks dönüştürmesi başarısız: %s" % exc)
    finally:
        pythoncom.CoUninitialize()

    if not saved or not os.path.isfile(target) or os.path.getsize(target) == 0:
        raise CadConversionError("SolidWorks STEP dosyasını yazamadı: %s" % target)

    # Read it back: a STEP that does not parse to a solid is not a conversion.
    summary = CadParser.parse_cad_file(target)
    if not summary.total_volume_mm3 or summary.total_volume_mm3 <= 0:
        raise CadConversionError("Dönüştürülen STEP dosyasında katı hacim bulunamadı: %s" % target)
    return {
        "step_path": target,
        "size_kb": round(os.path.getsize(target) / 1024.0, 2),
        "bounding_box_mm": summary.bounding_box,
        "volume_mm3": summary.total_volume_mm3,
        "surface_area_mm2": summary.total_surface_area_mm2,
    }
