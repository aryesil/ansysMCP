"""
ANSYS AI-MCP Server (v2.0 Architecture)
---------------------------------------
Production-grade Model Context Protocol (MCP) server for ANSYS FEA & CFD automation.
Implements asynchronous job lifecycle, strict validation, deterministic engineering,
and semantic geometry selection for Claude Code, Antigravity, Cursor, and Codex.
"""

import functools
import os
import json
import threading
import time
from typing import Dict, Any, List, Optional
try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from .domain.models import (
    JobState,
    AnalysisType,
    Material,
    MeshConfig,
    BoundaryCondition,
    Quantity,
    SemanticSelector,
    AnalysisManifest,
    ResultMetric
)
from .domain.units import UnitEngine
from .domain.geometry import GeometryIntelligence, GeometryEntity
from .domain.engineering import EngineeringCalculator
from .validation.gatekeeper import GatekeeperValidator
from .orchestrator.state_machine import JobStateMachine, InvalidStateTransitionError
from .orchestrator.storage import JobStorage
from .orchestrator.process_manager import ProcessManager
from .adapters.mechanical import MechanicalAdapter
from .adapters.transient_reader import read_displacement_history
from .adapters.mechanical_live import get_live_session, reset_live_session, LiveMechanicalError
from .adapters.fluent import FluentAdapter, TURBULENCE_MODELS
from .adapters.post_processing import PostProcessingAdapter
from .adapters.cfd_post_processing import CfdPostProcessingAdapter

from .geometry.cad_parser import CadParser, CadModelSummary
from .geometry.semantic_matcher import SemanticMatcher
from .geometry.visualizer import GeometryVisualizer
from .geometry.named_selections import NamedSelectionsContainer
from .geometry.contacts import ContactManager, ContactType

from .analysis.mesh_study_runner import MeshStudyRunner
from .analysis.gci_engine import GciEngine
from .analysis.singularity_detector import SingularityDetector

from .cfd.enclosure import WindTunnelEnclosure
from .cfd.inflation import InflationEngine, InflationLayerConfig

from .security.sandbox import SandboxEngine
from .security.provenance import ProvenanceEngine, ProvenanceRecord
from .security.client_installer import ClientConfigGenerator

from .optimization.doe_engine import DoeEngine, DesignParameter, DesignPoint
from .optimization.optimizer import ParetoOptimizer, OptimizationObjective, OptimizationConstraint
from .optimization.batch_runner import OptimizationBatchRunner

from .modal.modal_engine import ModalEngine, ModeShapeResult, ModalStudySummary
from .modal.resonance_evaluator import ResonanceEvaluator, ResonanceAssessmentReport
from .modal.mode_visualizer import ModeShapeVisualizer

from .thermal.thermal_stress import ThermalStressCoupler, ThermalStressReport
from .thermal.thermal_visualizer import ThermalVisualizer

from .config import SystemConfig
from .domain.material_library import lookup as lookup_material_card, list_materials
from .validation.load_specs import (
    STANDARD_GRAVITY_M_S2,
    build_layers,
    build_load,
    build_orthotropic,
    build_ply_strength,
    build_support,
)
from .domain import hand_calcs
from .validation.input_checks import (
    InputError,
    check_material_card,
    check_orthotropic,
    require_finite,
    require_positive,
    require_selection,
    unit_vector,
)

# Initialize MCP Server
mcp = FastMCP(
    name="ANSYS-AI-MCP-Server"
)

# Persistent Job Storage
#: Where job folders and the job database live.  ``AMPC_RUNS_DIR`` moves them (for
#: example to another drive when the default one is full); the default is a ``runs`` folder
#: at the project root, next to ``server.py``.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("AMPC_RUNS_DIR") or os.path.join(PROJECT_ROOT, "runs")
storage = JobStorage(workspace_root=RUNS_DIR)

# ---------------------------------------------------------------------------
# Çözücü oturumu kilidi
# ---------------------------------------------------------------------------
# Mechanical tek bir yerel gRPC oturumuyla sürülür ve her job projeyi sıfırlar.
# İki çözüm aynı anda koşarsa biri diğerinin modelini siler (ya da oturumu düşürür).
# Bu yüzden uzun süren araçlar tek tek çalışır; meşgulken beklemek yerine, istemcinin
# zaman aşımına düşmemesi için açık bir hata döner.

_SOLVER_LOCK = threading.Lock()

#: What the solver lock is currently held for, so another call can report it or stop it.
_ACTIVE_RUN: Dict[str, Any] = {}
_CANCEL_REQUESTED: set = set()


def exclusive_solver(func):
    """Run ``func`` only when no other solver-backed tool is running."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _SOLVER_LOCK.acquire(blocking=False):
            return {
                "status": "busy",
                "message": "Başka bir çözücü işlemi sürüyor (%s, job %s). Bitmesini bekleyin ya da 'cancel_job' kullanın."
                % (_ACTIVE_RUN.get("tool", "?"), _ACTIVE_RUN.get("job_id", "?")),
            }
        job_id = kwargs.get("job_id") or (args[0] if args and isinstance(args[0], str) else None)
        _ACTIVE_RUN.clear()
        _ACTIVE_RUN.update({"tool": func.__name__, "job_id": job_id, "started": time.time(), "close": None})
        _CANCEL_REQUESTED.discard(job_id)
        try:
            return func(*args, **kwargs)
        finally:
            _ACTIVE_RUN.clear()
            _CANCEL_REQUESTED.discard(job_id)
            _SOLVER_LOCK.release()

    return wrapper


def _mark_failed(job_id: str, error: Optional[Exception] = None) -> None:
    """Record the end of a run that did not finish: cancelled if the user asked, else failed."""
    try:
        if job_id in _CANCEL_REQUESTED:
            storage.update_job_state(job_id, JobState.CANCELLED, error_message="Kullanıcı iptal etti.")
        else:
            storage.update_job_state(
                job_id, JobState.FAILED, error_message=str(error) if error is not None else None
            )
    except InvalidStateTransitionError:
        # The job had already finished; its stored result is not overwritten.
        pass


def fresh_run_only(func):
    """Refuse to start a solve on a job whose result is already final."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        job_id = kwargs.get("job_id") or (args[0] if args and isinstance(args[0], str) else None)
        job = storage.get_job(job_id) if job_id else None
        if job and job["state"] in (JobState.COMPLETED.value, JobState.CANCELLED.value):
            return {
                "status": "error",
                "job_id": job_id,
                "state": job["state"],
                "message": "Bu iş '%s' durumunda; sonucu korumak için yeniden çözülemez. "
                "'clone_job' ile kopyalayın ya da 'configure_*' ile yeniden yapılandırın." % job["state"],
            }
        return func(*args, **kwargs)

    return wrapper



# ---------------------------------------------------------------------------
# 0. Sistem ve ANSYS 2025 Ortam Bilgisi
# ---------------------------------------------------------------------------

@mcp.tool()
def get_ansys_environment_info() -> Dict[str, Any]:
    """
    Sistemde kurulu olan ANSYS sürümünü (ANSYS 2025 R1/R2, 2024),
    yol değişkenlerini (AWP_ROOT251) ve çözücü hazır olma durumunu döner.
    """
    return SystemConfig.get_ansys_summary()


# ---------------------------------------------------------------------------
# 1. MCP Tools (Eylemler)
# ---------------------------------------------------------------------------

@mcp.tool()
def open_ansys_session(show_gui: bool = True) -> Dict[str, Any]:
    """
    Gerçek ANSYS 2025 Mechanical uygulamasını açar (veya açık olana bağlanır).

    show_gui=True iken Mechanical penceresi ekranda görünür ve sonraki tüm
    adımlar canlı olarak izlenebilir. İlk açılış yaklaşık 2 dakika sürer.

    Args:
        show_gui: Mechanical arayüzünün ekranda görünüp görünmeyeceği.
    """
    try:
        session = get_live_session(visible=show_gui)
        return {
            "status": "live",
            "ansys": SystemConfig.get_ansys_summary(),
            "session": session.step_log[0] if session.step_log else {},
            "project_directory": session.project_directory,
            "message": "ANSYS Mechanical canlı oturumu hazır.",
        }
    except Exception as exc:
        return {"status": "error", "message": f"Mechanical başlatılamadı: {exc}"}


@mcp.tool()
@exclusive_solver
def inspect_live_geometry_faces(cad_file_path: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    CAD dosyasını canlı Mechanical'a yükler ve GERÇEK yüz topolojisini listeler.

    Dönen her yüz için gerçek id, alan (mm^2), ağırlık merkezi (mm), yüzey tipi
    ve ortalama normal verilir. Sınır şartı seçicilerini bu listeye bakarak kurun.

    Args:
        cad_file_path: Katı CAD dosyası (STEP / IGES / Parasolid / SAT).
        show_gui: Mechanical arayüzünün görünüp görünmeyeceği.
    """
    try:
        session = get_live_session(visible=show_gui)
        session.reset_project()
        session.set_unit_system_mks()
        summary = session.import_geometry(cad_file_path)
        faces = session.list_faces()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return {
        "status": "success",
        "geometry": {k: v for k, v in summary.items()},
        "face_count": len(faces),
        "faces": [
            {
                "face_id": f["id"],
                "area_mm2": round(f["area_mm2"], 3),
                "centroid_mm": [round(v, 3) for v in f["centroid_mm"]],
                "surface_type": f["surface_type"],
                "mean_normal": [round(v, 4) for v in f["mean_normal"]],
                "is_planar": f["is_planar"],
            }
            for f in faces
        ],
        "message": "Yüzler gerçek ANSYS topolojisinden okundu.",
    }


@mcp.tool()
def get_live_session_steps(limit: int = 40) -> Dict[str, Any]:
    """
    Canlı ANSYS oturumunda şu ana kadar yürütülen adımların kaydını döner.

    Args:
        limit: Döndürülecek son adım sayısı.
    """
    try:
        session = get_live_session()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    return {
        "status": "success",
        "total_steps": len(session.step_log),
        "steps": session.step_log[-limit:],
    }


@mcp.tool()
def close_ansys_session(exit_application: bool = False) -> Dict[str, Any]:
    """
    Canlı ANSYS oturumunu bırakır.

    Args:
        exit_application: True ise Mechanical uygulaması kapatılır ve lisans serbest bırakılır.
                          False ise pencere açık kalır, sadece bağlantı bırakılır.
    """
    try:
        session = get_live_session()
        session.close(exit_app=exit_application)
        reset_live_session()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
    return {
        "status": "closed",
        "application_exited": exit_application,
        "message": "Mechanical kapatıldı." if exit_application else "Bağlantı bırakıldı, pencere açık.",
    }


@mcp.tool()
def create_analysis(analysis_type: str = "static_structural", name: str = "FEA_Assessment") -> Dict[str, Any]:
    """
    Yeni bir mühendislik simülasyon işi (Job) başlatır.
    
    Args:
        analysis_type: 'static_structural', 'modal', 'buckling', 'harmonic', 'transient', 'random_vibration', 'cfd_steady'
        name: Analiz için proje başlığı
    """
    try:
        a_type = AnalysisType(analysis_type)
    except ValueError:
        return {"status": "error", "message": f"Desteklenmeyen analiz türü: {analysis_type}"}

    job_id = f"{a_type.value.upper()}-{os.urandom(4).hex()}"
    res = storage.create_job(job_id=job_id, analysis_type=a_type)
    return {
        "status": "success",
        "job_id": job_id,
        "state": res["state"],
        "message": f"Analiz oluşturuldu. 'inspect_cad' ve 'configure_static_analysis' ile devam edin."
    }


@mcp.tool()
def convert_cad_to_step(cad_file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    SolidWorks parçasını (.sldprt) veya montajını (.sldasm) SolidWorks'ün kendisiyle STEP'e çevirir.

    Analiz araçları nötr B-Rep (STEP) dosyasıyla çalışır; bu araç, elinizdeki yerel SolidWorks
    dosyasını modeli yeniden çizmeden aynı katıya dönüştürür. SolidWorks bilgisayarda kurulu
    olmalıdır (otomatik başlatılır, birkaç dakika sürebilir). Çıktı geri okunup katı hacmi
    doğrulanır. Dönüştürme başarısızsa hata verilir, sahte dosya üretilmez.

    Args:
        cad_file_path: .sldprt veya .sldasm dosya yolu.
        output_dir: STEP dosyasının yazılacağı klasör. Boşsa runs/_converted altına yazılır.
    """
    from .geometry.cad_convert import CadConversionError, convert_to_step

    target_dir = output_dir or os.path.join(storage.workspace_root, "_converted")
    try:
        info = convert_to_step(cad_file_path, target_dir)
    except CadConversionError as exc:
        return {"status": "error", "message": str(exc)}
    info["status"] = "success"
    info["message"] = "STEP üretildi. Analiz araçlarında 'step_path' değerini cad_file_path olarak kullanın."
    return info


@mcp.tool()
def inspect_cad(cad_file_path: str) -> Dict[str, Any]:
    """
    CAD model dosyasını (STEP, IGES, Parasolid) inceler, parmak izi (SHA-256) çıkarır.
    
    Args:
        cad_file_path: 3D CAD dosyasının yolu.
    """
    if not os.path.exists(cad_file_path):
        return {"status": "error", "message": f"CAD dosyası bulunamadı: {cad_file_path}"}

    sha256 = GeometryIntelligence.compute_file_sha256(cad_file_path)
    file_size_kb = round(os.path.getsize(cad_file_path) / 1024, 2)
    ext = os.path.splitext(cad_file_path)[1].lower()

    return {
        "status": "success",
        "cad_file_path": cad_file_path,
        "format": ext,
        "size_kb": file_size_kb,
        "sha256": sha256,
        "info": "Geometri incelendi. Sınır şartları ve semantik yüzey seçimi için hazır."
    }


@mcp.tool()
def inspect_geometry_details(
    cad_file_path: str,
    mode: str = "summary",
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    CAD modelinin geometrik özetini (hacim, alan, ağırlık merkezi, sınırlayıcı kutu)
    ve isteğe bağlı olarak yüzey listesini döner.

    Tam yüzey listesi yüzlerce kayıt içerebilir ve istemcinin çıktı sınırını aşar;
    bu yüzden varsayılan 'summary' modu yalnızca özeti verir. Yüzeyleri sayfa sayfa
    almak için mode='faces' ile 'offset' ve 'limit' kullanın.

    Args:
        cad_file_path: CAD dosyasının tam yolu.
        mode: 'summary' (yalnızca özet) veya 'faces' (özet + sayfalanmış yüzeyler).
        offset: 'faces' modunda atlanacak yüzey sayısı.
        limit: 'faces' modunda döndürülecek en fazla yüzey (1-200).
    """
    if mode not in ("summary", "faces"):
        return {"status": "error", "message": "mode 'summary' veya 'faces' olmalıdır, verilen: %r" % mode}
    if offset < 0 or not (1 <= limit <= 200):
        return {"status": "error", "message": "offset >= 0 ve 1 <= limit <= 200 olmalıdır."}
    try:
        summary = CadParser.parse_cad_file(cad_file_path)
    except Exception as e:
        return {"status": "error", "message": f"Geometri incelenirken hata: {str(e)}"}

    data = summary.model_dump()
    all_faces = data.pop("faces", [])
    data["faces_total"] = len(all_faces)
    result: Dict[str, Any] = {"status": "success", "summary": data}
    if mode == "faces":
        page = all_faces[offset:offset + limit]
        result["faces"] = page
        result["page"] = {
            "offset": offset,
            "limit": limit,
            "returned": len(page),
            "next_offset": offset + limit if offset + limit < len(all_faces) else None,
        }
    else:
        result["hint"] = "Yüzeyleri görmek için mode='faces' kullanın."
    return result


@mcp.tool()
def find_semantic_faces(
    cad_file_path: str,
    normal_vector: Optional[List[float]] = None,
    normal_tolerance_deg: float = 5.0,
    min_area_mm2: Optional[float] = None,
    max_area_mm2: Optional[float] = None,
    position_axis: Optional[str] = None,
    position_min: Optional[float] = None,
    position_max: Optional[float] = None,
    confidence_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Geometrik kriterlere (normal yönü, alan aralığı, uzamsal konum) göre CAD modeli
    üzerindeki eşleşen yüzeyleri (Faces) ve güven skorlarını (Confidence) tespit eder.
    
    Args:
        cad_file_path: CAD dosyasının yolu.
        normal_vector: Aranan yüzey normali [nx, ny, nz].
        normal_tolerance_deg: Açısal tolerans (derece).
        min_area_mm2: Minimum yüzey alanı (mm²).
        max_area_mm2: Maksimum yüzey alanı (mm²).
        position_axis: Konum filtresi ekseni ('x', 'y', 'z').
        position_min: Eksen minimum koordinatı.
        position_max: Eksen maksimum koordinatı.
        confidence_threshold: Kabul edilebilir minimum güven skoru (0.0 - 1.0).
    """
    try:
        summary = CadParser.parse_cad_file(cad_file_path)
        pos_filter = None
        if position_axis and (position_min is not None or position_max is not None):
            pos_filter = {
                "axis": position_axis,
                "min": position_min if position_min is not None else float("-inf"),
                "max": position_max if position_max is not None else float("inf")
            }

        area_range = [min_area_mm2, max_area_mm2] if (min_area_mm2 is not None and max_area_mm2 is not None) else None

        selector = SemanticSelector(
            name="Query_Selector",
            normal_vector=normal_vector,
            normal_tolerance_deg=normal_tolerance_deg,
            area_range_mm2=area_range,
            position_filter=pos_filter,
            confidence_threshold=confidence_threshold
        )

        matches = SemanticMatcher.match_selector(selector, summary)

        return {
            "status": "success",
            "matches_found": len(matches),
            "candidates": [m.model_dump() for m in matches],
            "best_match": matches[0].face_id if matches else None,
            "has_ambiguity": any(m.is_ambiguous for m in matches)
        }
    except Exception as e:
        return {"status": "error", "message": f"Yüzey araması sırasında hata: {str(e)}"}


@mcp.tool()
def render_geometry_preview(cad_file_path: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    3D CAD modelinin izometrik ön izleme görselini (PNG) üretir.
    
    Args:
        cad_file_path: CAD dosyasının yolu.
        job_id: İsteğe bağlı job kimliği.
    """
    try:
        out_dir = storage.get_job_dir(job_id) if job_id else RUNS_DIR + "/previews"
        out_png = os.path.join(out_dir, "geometry_preview.png")
        rendered_path = GeometryVisualizer.render_cad_preview(cad_file_path, out_png)
        return {
            "status": "success",
            "preview_image_path": rendered_path
        }
    except Exception as e:
        return {"status": "error", "message": f"Render hatası: {str(e)}"}


def _extreme_filter(spec: Optional[str]) -> Optional[Dict[str, Any]]:
    """Turn a selector like 'min_x' into a position filter for face matching."""
    if not spec:
        return None
    token = spec.strip().lower().replace("-", "_")
    if "_" not in token:
        return None
    extreme, axis = token.split("_", 1)
    if extreme not in ("min", "max") or axis not in ("x", "y", "z"):
        return None
    return {"axis": axis, "extreme": extreme}


def _configure_structural(
    expected_type: str,
    job_id: str,
    cad_file_path: str,
    material_name: str,
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    ultimate_strength_mpa: Optional[float] = None,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    force_magnitude_n: Optional[float] = None,
    force_direction: Optional[List[float]] = None,
    force_face_normal: Optional[List[float]] = None,
    force_face_axis_extreme: Optional[str] = None,
    pressure_mpa: Optional[float] = None,
    pressure_face_normal: Optional[List[float]] = None,
    pressure_face_axis_extreme: Optional[str] = None,
    extra_supports: Optional[List[Dict[str, Any]]] = None,
    extra_loads: Optional[List[Dict[str, Any]]] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    num_modes: Optional[int] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    require_load: bool = True,
    layers: Optional[List[Dict[str, Any]]] = None,
    ply_strengths_mpa: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Shared setup for the structural job types; see the public tools for the argument list."""
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if job["analysis_type"] != expected_type:
        return {
            "status": "error",
            "message": "Bu job '%s' türünde; bu araç yalnızca '%s' içindir."
            % (job["analysis_type"], expected_type),
        }

    notes: List[str] = []
    try:
        # ---- material: explicit properties, or a library card by name ----
        ortho_model = None
        if orthotropic is not None:
            ortho_model = build_orthotropic(orthotropic)
            if youngs_modulus_gpa is None:
                youngs_modulus_gpa = orthotropic["ex_gpa"]
            if poissons_ratio is None:
                poissons_ratio = orthotropic["nu_xy"]
        explicit = [youngs_modulus_gpa, poissons_ratio, yield_strength_mpa, density_kg_m3]
        card = lookup_material_card(material_name)
        if all(v is None for v in explicit):
            if card is None:
                return {
                    "status": "error",
                    "message": "'%s' kütüphanede yok. E, poisson, akma ve yoğunluğu açıkça verin "
                    "(kütüphane listesi için 'lookup_material' çağırın)." % material_name,
                }
            youngs_modulus_gpa = card["youngs_modulus_gpa"]
            poissons_ratio = card["poissons_ratio"]
            yield_strength_mpa = card["yield_strength_mpa"]
            density_kg_m3 = card["density_kg_m3"]
            if ultimate_strength_mpa is None:
                ultimate_strength_mpa = card.get("ultimate_strength_mpa")
            notes.append("Malzeme kartı kütüphaneden alındı (%s). %s" % (card["name"], card["source"]))
            if card["fidelity"] != "isotropic":
                notes.append("UYARI: " + card["fidelity"])
            if card.get("orthotropic") and ortho_model is None:
                ortho_model = build_orthotropic(card["orthotropic"])
        elif any(v is None for v in explicit):
            missing = [n for n, v in zip(
                ("youngs_modulus_gpa", "poissons_ratio", "yield_strength_mpa", "density_kg_m3"), explicit
            ) if v is None]
            return {
                "status": "error",
                "message": "Malzeme özellikleri kısmen verildi; eksik: %s. Hepsini verin ya da hiçbirini "
                "vermeyip kütüphane kartını kullanın." % ", ".join(missing),
            }

        youngs_modulus_gpa = require_positive("youngs_modulus_gpa", youngs_modulus_gpa)
        poissons_ratio = require_finite("poissons_ratio", poissons_ratio)
        yield_strength_mpa = require_positive("yield_strength_mpa", yield_strength_mpa)
        density_kg_m3 = require_positive("density_kg_m3", density_kg_m3)
        mesh_global_size_mm = require_positive("mesh_global_size_mm", mesh_global_size_mm)
        if ultimate_strength_mpa is None:
            notes.append("Kopma mukavemeti verilmedi; akma değeri (%g MPa) kullanıldı." % yield_strength_mpa)
            ultimate_strength_mpa = yield_strength_mpa
        else:
            ultimate_strength_mpa = require_positive("ultimate_strength_mpa", ultimate_strength_mpa)

        ply_strength = None
        if ply_strengths_mpa is not None:
            if layers is None:
                raise InputError("'ply_strengths_mpa' yalnızca katmanlı kesitle (layers) birlikte anlamlıdır.")
            ply_strength = build_ply_strength(ply_strengths_mpa)

        material = Material(
            name=material_name,
            youngs_modulus=Quantity(value=youngs_modulus_gpa, unit="GPa"),
            poissons_ratio=poissons_ratio,
            yield_strength=Quantity(value=yield_strength_mpa, unit="MPa"),
            ultimate_strength=Quantity(value=ultimate_strength_mpa, unit="MPa"),
            density=Quantity(value=density_kg_m3, unit="kg/m^3"),
            orthotropic=ortho_model,
            ply_strength=ply_strength,
        )
        problems = check_material_card(
            material.youngs_modulus, material.poissons_ratio, material.yield_strength,
            material.ultimate_strength, material.density, orthotropic=ortho_model is not None,
        )
        if ortho_model is not None:
            problems += check_orthotropic(ortho_model)
            notes.append(
                "Ortotropik malzeme: lif doğrultusu global X kabul edilir. Von-Mises güvenlik faktörü "
                "bu malzeme için geçerli değildir ve sonuçta raporlanmaz."
            )
        if problems:
            return {"status": "error", "message": "Malzeme kartı geçersiz: " + " | ".join(problems)}

        # ---- layered section: every ply is stated by the caller ----
        laminate = None
        if layers is not None:
            if ortho_model is None:
                raise InputError(
                    "Katmanlı kesit için ply malzemesi ortotropik olmalıdır: 'orthotropic' sabitlerini verin "
                    "(ex = lif doğrultusu) ya da ortotropik bir kütüphane kartı kullanın."
                )
            layers = build_layers(layers)
            from .composites.clt import laminate_summary
            laminate = laminate_summary(
                layers, ortho_model.ex.value, ortho_model.ey.value, ortho_model.gxy.value, ortho_model.nu_xy,
            )
            notes.append(
                "Katmanlı kesit: %d ply, toplam %.4g mm; açılar yüzeye izdüşürülen global X ekseninden ölçülür "
                "(0 derece = lif bu eksen boyunca). Laminat etkin Ex = %.4g GPa." % (
                    len(layers), laminate["total_thickness_mm"], laminate["ex_effective_gpa"])
            )

        # ---- supports and loads: every face is named by the caller ----
        # The primary fixed support is optional only when 'extra_supports' holds
        # the constraints (e.g. displacement supports that fix all three axes).
        has_primary_support = bool(fixed_support_normal) or bool(fixed_support_axis_extreme)
        if has_primary_support or not extra_supports:
            support_normal, support_filter = require_selection(
                "fixed_support", fixed_support_normal, fixed_support_axis_extreme
            )
        else:
            support_normal = support_filter = None

        prescribes_motion = any(
            isinstance(s, dict) and str(s.get("type", "")).lower() == "displacement"
            for s in (extra_supports or [])
        )
        if require_load and force_magnitude_n is None and pressure_mpa is None and not extra_loads and not prescribes_motion:
            return {
                "status": "error",
                "message": "Yük verilmedi: 'force_magnitude_n', 'pressure_mpa' veya 'extra_loads' belirtin.",
            }

        bcs: List[BoundaryCondition] = []
        if has_primary_support or not extra_supports:
            bcs.append(
                BoundaryCondition(
                    name="Ankastre_Mesnet",
                    type="fixed_support",
                    target_selector=SemanticSelector(
                        name="Mesnet_Yuzeyi",
                        normal_vector=support_normal,
                        position_filter=support_filter,
                        confidence_threshold=0.85,
                    ),
                )
            )

        if force_magnitude_n is not None:
            force_n = require_positive("force_magnitude_n", force_magnitude_n)
            direction = unit_vector("force_direction", force_direction)
            load_normal, load_filter = require_selection(
                "force_face", force_face_normal, force_face_axis_extreme
            )
            bcs.append(
                BoundaryCondition(
                    name="Uygulanan_Kuvvet",
                    type="force",
                    target_selector=SemanticSelector(
                        name="Yuk_Yuzeyi",
                        normal_vector=load_normal,
                        position_filter=load_filter,
                        confidence_threshold=0.85,
                    ),
                    magnitude=Quantity(value=force_n, unit="N"),
                    direction=direction,
                )
            )

        if pressure_mpa is not None:
            pressure = require_positive("pressure_mpa", pressure_mpa)
            p_normal, p_filter = require_selection(
                "pressure_face", pressure_face_normal, pressure_face_axis_extreme
            )
            bcs.append(
                BoundaryCondition(
                    name="Uygulanan_Basinc",
                    type="pressure",
                    target_selector=SemanticSelector(
                        name="Basinc_Yuzeyi",
                        normal_vector=p_normal,
                        position_filter=p_filter,
                        confidence_threshold=0.85,
                    ),
                    magnitude=Quantity(value=pressure, unit="MPa"),
                )
            )
        for i, spec in enumerate(extra_supports or []):
            bcs.append(build_support(spec, i))
        for i, spec in enumerate(extra_loads or []):
            bcs.append(build_load(spec, i))
    except (InputError, ValueError) as exc:
        # InputError is our own check; ValueError covers pydantic's model validation.
        return {"status": "error", "message": str(exc)}

    mesh_config = MeshConfig(
        global_size=Quantity(value=mesh_global_size_mm, unit="mm"),
        target_quality=0.20,
    )

    config_data = {
        "material": material.model_dump(),
        "mesh_config": mesh_config.model_dump(),
        "boundary_conditions": [bc.model_dump() for bc in bcs],
    }
    if num_modes is not None:
        config_data["num_modes"] = int(num_modes)
    if layers is not None:
        config_data["layers"] = layers
        config_data["laminate"] = laminate
    if extra_config:
        config_data.update(extra_config)

    storage.save_job_config(job_id=job_id, cad_path=cad_file_path, config=config_data)

    return {
        "status": "configured",
        "job_id": job_id,
        "state": JobState.CONFIGURED.value,
        "material": material.model_dump(),
        "boundary_conditions": [bc.model_dump() for bc in bcs],
        "notes": notes,
        "laminate": laminate,
        "message": "Konfigürasyon kaydedildi. Lütfen 'validate_analysis' ile doğrulamayı tetikleyin.",
    }


@mcp.tool()
def configure_static_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str,
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    ultimate_strength_mpa: Optional[float] = None,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    force_magnitude_n: Optional[float] = None,
    force_direction: Optional[List[float]] = None,
    force_face_normal: Optional[List[float]] = None,
    force_face_axis_extreme: Optional[str] = None,
    pressure_mpa: Optional[float] = None,
    pressure_face_normal: Optional[List[float]] = None,
    pressure_face_axis_extreme: Optional[str] = None,
    extra_supports: Optional[List[Dict[str, Any]]] = None,
    extra_loads: Optional[List[Dict[str, Any]]] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Statik Yapısal Analiz için malzeme, mesh ve sınır şartlarını yapılandırır.

    Hiçbir değer sessizce uydurulmaz: mesnet yüzeyi, yük yüzeyi ve yük büyüklüğü
    açıkça verilmelidir; eksik veya geçersiz bir girdi hata döner. Malzeme için ya
    dört özelliği (E, poisson, akma, yoğunluk) verin ya da yalnızca 'material_name'
    verin; ad 'lookup_material' kütüphanesinde bulunursa kartı oradan alınır.

    Yüzler iki şekilde seçilebilir; ikisi birden verilirse ikisi de puanlanır:
      * Normal vektörüyle: örn. ankastre yüz için [-1, 0, 0].
      * Eksen ekstremumuyla: 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'.
    Hangi yüzlerin mevcut olduğunu görmek için 'inspect_live_geometry_faces' kullanın.

    Yük olarak kuvvet ve/veya basınç verilebilir; en az biri zorunludur. Daha fazlası
    (birden çok yük/mesnet, moment, yerçekimi, uzak kuvvet, sürtünmesiz mesnet,
    verilen yer değiştirme) 'extra_supports' ve 'extra_loads' listeleriyle tanımlanır.
    Bu listelerle, kuvvet/basınç hiç verilmeden de kurulum yapılabilir.

    extra_supports öğeleri ('type' zorunlu; yüz için face_normal ve/veya face_axis_extreme):
      {"type": "fixed", "face_axis_extreme": "max_x"}
      {"type": "frictionless", "face_normal": [1, 0, 0]}
      {"type": "displacement", "face_axis_extreme": "max_x", "ux_mm": 0.05}
        (verilmeyen bileşen serbest kalır; en az biri gerekir)

    extra_loads öğeleri:
      {"type": "force", "magnitude_n": 50, "direction": [0, -1, 0], "face_axis_extreme": "max_x"}
      {"type": "pressure", "magnitude_mpa": 2.5, "face_axis_extreme": "max_y"}
      {"type": "moment", "magnitude_nm": 12, "axis": [0, 0, 1], "face_axis_extreme": "max_x"}
      {"type": "remote_force", "magnitude_n": 50, "direction": [0, -1, 0],
       "origin_mm": [585, 0, 0], "face_axis_extreme": "max_x"}
      {"type": "gravity", "direction": [0, 0, -1]}   # büyüklük verilmezse 9.80665 m/s^2
    Yerçekimi için malzeme yoğunluğu gereklidir.

    Args:
        job_id: create_analysis ile üretilen 'static_structural' analiz kimliği.
        cad_file_path: Katı CAD dosyası (STEP / IGES / Parasolid / SAT). STL kabul edilmez.
        material_name: Malzeme adı (kütüphane anahtarı veya Engineering Data adı).
        youngs_modulus_gpa: Elastisite modülü (GPa). Boşsa kütüphaneden okunur.
        poissons_ratio: Poisson oranı [0, 0.5). Boşsa kütüphaneden okunur.
        yield_strength_mpa: Akma mukavemeti (MPa). Boşsa kütüphaneden okunur.
        density_kg_m3: Yoğunluk (kg/m^3). Boşsa kütüphaneden okunur.
        ultimate_strength_mpa: Kopma mukavemeti (MPa); boşsa akma değeri kullanılır ve not düşülür.
        mesh_global_size_mm: Global mesh eleman boyutu (mm).
        fixed_support_normal: Ankastre mesnet yüzeyinin normali [nx, ny, nz].
        fixed_support_axis_extreme: Alternatif seçim, örn. 'min_x'.
        force_magnitude_n: Uygulanacak kuvvet (N), sıfırdan büyük.
        force_direction: Kuvvetin yön vektörü [dx, dy, dz]; sıfır vektör olamaz.
        force_face_normal: Kuvvetin uygulanacağı yüzeyin normali.
        force_face_axis_extreme: Alternatif seçim, örn. 'max_x'.
        pressure_mpa: Yüzeye dik basınç (MPa), sıfırdan büyük; yüzeye doğru etkir.
        pressure_face_normal: Basınç yüzeyinin normali.
        pressure_face_axis_extreme: Alternatif seçim, örn. 'max_y'.
        extra_supports: Ek mesnetler (yukarıdaki biçim).
        extra_loads: Ek yükler (yukarıdaki biçim).
        orthotropic: Ortotropik elastisite. Dokuz sabitin hepsi zorunludur: ex_gpa, ey_gpa,
            ez_gpa, nu_xy, nu_yz, nu_xz, gxy_gpa, gyz_gpa, gxz_gpa (eksenler global X, Y, Z).
            Verilirse E ve poisson boş bırakılabilir (ex ve nu_xy alınır); akma ve yoğunluk
            yine gerekir. Ortotropik malzemede Von-Mises güvenlik faktörü geçerli değildir
            ve sonuçta raporlanmaz. 'carbon orthotropic' kütüphane kartı da kullanılabilir.
        layers: Katmanlı kompozit kesit (yalnızca YÜZEY/kabuk gövdeli STEP için). Her eleman {'thickness_mm': >0, 'angle_deg': [-180, 180]};
            alttan üste sıralanır. Ply malzemesi ortotropik olmalıdır (ex = lif yönü, 'orthotropic' veya ortotropik kütüphane kartı). Açı,
            yüzeye izdüşürülen global X ekseninden ölçülür. Mesnet ve yükler kabuk gövdenin KENARLARINA uygulanır
            ('fixed_support_axis_extreme' vb. uç kenarları seçer). Katı CAD ile verilirse hata döner.
    """
    return _configure_structural(
        AnalysisType.STATIC_STRUCTURAL.value,
        job_id=job_id,
        cad_file_path=cad_file_path,
        material_name=material_name,
        youngs_modulus_gpa=youngs_modulus_gpa,
        poissons_ratio=poissons_ratio,
        yield_strength_mpa=yield_strength_mpa,
        density_kg_m3=density_kg_m3,
        ultimate_strength_mpa=ultimate_strength_mpa,
        mesh_global_size_mm=mesh_global_size_mm,
        fixed_support_normal=fixed_support_normal,
        fixed_support_axis_extreme=fixed_support_axis_extreme,
        force_magnitude_n=force_magnitude_n,
        force_direction=force_direction,
        force_face_normal=force_face_normal,
        force_face_axis_extreme=force_face_axis_extreme,
        pressure_mpa=pressure_mpa,
        pressure_face_normal=pressure_face_normal,
        pressure_face_axis_extreme=pressure_face_axis_extreme,
        extra_supports=extra_supports,
        extra_loads=extra_loads,
        orthotropic=orthotropic,
        layers=layers,
    )


@mcp.tool()
def configure_buckling_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str,
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    ultimate_strength_mpa: Optional[float] = None,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    force_magnitude_n: Optional[float] = None,
    force_direction: Optional[List[float]] = None,
    force_face_normal: Optional[List[float]] = None,
    force_face_axis_extreme: Optional[str] = None,
    pressure_mpa: Optional[float] = None,
    pressure_face_normal: Optional[List[float]] = None,
    pressure_face_axis_extreme: Optional[str] = None,
    extra_supports: Optional[List[Dict[str, Any]]] = None,
    extra_loads: Optional[List[Dict[str, Any]]] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    num_modes: int = 3,
    layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Doğrusal özdeğer burkulma analizini yapılandırır.

    Kurulum 'configure_static_analysis' ile aynıdır (aynı malzeme, mesnet, yük ve
    'extra_supports' / 'extra_loads' kuralları, hiçbir değer uydurulmaz). Fark: verdiğiniz
    yük bir REFERANS yüktür. Çözücü, ön gerilmeli yapının burkulmaya gittiği yük
    çarpanlarını (kritik yük / referans yük) hesaplar. Kritik yük = çarpan x referans yük.

    Basınç yükü olan bir kolon için referans olarak basit bir değer (örn. 1 N eksenel
    kuvvet, mesnete doğru) verirseniz çarpan doğrudan kritik yükü Newton olarak verir.
    Burkulma yönü yük yönüne duyarlıdır: basınç yaratan (mesnete doğru) bir yük verin.

    Args:
        num_modes: Hesaplanacak burkulma modu sayısı (1-10). Diğer argümanlar için
            'configure_static_analysis' açıklamasına bakın.
    """
    if not isinstance(num_modes, int) or isinstance(num_modes, bool) or not (1 <= num_modes <= 10):
        return {"status": "error", "message": "num_modes 1 ile 10 arasında bir tam sayı olmalıdır."}
    return _configure_structural(
        AnalysisType.BUCKLING.value,
        job_id=job_id,
        cad_file_path=cad_file_path,
        material_name=material_name,
        youngs_modulus_gpa=youngs_modulus_gpa,
        poissons_ratio=poissons_ratio,
        yield_strength_mpa=yield_strength_mpa,
        density_kg_m3=density_kg_m3,
        ultimate_strength_mpa=ultimate_strength_mpa,
        mesh_global_size_mm=mesh_global_size_mm,
        fixed_support_normal=fixed_support_normal,
        fixed_support_axis_extreme=fixed_support_axis_extreme,
        force_magnitude_n=force_magnitude_n,
        force_direction=force_direction,
        force_face_normal=force_face_normal,
        force_face_axis_extreme=force_face_axis_extreme,
        pressure_mpa=pressure_mpa,
        pressure_face_normal=pressure_face_normal,
        pressure_face_axis_extreme=pressure_face_axis_extreme,
        extra_supports=extra_supports,
        extra_loads=extra_loads,
        orthotropic=orthotropic,
        num_modes=num_modes,
        layers=layers,
    )


@mcp.tool()
def configure_harmonic_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str,
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    ultimate_strength_mpa: Optional[float] = None,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    force_magnitude_n: Optional[float] = None,
    force_direction: Optional[List[float]] = None,
    force_face_normal: Optional[List[float]] = None,
    force_face_axis_extreme: Optional[str] = None,
    pressure_mpa: Optional[float] = None,
    pressure_face_normal: Optional[List[float]] = None,
    pressure_face_axis_extreme: Optional[str] = None,
    extra_supports: Optional[List[Dict[str, Any]]] = None,
    extra_loads: Optional[List[Dict[str, Any]]] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    frequency_min_hz: float = 0.0,
    frequency_max_hz: float = 0.0,
    intervals: int = 20,
    damping_ratio: float = 0.0,
    method: str = "full",
    layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Harmonik (frekans taramalı) tepki analizini yapılandırır.

    Kurulum 'configure_static_analysis' ile aynıdır; tanımladığınız yük bir HARMONİK uyarımın
    GENLİĞİdir (sıfır fazlı). Çözücü [frequency_min_hz, frequency_max_hz] aralığını 'intervals'
    eşit adımda tarar ve her frekanstaki en büyük yer değiştirme genliğini verir. Hiçbir değer
    uydurulmaz: frekans aralığı ve sönümleme zorunludur.

    Args:
        frequency_min_hz: Tarama başlangıcı (Hz, >= 0).
        frequency_max_hz: Tarama sonu (Hz, > başlangıç).
        intervals: Tarama adım sayısı (2-200). Çözülen frekanslar min + k*(max-min)/intervals, k=1..intervals.
        damping_ratio: Kritik sönümün oranı (0 < değer < 1). Rezonansta genlik ~1/(2*oran) kadar büyür.
        method: 'full' (mod kesmesi yok, kesin; sönüm eşdeğer histeretik katsayıyla) veya
            'mode_superposition' (hızlı, ama yüksek modlar kesildiği için düşük frekansta yüzde birkaç düşük çıkar).
            layers: Katmanlı kompozit kesit (yüzey/kabuk gövde, 'orthotropic' ply malzemesi gerekir; ayrıntı
            'configure_static_analysis' açıklamasında). Diğer argümanlar için 'configure_static_analysis' açıklamasına bakın.
    """
    try:
        f_min = require_finite("frequency_min_hz", frequency_min_hz)
        f_max = require_finite("frequency_max_hz", frequency_max_hz)
        zeta = require_finite("damping_ratio", damping_ratio)
        if f_min < 0 or f_max <= f_min:
            raise InputError("Frekans aralığı 0 <= frequency_min_hz < frequency_max_hz olmalıdır.")
        if not (0.0 < zeta < 1.0):
            raise InputError("damping_ratio 0 ile 1 arasında olmalıdır (0 ve 1 hariç); sönümsüz rezonans sonsuz genlik verir.")
        if isinstance(intervals, bool) or not isinstance(intervals, int) or not (2 <= intervals <= 200):
            raise InputError("intervals 2 ile 200 arasında bir tam sayı olmalıdır.")
        if method not in ("full", "mode_superposition"):
            raise InputError("method 'full' veya 'mode_superposition' olmalıdır.")
    except InputError as exc:
        return {"status": "error", "message": str(exc)}
    return _configure_structural(
        AnalysisType.HARMONIC.value,
        job_id=job_id,
        cad_file_path=cad_file_path,
        material_name=material_name,
        youngs_modulus_gpa=youngs_modulus_gpa,
        poissons_ratio=poissons_ratio,
        yield_strength_mpa=yield_strength_mpa,
        density_kg_m3=density_kg_m3,
        ultimate_strength_mpa=ultimate_strength_mpa,
        mesh_global_size_mm=mesh_global_size_mm,
        fixed_support_normal=fixed_support_normal,
        fixed_support_axis_extreme=fixed_support_axis_extreme,
        force_magnitude_n=force_magnitude_n,
        force_direction=force_direction,
        force_face_normal=force_face_normal,
        force_face_axis_extreme=force_face_axis_extreme,
        pressure_mpa=pressure_mpa,
        pressure_face_normal=pressure_face_normal,
        pressure_face_axis_extreme=pressure_face_axis_extreme,
        extra_supports=extra_supports,
        extra_loads=extra_loads,
        orthotropic=orthotropic,
        extra_config={
            "frequency_min_hz": f_min, "frequency_max_hz": f_max, "intervals": intervals,
            "damping_ratio": zeta, "method": method,
        },
        layers=layers,
    )


@mcp.tool()
def configure_transient_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str,
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    ultimate_strength_mpa: Optional[float] = None,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    force_magnitude_n: Optional[float] = None,
    force_direction: Optional[List[float]] = None,
    force_face_normal: Optional[List[float]] = None,
    force_face_axis_extreme: Optional[str] = None,
    pressure_mpa: Optional[float] = None,
    pressure_face_normal: Optional[List[float]] = None,
    pressure_face_axis_extreme: Optional[str] = None,
    extra_supports: Optional[List[Dict[str, Any]]] = None,
    extra_loads: Optional[List[Dict[str, Any]]] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    end_time_s: float = 0.0,
    time_step_s: float = 0.0,
    damping_alpha_per_s: Optional[float] = None,
    damping_beta_s: Optional[float] = None,
    layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Geçici (zaman tanım alanı) yapısal analizi yapılandırır.

    Kurulum 'configure_static_analysis' ile aynıdır; tanımladığınız yükler t=0'dan itibaren
    sabit genlikle uygulanır (basamak yük). Çözücü [0, end_time_s] aralığını sabit
    'time_step_s' adımıyla ilerletir ve zaman içindeki en büyük yer değiştirme geçmişini verir.
    Hiçbir değer uydurulmaz: süre, adım ve Rayleigh sönüm katsayıları zorunludur.

    Args:
        end_time_s: Toplam süre (s, > 0).
        time_step_s: Sabit zaman adımı (s, > 0). Adım sayısı (end_time_s / time_step_s) 2 ile 2000
            arasında bir tam sayı olmalıdır. Adım, en yüksek ilgilenilen modun periyodunun
            yaklaşık 1/20'sinden küçük olmalıdır; aksi halde uyarı verilir.
        damping_alpha_per_s: Rayleigh kütle katsayısı alfa (1/s, >= 0). Sönümsüz için 0 verin.
        damping_beta_s: Rayleigh rijitlik katsayısı beta (s, >= 0). Mod i için sönüm oranı
            zeta = alfa/(2 w) + beta w/2.
            layers: Katmanlı kompozit kesit (yüzey/kabuk gövde, 'orthotropic' ply malzemesi gerekir; ayrıntı
            'configure_static_analysis' açıklamasında). Diğer argümanlar için 'configure_static_analysis' açıklamasına bakın.
    """
    try:
        t_end = require_finite("end_time_s", end_time_s)
        dt = require_finite("time_step_s", time_step_s)
        if t_end <= 0 or dt <= 0 or dt >= t_end:
            raise InputError("0 < time_step_s < end_time_s olmalıdır.")
        steps = t_end / dt
        if abs(steps - round(steps)) > 1e-6 * max(1.0, steps) or not (2 <= round(steps) <= 2000):
            raise InputError("end_time_s / time_step_s 2 ile 2000 arasında bir tam sayı olmalıdır (şu an %.6g)." % steps)
        if damping_alpha_per_s is None or damping_beta_s is None:
            raise InputError("damping_alpha_per_s ve damping_beta_s zorunludur (sönümsüz için 0 verin).")
        alpha = require_finite("damping_alpha_per_s", damping_alpha_per_s)
        beta = require_finite("damping_beta_s", damping_beta_s)
        if alpha < 0 or beta < 0:
            raise InputError("Rayleigh katsayıları negatif olamaz.")
    except InputError as exc:
        return {"status": "error", "message": str(exc)}
    return _configure_structural(
        AnalysisType.TRANSIENT.value,
        job_id=job_id,
        cad_file_path=cad_file_path,
        material_name=material_name,
        youngs_modulus_gpa=youngs_modulus_gpa,
        poissons_ratio=poissons_ratio,
        yield_strength_mpa=yield_strength_mpa,
        density_kg_m3=density_kg_m3,
        ultimate_strength_mpa=ultimate_strength_mpa,
        mesh_global_size_mm=mesh_global_size_mm,
        fixed_support_normal=fixed_support_normal,
        fixed_support_axis_extreme=fixed_support_axis_extreme,
        force_magnitude_n=force_magnitude_n,
        force_direction=force_direction,
        force_face_normal=force_face_normal,
        force_face_axis_extreme=force_face_axis_extreme,
        pressure_mpa=pressure_mpa,
        pressure_face_normal=pressure_face_normal,
        pressure_face_axis_extreme=pressure_face_axis_extreme,
        extra_supports=extra_supports,
        extra_loads=extra_loads,
        orthotropic=orthotropic,
        extra_config={
            "end_time_s": t_end, "time_step_s": dt,
            "damping_alpha_per_s": alpha, "damping_beta_s": beta,
        },
        layers=layers,
    )


@mcp.tool()
def configure_random_vibration_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str,
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    ultimate_strength_mpa: Optional[float] = None,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    extra_supports: Optional[List[Dict[str, Any]]] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    base_axis: str = "",
    psd_frequencies_hz: Optional[List[float]] = None,
    psd_values_g2_per_hz: Optional[List[float]] = None,
    damping_ratio: float = 0.0,
    num_modes: int = 10,
    layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Rastgele titreşim (PSD) analizini yapılandırır: tabandan ivme güç spektral yoğunluğu (ASD) uyarımı.

    Mesnetler 'configure_static_analysis' ile aynıdır, ancak yük verilmez: uyarım, TÜM sabit
    mesnetlere 'base_axis' doğrultusunda uygulanan taban ivmesi PSD'sidir. Çözücü önce modal
    çözüm yapar, sonra PSD'yi modlara yayıp tepkinin 1 sigma (RMS) değerlerini verir. Hiçbir değer
    uydurulmaz: eksen, PSD tablosu, sönüm oranı zorunludur.

    Args:
        base_axis: Taban ivmesinin doğrultusu: 'x', 'y' veya 'z' (global).
        psd_frequencies_hz: Artan frekans noktaları (Hz, > 0, en az 2, en çok 100).
        psd_values_g2_per_hz: Her noktadaki ivme PSD'si (g^2/Hz, > 0), frekans listesiyle aynı uzunlukta.
            Noktalar arası log-log doğrusal enterpolasyon yapılır.
        damping_ratio: Sabit sönüm oranı (kritik sönümün oranı, 0 < oran < 1).
        num_modes: Hesaplanacak mod sayısı (1-50). En yüksek mod frekansı PSD'nin üst
            frekansından küçükse çözüm sonunda uyarı verilir.
            layers: Katmanlı kompozit kesit (yüzey/kabuk gövde, 'orthotropic' ply malzemesi gerekir; ayrıntı
            'configure_static_analysis' açıklamasında). Diğer argümanlar için 'configure_static_analysis' açıklamasına bakın.
    """
    try:
        axis = str(base_axis).strip().lower()
        if axis not in ("x", "y", "z"):
            raise InputError("base_axis 'x', 'y' veya 'z' olmalıdır.")
        if not isinstance(psd_frequencies_hz, (list, tuple)) or not isinstance(psd_values_g2_per_hz, (list, tuple)):
            raise InputError("psd_frequencies_hz ve psd_values_g2_per_hz liste olarak verilmelidir.")
        if len(psd_frequencies_hz) != len(psd_values_g2_per_hz) or not (2 <= len(psd_frequencies_hz) <= 100):
            raise InputError("PSD tablosu 2 ile 100 arasında, eşit uzunlukta frekans/değer noktası içermelidir.")
        freqs = [require_positive("psd_frequencies_hz[%d]" % i, v) for i, v in enumerate(psd_frequencies_hz)]
        values = [require_positive("psd_values_g2_per_hz[%d]" % i, v) for i, v in enumerate(psd_values_g2_per_hz)]
        if any(b <= a for a, b in zip(freqs, freqs[1:])):
            raise InputError("psd_frequencies_hz kesin artan olmalıdır.")
        zeta = require_finite("damping_ratio", damping_ratio)
        if not (0.0 < zeta < 1.0):
            raise InputError("damping_ratio 0 ile 1 arasında olmalıdır (0 ve 1 hariç).")
        if isinstance(num_modes, bool) or not isinstance(num_modes, int) or not (1 <= num_modes <= 50):
            raise InputError("num_modes 1 ile 50 arasında bir tam sayı olmalıdır.")
    except InputError as exc:
        return {"status": "error", "message": str(exc)}
    return _configure_structural(
        AnalysisType.RANDOM_VIBRATION.value,
        job_id=job_id,
        cad_file_path=cad_file_path,
        material_name=material_name,
        youngs_modulus_gpa=youngs_modulus_gpa,
        poissons_ratio=poissons_ratio,
        yield_strength_mpa=yield_strength_mpa,
        density_kg_m3=density_kg_m3,
        ultimate_strength_mpa=ultimate_strength_mpa,
        mesh_global_size_mm=mesh_global_size_mm,
        fixed_support_normal=fixed_support_normal,
        fixed_support_axis_extreme=fixed_support_axis_extreme,
        extra_supports=extra_supports,
        orthotropic=orthotropic,
        num_modes=num_modes,
        require_load=False,
        extra_config={
            "base_axis": axis, "psd_frequencies_hz": freqs, "psd_values_g2_per_hz": values,
            "damping_ratio": zeta,
        },
        layers=layers,
    )


@mcp.tool()
def validate_analysis(job_id: str) -> Dict[str, Any]:
    """
    Analizin fiziksel ve sayısal bütünlüğünü denetler (Gatekeeper Validator).
    Çözümün başlatılabilmesi için bu fonksiyonun 'valid: True' dönmesi şarttır.
    
    Args:
        job_id: Analiz kimliği.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Analiz bulunamadı veya henüz konfigüre edilmedi."}

    cfg = json.loads(job["config_json"])
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg["boundary_conditions"]]

    report = GatekeeperValidator.validate_setup(
        job_id=job_id,
        analysis_type=AnalysisType(job["analysis_type"]),
        cad_file_path=job["cad_file_path"],
        material=material,
        mesh_config=mesh_cfg,
        boundary_conditions=bcs
    )

    storage.save_validation_report(job_id=job_id, report=report)

    return {
        "status": "validated",
        "job_id": job_id,
        "valid": report.valid,
        "errors": report.errors,
        "warnings": report.warnings,
        "checks_passed": sum(1 for c in report.checks if c.passed),
        "total_checks": len(report.checks),
        "message": "Doğrulama başarılı. 'submit_analysis' ile çözümü başlatabilirsiniz." if report.valid else "Doğrulama başarısız! Hataları düzeltin."
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def submit_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Doğrulanmış analizi GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Bu araç script üretmez; canlı bir Mechanical uygulamasını sürer. Geometri
    gerçek ağaca yüklenir, Engineering Data'ya gerçek malzeme girer, gerçek
    mesher ve gerçek MAPDL çözücüsü çalışır, sonuçlar çözülmüş modelden okunur.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır ve her adım izlenebilir.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": "Analiz bulunamadı."}

    if job["analysis_type"] != AnalysisType.STATIC_STRUCTURAL.value:
        return {
            "status": "error",
            "message": "submit_analysis yalnızca 'static_structural' job içindir; bu job '%s'. "
            "Burkulma için 'solve_buckling_analysis', modal için 'solve_modal_analysis' kullanın."
            % job["analysis_type"],
        }

    if not job["validation_json"]:
        return {"status": "error", "message": "Analiz henüz doğrulanmadı! Önce 'validate_analysis' çalıştırılmalıdır."}

    v_report = json.loads(job["validation_json"])
    if not v_report.get("valid", False):
        return {"status": "error", "message": f"Doğrulamadan geçemeyen analiz çözülemez! Hatalar: {v_report.get('errors')}"}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg["boundary_conditions"]]

    adapter = MechanicalAdapter(visible=show_gui)

    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)
        layered_report = adapter.apply_layered_section(cfg["layers"]) if cfg.get("layers") else None

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        bc_report = adapter.apply_boundary_conditions(bcs)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        results = adapter.extract_results()
        images = adapter.export_result_images()
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()

    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS çözümü başarısız oldu. Sentetik sonuç üretilmedi.",
        }

    summary = PostProcessingAdapter.generate_engineering_summary(material, results)
    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])

    solver_report = {
        "solver": solve_res["solver"],
        "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"],
        "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"],
        "solver_messages": solve_res["messages"],
        "mechdb": mechdb,
        "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "face_count": len(geometry.get("faces", [])),
        "material": mat_report,
        "mesh": mesh_report,
        "boundary_conditions": bc_report["boundary_conditions"],
        "equilibrium": solve_res["equilibrium"],
        "executed_transcript": transcript,
    }

    manifest = AnalysisManifest(
        job_id=job_id,
        analysis_type=AnalysisType(job["analysis_type"]),
        created_at=job["created_at"],
        cad_file_path=job["cad_file_path"],
        cad_sha256=cad_sha256,
        material=material,
        mesh_config=mesh_cfg,
        boundary_conditions=bcs,
        results=results,
        artifacts=sorted(images.values()) + [transcript, mechdb]
        + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED,
        solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    return {
        "status": "completed",
        "job_id": job_id,
        "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"],
        "solve_time_s": solve_res["solve_time_s"],
        "rst_file": solve_res["rst_file"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "equilibrium": solve_res["equilibrium"],
        "results": [r.model_dump() for r in results],
        "engineering_summary": summary,
        "plots": images,
        "mechdb": mechdb,
        "message": "Gerçek ANSYS 2025 R1 çözümü tamamlandı. Mechanical penceresi açık kaldı, modeli inceleyebilirsiniz.",
    }


@mcp.tool()
def export_mechanical_script(job_id: str) -> Dict[str, Any]:
    """
    Analizi üreten ve GERÇEKTEN çalıştırılmış Mechanical adım dökümünü döner.

    Args:
        job_id: Analiz kimliği.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Analiz bulunamadı."}

    job_dir = storage.get_job_dir(job_id)
    transcript_path = os.path.join(job_dir, "executed_mechanical_steps.log")

    if not os.path.exists(transcript_path):
        return {
            "status": "error",
            "message": "Adım dökümü yok. Analiz henüz 'submit_analysis' ile çalıştırılmamış.",
        }

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "status": "success",
        "job_id": job_id,
        "transcript_path": transcript_path,
        "note": "Bu döküm, canlı ANSYS oturumunda gerçekten yürütülen adımların kaydıdır.",
        "content": content,
    }


@mcp.tool()
def get_static_equilibrium_status(job_id: str) -> Dict[str, Any]:
    """
    ANSYS'in hesapladığı mesnet reaksiyon kuvvetiyle Newton I dengesini döner.

    Args:
        job_id: Analiz kimliği.
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış analiz bulunamadı."}

    manifest = json.loads(job["manifest_json"])
    report = (manifest.get("solver_report") or {}).get("equilibrium")
    if not report:
        return {
            "status": "error",
            "message": "Denge raporu yok. Analiz canlı ANSYS oturumunda çözülmemiş olabilir.",
        }

    return {"status": "success", "job_id": job_id, "equilibrium": report}


@mcp.tool()
def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    İşin anlık yaşam döngüsü durumunu (State) ve loglarını döner.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}

    status = {
        "job_id": job["job_id"],
        "state": job["state"],
        "analysis_type": job["analysis_type"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error_message": job["error_message"]
    }
    if _ACTIVE_RUN.get("job_id") == job_id:
        status["running_tool"] = _ACTIVE_RUN.get("tool")
        status["elapsed_s"] = round(time.time() - _ACTIVE_RUN.get("started", time.time()), 1)
        status["cancel_requested"] = job_id in _CANCEL_REQUESTED
        try:
            live = get_live_session_steps_tail(job_id)
        except Exception:
            live = None
        if live:
            status["recent_steps"] = live
    return status


def get_live_session_steps_tail(job_id: str, count: int = 5):
    """Last few Mechanical steps of the run in progress, if one is using Mechanical."""
    from .adapters import mechanical_live

    session = getattr(mechanical_live, "_SESSION", None)
    if session is None or not getattr(session, "step_log", None):
        return None
    return [s.get("tag", s) if isinstance(s, dict) else s for s in session.step_log[-count:]]


@mcp.tool()
def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Çalışmakta olan çözümü iptal eder.

    Çözücü süreci (Mechanical veya Fluent) sonlandırılır, iş 'CANCELLED' olur ve çözücü
    kilidi serbest kalır. Çözümü çalışmayan bir iş iptal edilemez. Mechanical oturumu
    yeniden başlatılana kadar sonraki çağrı yeni bir oturum açar (yaklaşık 30 sn).

    Args:
        job_id: İptal edilecek işin kimliği.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if _ACTIVE_RUN.get("job_id") != job_id:
        return {
            "status": "error",
            "message": "Bu iş şu an çözülmüyor (durum: %s); iptal edilecek bir şey yok." % job["state"],
        }
    _CANCEL_REQUESTED.add(job_id)
    closer = _ACTIVE_RUN.get("close")
    try:
        if closer is not None:
            closer()
        else:
            from .adapters import mechanical_live

            session = getattr(mechanical_live, "_SESSION", None)
            if session is not None:
                session.close(exit_app=True)
            mechanical_live.reset_live_session()
    except Exception as exc:
        return {"status": "error", "message": "Çözücü durdurulamadı: %s" % exc}
    return {
        "status": "cancelling",
        "job_id": job_id,
        "tool": _ACTIVE_RUN.get("tool"),
        "message": "Çözücü sonlandırıldı; iş birkaç saniye içinde 'CANCELLED' olacak.",
    }


@mcp.tool()
def get_analysis_results(job_id: str) -> Dict[str, Any]:
    """
    Tamamlanan analizin deterministik mühendislik metriklerini döner.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}

    if job["state"] != JobState.COMPLETED.value or not job["manifest_json"]:
        return {
            "status": "pending",
            "job_id": job_id,
            "state": job["state"],
            "message": "Analiz henüz tamamlanmadı."
        }

    manifest = json.loads(job["manifest_json"])
    material = Material(**manifest["material"])
    results = [ResultMetric(**r) for r in manifest["results"]]
    summary = PostProcessingAdapter.generate_engineering_summary(material, results)

    return {
        "status": "success",
        "job_id": job_id,
        "manifest": manifest,
        "engineering_assessment": summary
    }


# ---------------------------------------------------------------------------
# 1.05 İş Yönetimi, Malzeme, Birim ve El Hesabı Araçları
# ---------------------------------------------------------------------------

@mcp.tool()
def list_jobs(
    state: Optional[str] = None,
    analysis_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Kayıtlı analiz job'larını en yeniden eskiye listeler.

    Args:
        state: Yalnızca bu durumdaki job'lar (örn. 'COMPLETED', 'FAILED').
        analysis_type: Yalnızca bu türdekiler (örn. 'static_structural').
        limit: En fazla kaç kayıt (1-200).
        offset: Atlanacak kayıt sayısı.
    """
    if not (1 <= limit <= 200) or offset < 0:
        return {"status": "error", "message": "1 <= limit <= 200 ve offset >= 0 olmalıdır."}
    if state and state not in {s.value for s in JobState}:
        return {"status": "error", "message": "Bilinmeyen durum: %r. Geçerli: %s" % (state, [s.value for s in JobState])}
    if analysis_type and analysis_type not in {a.value for a in AnalysisType}:
        return {"status": "error", "message": "Bilinmeyen analiz türü: %r" % analysis_type}
    page = storage.list_jobs(state=state, analysis_type=analysis_type, limit=limit, offset=offset)
    page["status"] = "success"
    page["next_offset"] = offset + limit if offset + limit < page["total"] else None
    return page


@mcp.tool()
def delete_job(job_id: str, delete_files: bool = False) -> Dict[str, Any]:
    """
    Bir job'un kaydını siler. GERİ ALINAMAZ.

    Varsayılan olarak yalnızca veritabanı kaydı silinir; çözüm dosyaları
    (.rst, .mechdb, grafikler) diskte kalır. 'delete_files=True' iş klasörünü de
    kalıcı olarak siler.

    Args:
        job_id: Silinecek analiz kimliği.
        delete_files: True ise runs/<job_id> klasörü de silinir.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if job["state"] in (
        JobState.QUEUED.value, JobState.IMPORTING.value, JobState.MESHING.value,
        JobState.SOLVING.value, JobState.POST_PROCESSING.value,
    ):
        return {
            "status": "error",
            "message": "Job çalışıyor (%s); çalışan bir job silinemez." % job["state"],
        }
    storage.delete_job(job_id, delete_files=delete_files)
    return {"status": "success", "job_id": job_id, "files_deleted": bool(delete_files)}


@mcp.tool()
def clone_job(job_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Bir job'un konfigürasyonunu yeni bir job'a kopyalar (parametre değiştirip tekrar çözmek için).

    Çözüm sonuçları kopyalanmaz. 'overrides' ile şu değerler değiştirilebilir:
    'mesh_global_size_mm', 'material' (alan alan), 'force_magnitude_n', 'pressure_mpa'.
    Yeni job CONFIGURED durumunda gelir; 'validate_analysis' ile devam edin.

    Args:
        job_id: Kaynak analiz kimliği.
        overrides: Değiştirilecek alanlar, örn. {"mesh_global_size_mm": 2.0}.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if not job["config_json"]:
        return {"status": "error", "message": "Kaynak job konfigüre edilmemiş; kopyalanacak bir şey yok."}

    cfg = json.loads(job["config_json"])
    overrides = dict(overrides or {})
    try:
        if "mesh_global_size_mm" in overrides:
            size = require_positive("mesh_global_size_mm", overrides.pop("mesh_global_size_mm"))
            cfg["mesh_config"]["global_size"] = {"value": size, "unit": "mm"}
        if "force_magnitude_n" in overrides:
            value = require_positive("force_magnitude_n", overrides.pop("force_magnitude_n"))
            forces = [b for b in cfg["boundary_conditions"] if b["type"] == "force"]
            if not forces:
                return {"status": "error", "message": "Kaynak job'da değiştirilecek bir kuvvet yok."}
            for b in forces:
                b["magnitude"] = {"value": value, "unit": "N"}
        if "pressure_mpa" in overrides:
            value = require_positive("pressure_mpa", overrides.pop("pressure_mpa"))
            presses = [b for b in cfg["boundary_conditions"] if b["type"] == "pressure"]
            if not presses:
                return {"status": "error", "message": "Kaynak job'da değiştirilecek bir basınç yok."}
            for b in presses:
                b["magnitude"] = {"value": value, "unit": "MPa"}
        if "material" in overrides:
            patch = overrides.pop("material")
            if not isinstance(patch, dict):
                return {"status": "error", "message": "'material' bir sözlük olmalıdır."}
            cfg["material"].update(patch)
            merged = Material(**cfg["material"])
            problems = check_material_card(
                merged.youngs_modulus, merged.poissons_ratio, merged.yield_strength,
                merged.ultimate_strength, merged.density,
            )
            if problems:
                return {"status": "error", "message": "Malzeme kartı geçersiz: " + " | ".join(problems)}
            cfg["material"] = merged.model_dump()
    except (InputError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}
    if overrides:
        return {
            "status": "error",
            "message": "Desteklenmeyen override anahtarları: %s" % sorted(overrides),
        }

    new_id = f"{job['analysis_type'].upper()}-{os.urandom(4).hex()}"
    storage.clone_job(job_id, new_id, cfg)
    return {
        "status": "success",
        "job_id": new_id,
        "source_job_id": job_id,
        "state": JobState.CONFIGURED.value,
        "message": "Kopya hazır. 'validate_analysis' ile doğrulayıp 'submit_analysis' ile çözün.",
    }


@mcp.tool()
def lookup_material(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Malzeme kütüphanesinden özellik kartı döner (E, poisson, akma, yoğunluk, ısıl değerler).

    Değerler oda sıcaklığı için tipik el kitabı verileridir; tedarikçi veri
    sayfasıyla doğrulanmalıdır. Kompozit gibi anizotropik malzemeler için kart,
    'fidelity' alanında izotropik varsayımın sınırını açıkça belirtir.

    Args:
        name: Malzeme adı veya takma adı ('6061', 'titanyum', 'karbon'). Boşsa liste döner.
    """
    if not name:
        return {"status": "success", "materials": list_materials()}
    card = lookup_material_card(name)
    if card is None:
        return {
            "status": "error",
            "message": "'%s' kütüphanede yok." % name,
            "materials": list_materials(),
        }
    return {"status": "success", "material": card}


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """
    Bir değeri bir birimden diğerine çevirir (MPa/GPa/Pa, mm/m, N/kN, C/K, kg/m^3 ...).

    Args:
        value: Çevrilecek sayı.
        from_unit: Kaynak birim.
        to_unit: Hedef birim.
    """
    try:
        number = require_finite("value", value)
        out = UnitEngine.convert(Quantity(value=number, unit=from_unit), to_unit)
    except (InputError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}
    return {
        "status": "success",
        "input": {"value": number, "unit": from_unit},
        "output": {"value": out.value, "unit": out.unit},
    }


@mcp.tool()
def hand_calc_check(
    model: str,
    parameters: Dict[str, Any],
    job_id: Optional[str] = None,
    compare_metric: Optional[str] = None,
    tolerance_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    Kapalı form (el hesabı) çözümü hesaplar; isteğe bağlı olarak bir job'un çözücü sonucuyla karşılaştırır.

    Modeller ve gereken parametreler:
      * 'cantilever_end_load': shape, length_mm, force_n, youngs_modulus_gpa + kesit
        (solid_round: outer_diameter_mm | hollow_round: outer_diameter_mm, inner_diameter_mm |
        rectangle: width_mm, height_mm). Çıktı: deflection_mm, max_bending_stress_mpa.
      * 'axial_bar': yukarıdakiyle aynı. Çıktı: elongation_mm, axial_stress_mpa.
      * 'simply_supported_center_load': yukarıdakiyle aynı. Çıktı: deflection_mm,
        max_bending_stress_mpa.
      * 'thin_wall_pressure_vessel': internal_pressure_mpa, mean_radius_mm,
        wall_thickness_mm. Çıktı: hoop_stress_mpa, axial_stress_mpa.
      * 'plate_with_hole': far_field_stress_mpa, plate_width_mm, hole_diameter_mm.
        Çıktı: peak_stress_mpa.

    'shape' bir sayı değil metindir; parameters içinde "shape": "hollow_round" verin.

    Karşılaştırma: job_id ve compare_metric verilirse çözücünün sonucu
    ('maximum_total_deformation' | 'maximum_von_mises_stress') ile analitik değer
    kıyaslanır; sapma tolerance_pct içindeyse 'agrees' True döner.

    Args:
        model: Yukarıdaki model adlarından biri.
        parameters: Modelin parametreleri.
        job_id: Karşılaştırılacak tamamlanmış job.
        compare_metric: Çözücü metriği adı.
        tolerance_pct: Kabul edilen yüzde sapma.
    """
    fn = hand_calcs.MODELS.get(model)
    if fn is None:
        return {"status": "error", "message": "Bilinmeyen model: %r. Geçerli: %s" % (model, sorted(hand_calcs.MODELS))}
    try:
        tolerance = require_positive("tolerance_pct", tolerance_pct)
        params = dict(parameters)
        analytic = fn(**params)
    except TypeError as exc:
        return {"status": "error", "message": "Parametre hatası: %s" % exc}
    except (InputError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}

    out: Dict[str, Any] = {"status": "success", "analytic": analytic}
    if job_id is None and compare_metric is None:
        return out
    if job_id is None or compare_metric is None:
        return {"status": "error", "message": "Karşılaştırma için 'job_id' ve 'compare_metric' birlikte verilmelidir."}

    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Job tamamlanmamış veya sonucu yok: %s" % job_id}
    results = {r["metric_name"]: r["value"] for r in json.loads(job["manifest_json"]).get("results", [])}
    if compare_metric not in results:
        return {"status": "error", "message": "Sonuçlarda '%s' yok. Mevcut: %s" % (compare_metric, sorted(results))}

    analytic_key = {
        "maximum_total_deformation": ("deflection_mm", "elongation_mm"),
        "maximum_von_mises_stress": ("max_bending_stress_mpa", "axial_stress_mpa", "peak_stress_mpa", "hoop_stress_mpa"),
    }.get(compare_metric)
    if analytic_key is None:
        return {"status": "error", "message": "Karşılaştırılabilir metrikler: maximum_total_deformation, maximum_von_mises_stress"}
    key = next((k for k in analytic_key if k in analytic), None)
    if key is None:
        return {"status": "error", "message": "'%s' modeli '%s' metriği için analitik değer üretmiyor." % (model, compare_metric)}

    try:
        dev = hand_calcs.deviation_pct(results[compare_metric], analytic[key])
    except InputError as exc:
        return {"status": "error", "message": str(exc)}
    out["comparison"] = {
        "metric": compare_metric,
        "fea_value": results[compare_metric],
        "analytic_value": analytic[key],
        "analytic_quantity": key,
        "deviation_pct": dev,
        "tolerance_pct": tolerance,
        "agrees": abs(dev) <= tolerance,
    }
    return out


@mcp.tool()
def export_report(job_id: str, report_format: str = "markdown") -> Dict[str, Any]:
    """
    Tamamlanmış bir job için paylaşılabilir rapor dosyası üretir (job klasörüne yazılır).

    Rapor, çözücünün okuduğu sayıları, malzemeyi, sınır şartlarını, ağ bilgisini,
    doğrulama sonucunu ve üretilen grafiklerin yollarını içerir; hiçbir değer
    eklenmez veya yuvarlanmaz.

    Args:
        job_id: Tamamlanmış analiz kimliği.
        report_format: 'markdown' veya 'html'.
    """
    if report_format not in ("markdown", "html"):
        return {"status": "error", "message": "report_format 'markdown' veya 'html' olmalıdır."}
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if job["state"] != JobState.COMPLETED.value or not job["manifest_json"]:
        return {"status": "error", "message": "Job tamamlanmadı (durum: %s); raporlanacak sonuç yok." % job["state"]}

    manifest = json.loads(job["manifest_json"])
    cfg = json.loads(job["config_json"]) if job["config_json"] else {}
    validation = json.loads(job["validation_json"]) if job["validation_json"] else None

    lines: List[str] = ["# Analiz Raporu: %s" % job_id, ""]
    lines += ["- Analiz türü: %s" % job["analysis_type"], "- CAD: %s" % job["cad_file_path"],
              "- Oluşturulma: %s" % job["created_at"], ""]
    mat = cfg.get("material")
    if mat:
        lines += ["## Malzeme", "", "- Ad: %s" % mat["name"],
                  "- E: %s %s" % (mat["youngs_modulus"]["value"], mat["youngs_modulus"]["unit"]),
                  "- Poisson: %s" % mat["poissons_ratio"],
                  "- Akma: %s %s" % (mat["yield_strength"]["value"], mat["yield_strength"]["unit"]), ""]
    bcs = cfg.get("boundary_conditions") or []
    if bcs:
        lines += ["## Sınır Şartları", ""]
        for b in bcs:
            mag = b.get("magnitude")
            lines.append("- %s (%s)%s%s" % (
                b["name"], b["type"],
                ": %s %s" % (mag["value"], mag["unit"]) if mag else "",
                " yön %s" % b["direction"] if b.get("direction") else "",
            ))
        lines.append("")
    results = manifest.get("results") or []
    if results:
        lines += ["## Sonuçlar", "", "| Metrik | Değer | Birim |", "|---|---|---|"]
        for r in results:
            lines.append("| %s | %s | %s |" % (r["metric_name"], r["value"], r.get("unit", "")))
        lines.append("")
    if validation:
        lines += ["## Doğrulama", "", "- Geçerli: %s" % validation.get("valid"),
                  "- Uyarılar: %s" % (validation.get("warnings") or "yok"), ""]
    artifacts = manifest.get("artifacts") or []
    if artifacts:
        lines += ["## Dosyalar", ""] + ["- %s" % a for a in artifacts] + [""]

    body = "\n".join(lines)
    job_dir = storage.get_job_dir(job_id)
    if report_format == "markdown":
        path = os.path.join(job_dir, "report.md")
        content = body
    else:
        import html as _html
        path = os.path.join(job_dir, "report.html")
        content = (
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>Analiz Raporu %s</title></head>"
            "<body><pre>%s</pre></body></html>" % (_html.escape(job_id), _html.escape(body))
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return {"status": "success", "job_id": job_id, "report_path": path, "format": report_format}


# ---------------------------------------------------------------------------
# 1.1 Faz 4 İleri Araçlar (GCI, Singularity, Contact Pairs)
# ---------------------------------------------------------------------------

@mcp.tool()
@exclusive_solver
def run_mesh_convergence_study(
    job_id: str,
    refinement_ratio: float = 1.5,
    target_gci_pct: float = 3.0,
    show_gui: bool = False
) -> Dict[str, Any]:
    """
    3 seviyeli mesh sıklaştırma çalışmasını GERÇEK ANSYS çözümleriyle yürütür.

    Kaba, orta ve ince ağ için model üç kez baştan çözülür; ASME V&V 20 / Roache
    GCI raporu bu üç gerçek çözümün sonuçlarından hesaplanır. Üç çözüm arka arkaya
    çalıştığı için işlem dakikalar sürebilir.

    Args:
        job_id: Statik yapısal olarak konfigüre edilmiş analiz kimliği.
        refinement_ratio: Mesh sıklaştırma oranı (Örn: 1.5).
        target_gci_pct: Hedef GCI bağımsızlık yüzdesi (Örn: 3.0%).
        show_gui: True ise her seviye Mechanical penceresinde izlenir.
    """
    return MeshStudyRunner.run_mesh_study(
        storage=storage,
        job_id=job_id,
        refinement_ratio=refinement_ratio,
        target_gci_pct=target_gci_pct,
        show_gui=show_gui
    )


@mcp.tool()
@exclusive_solver
def detect_stress_singularities(
    job_id: str, show_gui: bool = False, reuse_stored: bool = True
) -> Dict[str, Any]:
    """
    Gerilme tekilliğini (Stress Singularity) üç gerçek ağ seviyesini karşılaştırarak teşhis eder.

    Tekillikte gerilme ağ sıklaştıkça yakınsamaz, yer değiştirme ise yakınsar;
    teşhis bu iki eğilimin karşılaştırılmasına dayanır. Bu nedenle araç
    'run_mesh_convergence_study' ile aynı üç gerçek çözümü kullanır. Bu job için
    daha önce bir çalışma yapılmışsa kayıtlı sonuç yeniden kullanılır ve çözüm
    tekrarlanmaz.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise her seviye Mechanical penceresinde izlenir.
        reuse_stored: False ise kayıtlı çalışma yok sayılır ve üç çözüm yeniden koşulur.
    """
    study_res = MeshStudyRunner.load_stored(storage, job_id) if reuse_stored else None
    if study_res is None:
        study_res = MeshStudyRunner.run_mesh_study(
            storage=storage, job_id=job_id, show_gui=show_gui
        )
    if study_res.get("status") == "error":
        return study_res
    return {
        "status": "success",
        "job_id": job_id,
        "singularity_report": study_res["singularity_report"]
    }


@mcp.tool()
def configure_assembly_contacts(
    job_id: str,
    body_1: str = "Body_A",
    face_1: str = "Face_1",
    body_2: str = "Body_B",
    face_2: str = "Face_2",
    contact_type: str = "bonded",
    friction_coefficient: float = 0.20
) -> Dict[str, Any]:
    """
    Montaj analizlerinde parçalar arasındaki temas çiftlerini (Bonded, Frictional, Frictionless) yapılandırır.
    
    Args:
        job_id: Analiz kimliği.
        body_1: 1. Gövde adı.
        face_1: 1. Gövde temas yüzeyi.
        body_2: 2. Gövde adı.
        face_2: 2. Gövde temas yüzeyi.
        contact_type: 'bonded', 'frictional', 'frictionless', 'no_separation'
        friction_coefficient: Sürtünme katsayısı (frictional için, örn: 0.20).
    """
    try:
        c_type = ContactType(contact_type)
        pair = ContactManager.create_contact_pair(
            pair_id=f"Contact_{body_1}_{body_2}",
            body_1=body_1,
            face_1=face_1,
            body_2=body_2,
            face_2=face_2,
            contact_type=c_type,
            friction_coeff=friction_coefficient
        )
        script = ContactManager.generate_mechanical_contacts_script([pair])
        return {
            "status": "success",
            "job_id": job_id,
            "contact_pair": pair.model_dump(),
            "mechanical_script": script
        }
    except Exception as e:
        return {"status": "error", "message": f"Kontakt tanımlama hatası: {str(e)}"}


# ---------------------------------------------------------------------------
# 1.2 Faz 5 Aerodinamik & CFD Araçları (Ansys Fluent)
# ---------------------------------------------------------------------------

SOUND_SPEED_M_S = 340.3
MAX_CFD_MACH = 0.3
MAX_CFD_AOA_DEG = 10.0


@mcp.tool()
def configure_aerodynamics_cfd(
    job_id: str,
    cad_file_path: str,
    inlet_velocity_m_s: float,
    angle_of_attack_deg: float = 0.0,
    turbulence_model: str = "k-omega-sst",
    target_y_plus: float = 1.0,
    mesh_size_mm: float = 5.0,
    sideslip_deg: float = 0.0,
    flow_direction: Optional[List[float]] = None,
    lift_direction: Optional[List[float]] = None,
    reference_area_m2: Optional[float] = None,
    reference_length_m: Optional[float] = None,
    moment_center_mm: Optional[List[float]] = None,
    fluid_density_kg_m3: Optional[float] = None,
    dynamic_viscosity_pa_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    ANSYS Fluent dış aerodinamik akış analizini (Rüzgar Tüneli, y+ Sınır Tabaka, Türbülans Modeli) yapılandırır.

    Akış +X yönünde, taşıma +Y yönündedir: CAD modelini bu eksenlere göre yönlendirin ya da
    'flow_direction' ve 'lift_direction' vererek herhangi bir akış yönünü seçin (gövde buna göre çevrilir).
    Hücum açısı XY düzleminde uygulanır. Çözücü basınç tabanlı sıkıştırılamaz akıştır; bu
    yüzden Mach > 0.3 reddedilir. Hücum açısı en çok 10 derecedir (tünel yan duvarları simetridir).

    Args:
        job_id: create_analysis(analysis_type='cfd_steady') ile oluşturulan job kimliği.
        cad_file_path: Analiz edilecek gövdenin CAD dosya yolu.
        inlet_velocity_m_s: Giriş akış hızı (m/s), zorunlu.
        angle_of_attack_deg: Hücum açısı (derece), |değer| <= 10.
        sideslip_deg: Yan kayma açısı (derece), |değer| <= 10; akış +Z'ye doğru döner. Yan kuvvet katsayısı Cy raporlanır.
        flow_direction: Serbest akış yönü, CAD çerçevesinde [x, y, z] (STEP, montaj STEP veya STL/OBJ/PLY). Verilirse 'lift_direction' da zorunludur;
            gövde, akış +X ve taşıma +Y olacak biçimde tek bir dönmeyle çevrilir (B-rep bozulmaz, hacim ve kutu doğrulanır).
            Hücum açısı ve yan kayma bu yeni çerçeveye göre uygulanır. moment_center_mm CAD çerçevesinde verilir ve aynı dönmeyle çevrilir.
        lift_direction: CAD çerçevesinde taşıma yönü ipucu [x, y, z]; akıma dik bileşeni alınır, akıma paralel olamaz.
        turbulence_model: 'k-omega-sst', 'k-omega', 'k-epsilon', 'spalart-allmaras', 'laminar'.
        target_y_plus: Hedef y+ (0.1-300). İlk prizma katman yüksekliği buna göre verilir ve çözümden sonra gerçek y+ okunup raporlanır.
        mesh_size_mm: Gövde yüzey ağı için maksimum eleman boyu (mm).
        reference_area_m2: Katsayı referans alanı. Verilirse aynen kullanılır; boşsa gövdenin ön kesit (YZ kutusu) alanı alınır.
        reference_length_m: Cm ve Reynolds için referans uzunluk. Boşsa gövde X uzunluğu.
        moment_center_mm: [x, y, z] moment merkezi (mm). Boşsa gövde kutusunun merkezi. Cm, Z ekseni etrafındadır.
        fluid_density_kg_m3: Akışkan yoğunluğu. Boşsa deniz seviyesi havası (1.225).
        dynamic_viscosity_pa_s: Dinamik viskozite. Boşsa deniz seviyesi havası (1.7894e-5).
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if job["analysis_type"] != AnalysisType.CFD_STEADY.value:
        return {
            "status": "error",
            "message": "Bu job '%s' türünde; bu araç yalnızca 'cfd_steady' içindir." % job["analysis_type"],
        }

    try:
        velocity = require_positive("inlet_velocity_m_s", inlet_velocity_m_s)
        aoa = require_finite("angle_of_attack_deg", angle_of_attack_deg)
        beta = require_finite("sideslip_deg", sideslip_deg)
        mesh_size = require_positive("mesh_size_mm", mesh_size_mm)
        y_target = require_positive("target_y_plus", target_y_plus)
        area_user = require_positive("reference_area_m2", reference_area_m2) if reference_area_m2 is not None else None
        length_user = require_positive("reference_length_m", reference_length_m) if reference_length_m is not None else None
        density = require_positive("fluid_density_kg_m3", fluid_density_kg_m3) if fluid_density_kg_m3 is not None else None
        viscosity = require_positive("dynamic_viscosity_pa_s", dynamic_viscosity_pa_s) if dynamic_viscosity_pa_s is not None else None
        centre = None
        if moment_center_mm is not None:
            if not isinstance(moment_center_mm, (list, tuple)) or len(moment_center_mm) != 3:
                raise InputError("moment_center_mm [x, y, z] biçiminde 3 sayı olmalıdır.")
            centre = [require_finite("moment_center_mm", v) for v in moment_center_mm]
    except InputError as exc:
        return {"status": "error", "message": str(exc)}

    if velocity / SOUND_SPEED_M_S > MAX_CFD_MACH:
        return {
            "status": "error",
            "message": "Mach %.2f > %.1f: çözücü sıkıştırılamaz akış varsayar, bu hız desteklenmiyor."
            % (velocity / SOUND_SPEED_M_S, MAX_CFD_MACH),
        }
    if abs(beta) > MAX_CFD_AOA_DEG:
        return {
            "status": "error",
            "message": "Yan kayma açısı |%.2f| > %.0f derece: tünel yan duvarları simetri olduğundan desteklenmiyor."
            % (beta, MAX_CFD_AOA_DEG),
        }
    if abs(aoa) > MAX_CFD_AOA_DEG:
        return {
            "status": "error",
            "message": "Hücum açısı |%.2f| > %.0f derece: tünel yan duvarları simetri olduğundan desteklenmiyor."
            % (aoa, MAX_CFD_AOA_DEG),
        }
    if not (0.1 <= y_target <= 300.0):
        return {"status": "error", "message": "target_y_plus 0.1 ile 300 arasında olmalıdır."}

    if turbulence_model not in TURBULENCE_MODELS:
        return {
            "status": "error",
            "message": "Desteklenmeyen türbülans modeli: %s (geçerli: %s)"
                       % (turbulence_model, ", ".join(sorted(TURBULENCE_MODELS)))
        }

    analysis_cad = cad_file_path
    oriented = None
    if (flow_direction is None) != (lift_direction is None):
        return {"status": "error", "message": "flow_direction ve lift_direction birlikte verilmelidir."}
    if flow_direction is not None:
        from .geometry import step_rotate
        try:
            frame = step_rotate.wind_frame(flow_direction, lift_direction)
            if not step_rotate.is_identity(frame):
                target = os.path.join(storage.get_job_dir(job_id), 'oriented')
                stem = os.path.splitext(os.path.basename(cad_file_path))[0]
                faceted = os.path.splitext(cad_file_path)[1].lower() in ('.stl', '.obj', '.ply')
                analysis_cad = os.path.join(target, ''.join(ch if ch.isalnum() else '_' for ch in stem) + ('_wind.stl' if faceted else '_wind.step'))
                rotate = step_rotate.rotate_mesh_file if faceted else step_rotate.rotate_step
                counts = rotate(cad_file_path, analysis_cad, frame)
                check = step_rotate.verify_rotation(cad_file_path, analysis_cad, frame)
                oriented = {'rotation_rows': frame, 'rotated': counts, 'check': check}
                if centre is not None:
                    centre = [round(v, 6) for v in step_rotate.apply(frame, centre)]
            else:
                oriented = {'rotation_rows': frame, 'rotated': None, 'check': 'birim dönme, dosya aynen kullanıldı'}
        except step_rotate.StepRotateError as exc:
            return {"status": "error", "message": str(exc)}

    try:
        cad_summary = CadParser.parse_cad_file(analysis_cad)
    except Exception as exc:
        return {"status": "error", "message": f"CAD dosyası okunamadı: {exc}"}

    bbox = cad_summary.bounding_box
    enclosure_bounds = WindTunnelEnclosure.compute_domain_dimensions(bbox)

    # The flow runs along +X, so the projected frontal area is the YZ face of
    # the bounding box; it is also what the coefficients are normalised by.
    frontal_area_m2 = (
        (bbox["max_y"] - bbox["min_y"]) / 1000.0 * (bbox["max_z"] - bbox["min_z"]) / 1000.0
    )
    characteristic_length_m = (bbox["max_x"] - bbox["min_x"]) / 1000.0
    if frontal_area_m2 <= 0.0 or characteristic_length_m <= 0.0:
        return {
            "status": "error",
            "message": "CAD sınır kutusu dejenere; aerodinamik referans alan hesaplanamadı."
        }
    reference_length = length_user if length_user is not None else characteristic_length_m

    inflation_cfg = InflationEngine.calculate_inflation(
        target_y_plus=y_target,
        velocity_m_s=velocity,
        characteristic_length_m=reference_length,
        turbulence_model=turbulence_model
    )

    notes = [
        "Akış +X, taşıma +Y yönündedir; Cm Z ekseni etrafındadır.",
    ] + ([
        "Serbest akış yönü %s, taşıma %s: gövde bu çerçeveye döndürüldü (hacim ve sınır kutusu doğrulandı)."
        % (list(flow_direction), list(lift_direction)),
    ] if oriented and oriented.get("rotated") else []) + [
        "Tünel yan sınırları simetridir; engelleme etkisini küçük tutmak için gövde ön kesiti tünele göre küçük olmalıdır.",
    ]
    cfd_config = {
        "analysis_type": "cfd_steady",
        "inlet_velocity_m_s": velocity,
        "angle_of_attack_deg": aoa,
        "sideslip_deg": beta,
        "flow_direction": list(flow_direction) if flow_direction is not None else None,
        "lift_direction": list(lift_direction) if lift_direction is not None else None,
        "oriented_cad_path": analysis_cad if analysis_cad != cad_file_path else None,
        "wind_frame": oriented,
        "turbulence_model": turbulence_model,
        "target_y_plus": y_target,
        "mesh_size_mm": mesh_size,
        "reference_area_m2": area_user if area_user is not None else round(frontal_area_m2, 8),
        "user_reference_area_m2": area_user,
        "reference_area_source": "user" if area_user is not None else "cad_bounding_box",
        "characteristic_length_m": round(reference_length, 6),
        "user_reference_length_m": length_user,
        "moment_center_mm": centre,
        "fluid_density_kg_m3": density,
        "dynamic_viscosity_pa_s": viscosity,
        "enclosure_bounds": enclosure_bounds.model_dump(),
        "inflation_config": inflation_cfg.model_dump()
    }

    storage.save_job_config(job_id=job_id, cad_path=cad_file_path, config=cfd_config)

    return {
        "status": "configured",
        "job_id": job_id,
        "geometry": {
            "bounding_box_mm": bbox,
            "frontal_area_m2": round(frontal_area_m2, 8),
            "characteristic_length_m": round(characteristic_length_m, 6),
        },
        "reference_area_m2": cfd_config["reference_area_m2"],
        "reference_area_source": cfd_config["reference_area_source"],
        "enclosure": enclosure_bounds.model_dump(),
        "inflation": inflation_cfg.model_dump(),
        "notes": notes,
        "message": "CFD analizi başarıyla yapılandırıldı. 'solve_cfd_simulation' ile çözümü başlatabilirsiniz."
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_cfd_simulation(job_id: str, max_iterations: int = 250) -> Dict[str, Any]:
    """
    Yapılandırılmış Fluent CFD simülasyonunu ANSYS Fluent çözücüsünde gerçekten çalıştırır.

    Fluent CAD'i kendisi fasetler, rüzgar tüneli hacim ağını üretir, k-omega SST
    (veya seçilen model) ile iterasyonları koşturur ve Cd/Cl/kuvvet değerlerini
    çözücüden okur.

    Args:
        job_id: Analiz kimliği.
        max_iterations: Maksimum iterasyon sayısı.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "CFD analizi bulunamadı veya henüz yapılandırılmadı."}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    adapter = FluentAdapter()
    _ACTIVE_RUN["close"] = adapter.close

    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(cfg.get("oriented_cad_path") or job["cad_file_path"])
        adapter.configure_aerodynamics(
            inlet_velocity_m_s=cfg["inlet_velocity_m_s"],
            angle_of_attack_deg=cfg.get("angle_of_attack_deg", 0.0),
            turbulence_model=cfg.get("turbulence_model", "k-omega-sst"),
            target_y_plus=cfg.get("target_y_plus", 1.0),
            sideslip_deg=cfg.get("sideslip_deg", 0.0),
            characteristic_length_m=cfg.get("characteristic_length_m"),
            reference_area_m2=cfg.get("user_reference_area_m2"),
            reference_length_m=cfg.get("user_reference_length_m"),
            moment_center_mm=cfg.get("moment_center_mm"),
            fluid_density_kg_m3=cfg.get("fluid_density_kg_m3"),
            dynamic_viscosity_pa_s=cfg.get("dynamic_viscosity_pa_s"),
        )

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_config = MeshConfig(
            global_size=Quantity(value=cfg.get("mesh_size_mm", 5.0), unit="mm"),
            growth_rate=1.2,
            inflation_layers=cfg.get("inflation_config", {}).get("number_of_layers")
        )
        mesh_info = adapter.generate_mesh(mesh_config)
        # The solver normalises its coefficients by the faceted body's own
        # extents, which can differ from the CAD bounding box by the facet
        # tolerance. Store what Fluent used so later force reports match.
        cfg["reference_area_m2"] = mesh_info["reference_area_m2"]
        cfg["characteristic_length_m"] = mesh_info["reference_length_m"]
        storage.save_job_config(job_id=job_id, cad_path=job["cad_file_path"], config=cfg)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve(max_iterations=max_iterations)

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        results = adapter.extract_results()
        plots = CfdPostProcessingAdapter.render_cfd_plots(
            job_dir=job_dir,
            job_id=job_id,
            solver=adapter.solver_session,
            body_surfaces=adapter.body_wall_zones,
            inlet_surfaces=adapter.inlet_zones
        )

        cd_val = next(r.value for r in results if r.metric_name == "drag_coefficient_Cd")
        cl_val = next(r.value for r in results if r.metric_name == "lift_coefficient_Cl")
        aero_summary = CfdPostProcessingAdapter.calculate_aerodynamic_performance(
            drag_coeff_cd=cd_val,
            lift_coeff_cl=cl_val,
            velocity_m_s=cfg["inlet_velocity_m_s"],
            reference_area_m2=mesh_info["reference_area_m2"],
            fluid_density=adapter.density_kg_m3
        )

        air = Material(
            name="Air (incompressible, sea level)",
            youngs_modulus=Quantity(value=0.0, unit="Pa"),
            poissons_ratio=0.0,
            yield_strength=Quantity(value=0.0, unit="Pa"),
            density=Quantity(value=adapter.density_kg_m3, unit="kg/m^3")
        )
        manifest = AnalysisManifest(
            job_id=job_id,
            analysis_type=AnalysisType.CFD_STEADY,
            created_at=job["created_at"],
            cad_file_path=job["cad_file_path"],
            cad_sha256=GeometryIntelligence.compute_file_sha256(job["cad_file_path"]),
            material=air,
            mesh_config=mesh_config,
            boundary_conditions=[],
            results=results,
            artifacts=list(plots.values()) + [solve_res["case_path"]],
            status=JobState.COMPLETED
        )
        storage.save_manifest(job_id, manifest)
        storage.update_job_state(job_id, JobState.COMPLETED)
        # The solver's own convergence record, kept so the cfd-monitors resource
        # reports what Fluent actually returned rather than a stock answer.
        with open(os.path.join(job_dir, "cfd_solver_info.json"), "w", encoding="utf-8") as handle:
            json.dump(solve_res, handle, ensure_ascii=False, indent=2)

        return {
            "status": "completed",
            "job_id": job_id,
            "state": JobState.COMPLETED.value,
            "geometry": geometry,
            "mesh": mesh_info,
            "solver_info": solve_res,
            "aerodynamic_performance": aero_summary,
            "results": [r.model_dump() for r in results],
            "plots": plots,
            "message": "Fluent CFD simülasyonu tamamlandı. Katsayılar doğrudan çözücüden okundu."
        }
    except Exception as exc:
        _mark_failed(job_id, exc)
        return {"status": "error", "job_id": job_id, "message": f"Fluent CFD hatası: {exc}"}
    finally:
        adapter.close()


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_cfd_polar(
    job_id: str,
    angles_of_attack_deg: List[float],
    max_iterations_per_angle: int = 150,
) -> Dict[str, Any]:
    """
    Aynı ağ üzerinde birden fazla hücum açısı için Fluent çözümü yapar (polar taraması).

    İlk açı hibrit ilklendirmeyle başlar, sonrakiler önceki çözümden devam eder. Her nokta için
    Cd, Cl, Cm, L/D, sürükleme kuvveti ve kendi yakınsama durumu döner. Girdiler
    'configure_aerodynamics_cfd' ile kurulan işten gelir; işteki tek hücum açısı bu araçta
    kullanılmaz. Açıların her biri |değer| <= 10 derece olmalıdır.

    Args:
        job_id: 'configure_aerodynamics_cfd' ile yapılandırılmış cfd_steady işi.
        angles_of_attack_deg: Hücum açıları listesi (derece), en çok 12 nokta, tekrarsız.
        max_iterations_per_angle: Her açı için en fazla iterasyon.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "CFD analizi bulunamadı veya henüz yapılandırılmadı."}
    if job["analysis_type"] != AnalysisType.CFD_STEADY.value:
        return {"status": "error", "message": "Bu job '%s' türünde; bu araç yalnızca 'cfd_steady' içindir." % job["analysis_type"]}
    try:
        if not isinstance(angles_of_attack_deg, (list, tuple)) or not (1 <= len(angles_of_attack_deg) <= 12):
            raise InputError("angles_of_attack_deg 1 ile 12 arasında açı içeren bir liste olmalıdır.")
        angles = [require_finite("angles_of_attack_deg", a) for a in angles_of_attack_deg]
        if len(set(angles)) != len(angles):
            raise InputError("angles_of_attack_deg tekrarlayan açı içeremez.")
        if any(abs(a) > MAX_CFD_AOA_DEG for a in angles):
            raise InputError("Her hücum açısı |değer| <= %.0f derece olmalıdır." % MAX_CFD_AOA_DEG)
        iterations = int(require_positive("max_iterations_per_angle", max_iterations_per_angle))
    except InputError as exc:
        return {"status": "error", "message": str(exc)}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    adapter = FluentAdapter()
    _ACTIVE_RUN["close"] = adapter.close

    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(cfg.get("oriented_cad_path") or job["cad_file_path"])
        adapter.configure_aerodynamics(
            inlet_velocity_m_s=cfg["inlet_velocity_m_s"],
            angle_of_attack_deg=angles[0],
            turbulence_model=cfg.get("turbulence_model", "k-omega-sst"),
            target_y_plus=cfg.get("target_y_plus", 1.0),
            sideslip_deg=cfg.get("sideslip_deg", 0.0),
            characteristic_length_m=cfg.get("characteristic_length_m"),
            reference_area_m2=cfg.get("user_reference_area_m2"),
            reference_length_m=cfg.get("user_reference_length_m"),
            moment_center_mm=cfg.get("moment_center_mm"),
            fluid_density_kg_m3=cfg.get("fluid_density_kg_m3"),
            dynamic_viscosity_pa_s=cfg.get("dynamic_viscosity_pa_s"),
        )

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_config = MeshConfig(
            global_size=Quantity(value=cfg.get("mesh_size_mm", 5.0), unit="mm"),
            growth_rate=1.2,
            inflation_layers=cfg.get("inflation_config", {}).get("number_of_layers"),
        )
        mesh_info = adapter.generate_mesh(mesh_config)

        storage.update_job_state(job_id, JobState.SOLVING)
        points = adapter.solve_polar(angles, max_iterations=iterations)

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        y_plus = adapter._read_y_plus()
        polar_path = os.path.join(job_dir, "cfd_polar.json")
        with open(polar_path, "w", encoding="utf-8") as handle:
            json.dump({"points": points, "y_plus": y_plus, "mesh": mesh_info}, handle, ensure_ascii=False, indent=2)

        metrics = []
        for p in points:
            tag = ("%g" % p["angle_of_attack_deg"]).replace("-", "m").replace(".", "p")
            metrics.append(ResultMetric(metric_name="Cd_aoa_%s" % tag, value=round(p["cd"], 6), unit=""))
            metrics.append(ResultMetric(metric_name="Cl_aoa_%s" % tag, value=round(p["cl"], 6), unit=""))
            metrics.append(ResultMetric(metric_name="Cm_aoa_%s" % tag, value=round(p["cm"], 6), unit=""))
        metrics.append(ResultMetric(metric_name="wall_y_plus_average", value=round(y_plus["average"], 4), unit=""))
        metrics.append(ResultMetric(metric_name="wall_y_plus_maximum", value=round(y_plus["maximum"], 4), unit=""))

        air = Material(
            name="Air (incompressible, sea level)",
            youngs_modulus=Quantity(value=0.0, unit="Pa"),
            poissons_ratio=0.0,
            yield_strength=Quantity(value=0.0, unit="Pa"),
            density=Quantity(value=adapter.density_kg_m3, unit="kg/m^3"),
        )
        manifest = AnalysisManifest(
            job_id=job_id,
            analysis_type=AnalysisType.CFD_STEADY,
            created_at=job["created_at"],
            cad_file_path=job["cad_file_path"],
            cad_sha256=GeometryIntelligence.compute_file_sha256(job["cad_file_path"]),
            material=air,
            mesh_config=mesh_config,
            boundary_conditions=[],
            results=metrics,
            artifacts=[polar_path],
            status=JobState.COMPLETED,
        )
        storage.save_manifest(job_id, manifest)
        storage.update_job_state(job_id, JobState.COMPLETED)

        return {
            "status": "completed",
            "job_id": job_id,
            "state": JobState.COMPLETED.value,
            "geometry": geometry,
            "mesh": mesh_info,
            "reference_area_m2": mesh_info["reference_area_m2"],
            "polar": points,
            "wall_y_plus": y_plus,
            "all_points_converged": all(p["residuals_converged"] and p["force_stable"] for p in points),
            "polar_file": polar_path,
            "message": "Fluent polar taraması tamamlandı. Her noktanın yakınsama durumu 'polar' içinde.",
        }
    except Exception as exc:
        _mark_failed(job_id, exc)
        return {"status": "error", "job_id": job_id, "message": f"Fluent CFD hatası: {exc}"}
    finally:
        adapter.close()


@mcp.tool()
def get_cfd_aerodynamic_coefficients(job_id: str) -> Dict[str, Any]:
    """
    Tamamlanan CFD analizinin aerodinamik sürükleme (Cd), taşıma (Cl) ve L/D verilerini döner.

    Args:
        job_id: Analiz kimliği.
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış analiz bulunamadı."}

    manifest = json.loads(job["manifest_json"])
    results = [ResultMetric(**r) for r in manifest["results"]]
    by_name = {r.metric_name: r.value for r in results}
    if "drag_coefficient_Cd" not in by_name or "lift_coefficient_Cl" not in by_name:
        return {
            "status": "error",
            "message": "Bu job bir CFD analizi değil ya da katsayılar kaydedilmemiş."
        }

    cfg = json.loads(job["config_json"]) if job["config_json"] else {}
    density = (manifest.get("material") or {}).get("density") or {}
    # Coefficients are only forces once you know the area and speed they were
    # normalised by; a stand-in value here would scale every force silently.
    missing = [key for key in ("reference_area_m2", "inlet_velocity_m_s") if not cfg.get(key)]
    if missing or not density.get("value"):
        return {
            "status": "error",
            "message": (
                "Kuvvet raporu için referans verisi eksik (%s). "
                "Analizi configure_aerodynamics_cfd ile yeniden kurun."
                % ", ".join(missing + ([] if density.get("value") else ["akışkan yoğunluğu"]))
            )
        }

    perf = CfdPostProcessingAdapter.calculate_aerodynamic_performance(
        drag_coeff_cd=by_name["drag_coefficient_Cd"],
        lift_coeff_cl=by_name["lift_coefficient_Cl"],
        velocity_m_s=float(cfg["inlet_velocity_m_s"]),
        reference_area_m2=float(cfg["reference_area_m2"]),
        fluid_density=float(density["value"])
    )
    return {
        "status": "success",
        "job_id": job_id,
        "solver_metrics": {r.metric_name: r.value for r in results},
        "aerodynamic_performance": perf
    }


# ---------------------------------------------------------------------------
# 1.3 Faz 6 Güvenlik & Provenance Araçları
# ---------------------------------------------------------------------------

@mcp.tool()
def execute_safe_python_script(
    job_id: str,
    script_code: str,
    security_level: int = 2,
    confirmation_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    AI tarafından üretilen ileri düzey PyAnsys/Matematik scriptini AST güvenlik
    filtresi ve izole sanal ortamda (Sandbox) çalıştırır.
    
    Args:
        job_id: Analiz kimliği.
        script_code: Çalıştırılacak Python script kodu.
        security_level: 1 (Yasaklı), 2 (AST Sandbox, Varsayılan), 3 (Kullanıcı onaylı Uzman Modu).
        confirmation_token: Level 3 için onay belirteci ('CONFIRM_EXPERT_EXECUTION').
    """
    job = storage.get_job(job_id)
    job_dir = storage.get_job_dir(job_id) if job else RUNS_DIR + "/sandbox"

    return SandboxEngine.execute_sandboxed_code(
        script_code=script_code,
        job_dir=job_dir,
        security_level=security_level,
        confirmation_token=confirmation_token
    )


@mcp.tool()
def verify_analysis_provenance(job_id: str) -> Dict[str, Any]:
    """
    Tamamlanan analizin SHA-256 dijital imzasını doğrulayarak sonuçların
    simülasyondan sonra değiştirilip değiştirilmediğini (Tamper Detection) denetler.
    
    Args:
        job_id: Analiz kimliği.
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Analiz manifestosu bulunamadı."}

    manifest = json.loads(job["manifest_json"])
    cfg = json.loads(job["config_json"]) if job["config_json"] else {}
    results = manifest.get("results", [])

    # The record was written when the result was stored. A job that finished
    # before records were kept has nothing to verify against, and saying so is
    # the only honest answer: a record rebuilt now would match by construction.
    stored = storage.load_provenance(job_id)
    if stored is None:
        return {
            "status": "error",
            "message": "Bu job için sonuç kaydedilirken yazılmış bir provenance kaydı yok; "
            "doğrulanacak bir referans bulunmuyor.",
        }

    return ProvenanceEngine.verify_against_record(
        record=stored,
        cad_file_path=job.get("cad_file_path") or "",
        current_config_dict=cfg,
        current_results_list=results,
    )


@mcp.tool()
def get_client_configurations() -> Dict[str, Any]:
    """
    Claude Code, Antigravity ve Cursor IDE için doğrudan kopyalanabilir
    JSON MCP sunucu bağlantı ayarlarını döner.
    """
    return ClientConfigGenerator.generate_all_configs()


# ---------------------------------------------------------------------------
# 1.4 Faz 7 Parametrik Tasarım Keşfi & Optimizasyon Araçları
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_optimization_design_points(
    study_id: str,
    parameters: List[Dict[str, Any]],
    num_samples: int = 8,
    sampling_method: str = "lhs"
) -> Dict[str, Any]:
    """
    Parametrik tasarım keşfi için değerlendirilecek tasarım noktalarını üretir.

    Üretilen her nokta gerçek bir analizle çözülmeli ve sonucu
    'submit_design_point_results' ile kaydedilmelidir. Çalışma diske yazılır.

    Args:
        study_id: Optimizasyon çalışma kimliği (Örn: 'OPT-BRACKET-01').
        parameters: Parametre listesi [name, min_value, max_value, default_value, step_size].
        num_samples: Örnekleme nokta sayısı (DoE).
        sampling_method: 'lhs' (Latin Hypercube) veya 'grid' (Full Factorial).
    """
    try:
        p_models = [DesignParameter(**p) for p in parameters]
        state = OptimizationBatchRunner.create_study(
            storage=storage,
            study_id=study_id,
            parameters=p_models,
            num_samples=num_samples,
            sampling_method=sampling_method
        )
        return {
            "status": "created",
            "study_id": study_id,
            "sampling_method": sampling_method,
            "design_points": state["points"],
            "message": (
                "%d tasarım noktası üretildi. Her nokta için analizi çalıştırıp "
                "sonuçları 'submit_design_point_results' ile kaydedin."
                % len(state["points"])
            )
        }
    except Exception as e:
        return {"status": "error", "message": f"Tasarım noktaları üretilemedi: {str(e)}"}


@mcp.tool()
def submit_design_point_results(
    study_id: str,
    point_id: int,
    results: Dict[str, float]
) -> Dict[str, Any]:
    """
    Bir tasarım noktası için gerçek analizden elde edilen tepki değerlerini kaydeder.

    Args:
        study_id: Çalışma kimliği.
        point_id: 'generate_optimization_design_points' çıktısındaki nokta kimliği.
        results: Ölçülen tepkiler, örn. {'mass_kg': 1.84, 'max_von_mises_stress': 142.6}.
    """
    try:
        entry = OptimizationBatchRunner.record_result(
            storage=storage, study_id=study_id, point_id=point_id, results=results
        )
        return {"status": "recorded", "study_id": study_id, "design_point": entry}
    except Exception as e:
        return {"status": "error", "message": f"Sonuç kaydedilemedi: {str(e)}"}


@mcp.tool()
def evaluate_optimization_study(
    study_id: str,
    objectives: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Kaydedilmiş tasarım noktaları üzerinde Pareto cephesini hesaplar ve grafiğini üretir.

    Args:
        study_id: Çalışma kimliği.
        objectives: Amaç fonksiyonları [{'name': 'mass_kg', 'goal': 'minimize', 'weight': 1.0}].
        constraints: Kısıtlar [{'name': 'max_von_mises_stress', 'operator': '<=', 'limit_value': 180.0}].
    """
    try:
        if not objectives or len(objectives) < 2:
            return {
                "status": "error",
                "message": "Pareto analizi için en az iki amaç fonksiyonu belirtilmelidir."
            }
        obj_models = [OptimizationObjective(**o) for o in objectives]
        con_models = [OptimizationConstraint(**c) for c in constraints] if constraints else []
        return OptimizationBatchRunner.evaluate_study(
            storage=storage,
            study_id=study_id,
            objectives=obj_models,
            constraints=con_models
        )
    except Exception as e:
        return {"status": "error", "message": f"Optimizasyon değerlendirmesinde hata: {str(e)}"}


@mcp.tool()
def get_pareto_optimal_designs(study_id: str) -> Dict[str, Any]:
    """
    Tamamlanmış parametrik optimizasyon çalışmasının Pareto-Optimal (baskın)
    tasarım noktalarını ve en ideal uzlaşma (best trade-off) tasarımını döner.

    Args:
        study_id: Çalışma kimliği.
    """
    try:
        return OptimizationBatchRunner.get_report(storage=storage, study_id=study_id)
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# 1.5 Faz 8 Modal & Dinamik Titreşim Analizi Araçları
# ---------------------------------------------------------------------------

@mcp.tool()
def configure_modal_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str = "Structural Steel",
    youngs_modulus_gpa: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    num_modes: int = 6,
    mesh_global_size_mm: float = 5.0,
    fixed_support_normal: Optional[List[float]] = None,
    fixed_support_axis_extreme: Optional[str] = None,
    yield_strength_mpa: Optional[float] = None,
    orthotropic: Optional[Dict[str, float]] = None,
    layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    ANSYS Mechanical serbest titreşim (Modal) analizini yapılandırır.

    Modal analiz bir özdeğer problemidir: kuvvet, basınç veya moment tanımlanmaz,
    yalnızca mesnet tanımlanır. Mesnet yüzeyi normaliyle ya da eksen ekstremumuyla
    ('min_x', 'max_z' gibi) seçilir.

    Args:
        job_id: Analiz kimliği.
        cad_file_path: Katı CAD geometri yolu (STEP / IGES / Parasolid / SAT).
        material_name: Malzeme adı.
        youngs_modulus_gpa: Elastisite Modülü (GPa).
        poissons_ratio: Poisson oranı.
        density_kg_m3: Malzeme yoğunluğu (kg/m³ - modal analizde kütle için zorunludur).
        num_modes: Hesaplanacak doğal frekans mod sayısı (Örn: 6).
        mesh_global_size_mm: Eleman boyutu (mm).
        fixed_support_normal: Ankastre mesnet yüzeyinin normali [nx, ny, nz].
        fixed_support_axis_extreme: Alternatif mesnet seçimi, örn. 'min_x'.
        yield_strength_mpa, orthotropic, layers: Yalnızca ortotropik/katmanlı modelde kullanılır; bu durumda
            'configure_static_analysis' ile aynı katı kurallar geçerlidir (hiçbir malzeme değeri varsayılmaz).
            'layers' için yüzey (kabuk) gövde gerekir. Bunlar verilmezse eski davranış sürer: boş bırakılan
            E, poisson ve yoğunluk için 200 GPa / 0.30 / 7850 kg/m3 çelik değerleri kullanılır.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}
    if layers is not None or orthotropic is not None:
        if job["analysis_type"] != AnalysisType.MODAL.value:
            return {"status": "error", "message": "Bu job '%s' türünde; bu araç yalnızca 'modal' içindir." % job["analysis_type"]}
        if isinstance(num_modes, bool) or not isinstance(num_modes, int) or not (1 <= num_modes <= 50):
            return {"status": "error", "message": "num_modes 1 ile 50 arasında bir tam sayı olmalıdır."}
        return _configure_structural(
            AnalysisType.MODAL.value, job_id=job_id, cad_file_path=cad_file_path, material_name=material_name,
            youngs_modulus_gpa=youngs_modulus_gpa, poissons_ratio=poissons_ratio,
            yield_strength_mpa=yield_strength_mpa, density_kg_m3=density_kg_m3,
            mesh_global_size_mm=mesh_global_size_mm, fixed_support_normal=fixed_support_normal,
            fixed_support_axis_extreme=fixed_support_axis_extreme, orthotropic=orthotropic,
            num_modes=num_modes, require_load=False, layers=layers,
        )
    youngs_modulus_gpa = 200.0 if youngs_modulus_gpa is None else youngs_modulus_gpa
    poissons_ratio = 0.30 if poissons_ratio is None else poissons_ratio
    density_kg_m3 = 7850.0 if density_kg_m3 is None else density_kg_m3

    mat = Material(
        name=material_name,
        youngs_modulus=Quantity(value=youngs_modulus_gpa, unit="GPa"),
        poissons_ratio=poissons_ratio,
        density=Quantity(value=density_kg_m3, unit="kg/m^3"),
        yield_strength=Quantity(value=250.0, unit="MPa"),
    )
    mesh_cfg = MeshConfig(global_size=Quantity(value=mesh_global_size_mm, unit="mm"))

    support_filter = _extreme_filter(fixed_support_axis_extreme)
    if not fixed_support_normal and not support_filter:
        fixed_support_normal = [-1.0, 0.0, 0.0]

    bcs = [
        BoundaryCondition(
            name="Ankastre_Mesnet",
            type="fixed_support",
            target_selector=SemanticSelector(
                name="Mesnet_Yuzeyi",
                normal_vector=fixed_support_normal,
                position_filter=support_filter,
                confidence_threshold=0.85,
            ),
        )
    ]

    modal_config = {
        "analysis_type": "modal",
        "material": mat.model_dump(),
        "mesh_config": mesh_cfg.model_dump(),
        "boundary_conditions": [bc.model_dump() for bc in bcs],
        "num_modes": num_modes,
    }

    storage.save_job_config(job_id=job_id, cad_path=cad_file_path, config=modal_config)

    return {
        "status": "configured",
        "job_id": job_id,
        "num_modes": num_modes,
        "material": material_name,
        "boundary_conditions": [bc.model_dump() for bc in bcs],
        "message": "Modal analiz yapılandırıldı. 'solve_modal_analysis' ile gerçek ANSYS çözümünü başlatabilirsiniz.",
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_modal_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Modal analizi GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Geometri gerçek ağaca yüklenir, gerçek mesh üretilir, MAPDL özdeğer çözücüsü
    çalışır; doğal frekanslar çözülmüş modelden, etkin kütle oranları ise
    çözücünün kendi 'solve.out' katılım faktörü tablosundan okunur.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır ve adımlar izlenir.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Modal analiz bulunamadı veya konfigüre edilmedi."}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    n_modes = cfg.get("num_modes", 6)
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg.get("boundary_conditions", [])]

    adapter = MechanicalAdapter(visible=show_gui)

    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)
        layered_report = adapter.apply_layered_section(cfg["layers"]) if cfg.get("layers") else None

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        modal_setup = adapter.setup_modal(bcs, modes=n_modes)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve_modal()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        modal_data = adapter.extract_modal()
        mode_plots = adapter.export_modal_images()
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()

    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS modal çözümü başarısız oldu. Sentetik frekans üretilmedi.",
        }

    modal_summary = ModalEngine.summarize_solved_modes(
        frequencies=modal_data["frequencies"],
        participation=modal_data.get("participation"),
    )

    metrics = [
        ResultMetric(
            metric_name=f"natural_frequency_mode_{m.mode_number}",
            value=m.frequency_hz,
            unit="Hz",
            note=f"{m.mode_type} (T={m.period_sec} s)",
        )
        for m in modal_summary.modes
    ]

    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])

    solver_report = {
        "solver": solve_res["solver"],
        "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"],
        "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"],
        "solver_messages": solve_res["messages"],
        "mechdb": mechdb,
        "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "material": mat_report,
        "mesh": mesh_report,
        "boundary_conditions": modal_setup["boundary_conditions"],
        "participation_factors": modal_data.get("participation"),
        "executed_transcript": transcript,
    }

    manifest = AnalysisManifest(
        job_id=job_id,
        analysis_type=AnalysisType.MODAL,
        created_at=job["created_at"],
        cad_file_path=job["cad_file_path"],
        cad_sha256=cad_sha256,
        material=material,
        mesh_config=mesh_cfg,
        boundary_conditions=bcs,
        results=metrics,
        artifacts=sorted(mode_plots.values()) + [transcript, mechdb]
        + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED,
        solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    return {
        "status": "completed",
        "job_id": job_id,
        "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"],
        "solve_time_s": solve_res["solve_time_s"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "fundamental_frequency_hz": modal_summary.fundamental_frequency_hz,
        "modes": [m.model_dump() for m in modal_summary.modes],
        "cumulative_mass_participation": modal_summary.cumulative_mass_participation,
        "assessment": modal_summary.dynamic_stiffness_assessment,
        "mode_plots": mode_plots,
        "mechdb": mechdb,
        "message": "Gerçek ANSYS modal çözümü tamamlandı. Rezonans denetimi için 'evaluate_resonance_risk' kullanabilirsiniz.",
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_buckling_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Doğrusal özdeğer burkulma analizini GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Önce referans yükle ön gerilmeli statik çözüm yapılır, sonra bağlı burkulma sistemi
    çözülür. Yük çarpanları çözülmüş modelden okunur; kritik yük = çarpan x referans yük.
    'configure_buckling_analysis' ve 'validate_analysis' önceden çalıştırılmış olmalıdır.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Burkulma analizi bulunamadı veya konfigüre edilmedi."}
    if job["analysis_type"] != AnalysisType.BUCKLING.value:
        return {"status": "error", "message": "Bu job '%s' türünde; bu araç yalnızca 'buckling' içindir." % job["analysis_type"]}
    if not job["validation_json"]:
        return {"status": "error", "message": "Analiz henüz doğrulanmadı! Önce 'validate_analysis' çalıştırılmalıdır."}
    v_report = json.loads(job["validation_json"])
    if not v_report.get("valid", False):
        return {"status": "error", "message": f"Doğrulamadan geçemeyen analiz çözülemez! Hatalar: {v_report.get('errors')}"}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    n_modes = cfg.get("num_modes")
    if n_modes is None:
        return {"status": "error", "message": "Konfigürasyonda num_modes yok; 'configure_buckling_analysis' çalıştırın."}
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg.get("boundary_conditions", [])]

    adapter = MechanicalAdapter(visible=show_gui)

    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)
        layered_report = adapter.apply_layered_section(cfg["layers"]) if cfg.get("layers") else None

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        setup = adapter.setup_buckling(bcs, modes=n_modes)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve_buckling()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        buckling = adapter.extract_buckling()
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()

    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS burkulma çözümü başarısız oldu. Sentetik yük çarpanı üretilmedi.",
        }

    metrics = []
    for m in buckling["modes"]:
        metrics.append(ResultMetric(
            metric_name=f"buckling_load_factor_mode_{m['mode']}",
            value=m["load_multiplier"], unit="-",
            note="Kritik yük / referans yük",
        ))
        if m["critical_load_n"] is not None:
            metrics.append(ResultMetric(
                metric_name=f"critical_load_mode_{m['mode']}",
                value=m["critical_load_n"], unit="N",
                note="Çarpan x referans yük (%.6g N)" % buckling["reference_load_n"],
            ))

    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])
    solver_report = {
        "solver": solve_res["solver"],
        "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"],
        "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"],
        "solver_messages": solve_res["messages"],
        "mechdb": mechdb,
        "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "material": mat_report,
        "mesh": mesh_report,
        "boundary_conditions": setup["boundary_conditions"],
        "buckling": buckling,
        "executed_transcript": transcript,
    }

    manifest = AnalysisManifest(
        job_id=job_id,
        analysis_type=AnalysisType.BUCKLING,
        created_at=job["created_at"],
        cad_file_path=job["cad_file_path"],
        cad_sha256=cad_sha256,
        material=material,
        mesh_config=mesh_cfg,
        boundary_conditions=bcs,
        results=metrics,
        artifacts=[transcript, mechdb] + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED,
        solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    first = buckling["modes"][0]
    return {
        "status": "completed",
        "job_id": job_id,
        "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"],
        "solve_time_s": solve_res["solve_time_s"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "reference_load_n": buckling["reference_load_n"],
        "first_load_factor": first["load_multiplier"],
        "first_critical_load_n": first["critical_load_n"],
        "modes": buckling["modes"],
        "mechdb": mechdb,
        "message": "Gerçek ANSYS burkulma çözümü tamamlandı. Kritik yük = yük çarpanı x referans yük. "
        "Doğrusal burkulma ideal geometri varsayar; kusur ve plastisite için güvenlik katsayısı uygulayın.",
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_harmonic_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Harmonik frekans taramasını GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Her taranan frekanstaki en büyük yer değiştirme genliği çözülmüş modelden okunur.
    'configure_harmonic_analysis' ve 'validate_analysis' önceden çalıştırılmış olmalıdır.
    Tepe genlik yalnızca taranan frekanslar arasından seçilir; dar bir rezonans tepesi
    iki adım arasında kalabilir, bu yüzden adım aralığı tepe yarı-güç bant genişliğiyle
    karşılaştırılıp uyarı verilir.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Harmonik analiz bulunamadı veya konfigüre edilmedi."}
    if job["analysis_type"] != AnalysisType.HARMONIC.value:
        return {"status": "error", "message": "Bu job '%s' türünde; bu araç yalnızca 'harmonic' içindir." % job["analysis_type"]}
    if not job["validation_json"]:
        return {"status": "error", "message": "Analiz henüz doğrulanmadı! Önce 'validate_analysis' çalıştırılmalıdır."}
    v_report = json.loads(job["validation_json"])
    if not v_report.get("valid", False):
        return {"status": "error", "message": f"Doğrulamadan geçemeyen analiz çözülemez! Hatalar: {v_report.get('errors')}"}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    needed = ("frequency_min_hz", "frequency_max_hz", "intervals", "damping_ratio", "method")
    if any(k not in cfg for k in needed):
        return {"status": "error", "message": "Konfigürasyonda harmonik parametreler yok; 'configure_harmonic_analysis' çalıştırın."}
    f_min, f_max, intervals = cfg["frequency_min_hz"], cfg["frequency_max_hz"], cfg["intervals"]
    zeta, method = cfg["damping_ratio"], cfg["method"]
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg.get("boundary_conditions", [])]
    step = (f_max - f_min) / intervals
    frequencies = [f_min + step * k for k in range(1, intervals + 1)]

    adapter = MechanicalAdapter(visible=show_gui)
    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)
        layered_report = adapter.apply_layered_section(cfg["layers"]) if cfg.get("layers") else None

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        setup = adapter.setup_harmonic(bcs, f_min, f_max, intervals, zeta, method=method)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve_harmonic()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        sweep = adapter.session.extract_harmonic_sweep(frequencies)
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()
    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS harmonik çözümü başarısız oldu. Sentetik genlik üretilmedi.",
        }

    curve = [
        {"frequency_hz": row["reported_frequency_hz"], "amplitude_mm": row["max_amplitude_m"] * 1.0e3}
        for row in sweep
    ]
    peak = max(curve, key=lambda r: r["amplitude_mm"])
    warnings = []
    if any(abs(row["frequency_hz"] - f) > 1e-6 * max(1.0, f_max) for row, f in zip(curve, frequencies)):
        warnings.append("Çözülen frekanslar istenenlerle eşleşmiyor; tepe değerini dikkatle yorumlayın.")
    half_power_bandwidth = 2.0 * zeta * peak["frequency_hz"]
    if step > half_power_bandwidth:
        warnings.append(
            "Tarama adımı (%.3g Hz), tepe frekansındaki yarı-güç bant genişliğinden (%.3g Hz) büyük; "
            "gerçek rezonans tepesi adımlar arasında kalmış olabilir. 'intervals' artırın."
            % (step, half_power_bandwidth)
        )
    if peak is curve[-1] and len(curve) > 1:
        warnings.append("Tepe genlik aralığın son frekansında; rezonans aralığın dışında olabilir, 'frequency_max_hz' artırın.")

    metrics = [
        ResultMetric(metric_name="peak_displacement_amplitude", value=round(peak["amplitude_mm"], 6), unit="mm",
                     note="Taranan frekanslar arasındaki en büyük genlik"),
        ResultMetric(metric_name="peak_frequency", value=round(peak["frequency_hz"], 6), unit="Hz"),
    ]
    for row in curve:
        metrics.append(ResultMetric(
            metric_name="displacement_amplitude_at_%gHz" % row["frequency_hz"],
            value=round(row["amplitude_mm"], 6), unit="mm",
        ))

    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])
    solver_report = {
        "solver": solve_res["solver"], "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"], "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"], "solver_messages": solve_res["messages"],
        "mechdb": mechdb, "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "material": mat_report, "mesh": mesh_report,
        "boundary_conditions": setup["boundary_conditions"],
        "harmonic": {"method": method, "damping_ratio": zeta, "frequency_min_hz": f_min,
                     "frequency_max_hz": f_max, "intervals": intervals, "curve": curve,
                     "warnings": warnings},
        "executed_transcript": transcript,
    }
    manifest = AnalysisManifest(
        job_id=job_id, analysis_type=AnalysisType.HARMONIC, created_at=job["created_at"],
        cad_file_path=job["cad_file_path"], cad_sha256=cad_sha256, material=material,
        mesh_config=mesh_cfg, boundary_conditions=bcs, results=metrics,
        artifacts=[transcript, mechdb] + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED, solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    return {
        "status": "completed", "job_id": job_id, "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"], "solve_time_s": solve_res["solve_time_s"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "method": method, "damping_ratio": zeta,
        "peak": peak, "curve": curve, "warnings": warnings, "mechdb": mechdb,
        "message": "Gerçek ANSYS harmonik çözümü tamamlandı. Genlikler yük genliğine (sıfır faz) göre en büyük yer değiştirmedir.",
    }


TRANSIENT_MAX_SAMPLES = 40


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_transient_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Geçici yapısal analizi GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Zamanın her örnek anındaki en büyük yer değiştirme çözücünün sonuç dosyasından (.rst) DPF ile
    okunur (en fazla 40 örnek, eşit aralıklı çözülmüş adımlar). 'configure_transient_analysis' ve 'validate_analysis'
    önceden çalıştırılmış olmalıdır. Yük t=0'da basamak olarak uygulanır; sönümsüz/az sönümlü
    sistemde tepe yer değiştirme statik değerin yaklaşık iki katına çıkar (dinamik büyütme).

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Geçici analiz bulunamadı veya konfigüre edilmedi."}
    if job["analysis_type"] != AnalysisType.TRANSIENT.value:
        return {"status": "error", "message": "Bu job '%s' türünde; bu araç yalnızca 'transient' içindir." % job["analysis_type"]}
    if not job["validation_json"]:
        return {"status": "error", "message": "Analiz henüz doğrulanmadı! Önce 'validate_analysis' çalıştırılmalıdır."}
    v_report = json.loads(job["validation_json"])
    if not v_report.get("valid", False):
        return {"status": "error", "message": f"Doğrulamadan geçemeyen analiz çözülemez! Hatalar: {v_report.get('errors')}"}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    needed = ("end_time_s", "time_step_s", "damping_alpha_per_s", "damping_beta_s")
    if any(k not in cfg for k in needed):
        return {"status": "error", "message": "Konfigürasyonda geçici analiz parametreleri yok; 'configure_transient_analysis' çalıştırın."}
    t_end, dt = cfg["end_time_s"], cfg["time_step_s"]
    alpha, beta = cfg["damping_alpha_per_s"], cfg["damping_beta_s"]
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg.get("boundary_conditions", [])]
    steps = int(round(t_end / dt))

    adapter = MechanicalAdapter(visible=show_gui)
    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)
        layered_report = adapter.apply_layered_section(cfg["layers"]) if cfg.get("layers") else None

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        setup = adapter.setup_transient(bcs, t_end, dt, alpha, beta)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve_transient()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        history = read_displacement_history(solve_res.get("rst_file"), TRANSIENT_MAX_SAMPLES)
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()
    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS geçici çözümü başarısız oldu. Sentetik geçmiş üretilmedi.",
        }

    curve = [
        {"time_s": row["reported_time_s"], "max_displacement_mm": row["max_displacement_m"] * 1.0e3}
        for row in history
    ]
    peak = max(curve, key=lambda r: r["max_displacement_mm"])
    warnings = []
    if abs(curve[-1]["time_s"] - t_end) > 1e-6 * max(1.0, t_end):
        warnings.append("Son okunan zaman (%.6g s) end_time_s ile eşleşmiyor." % curve[-1]["time_s"])
    if peak is curve[-1] and len(curve) > 1:
        warnings.append("Tepe yer değiştirme sürenin son örneğinde; tepe süre dışında olabilir, 'end_time_s' artırın.")

    metrics = [
        ResultMetric(metric_name="peak_displacement", value=round(peak["max_displacement_mm"], 6), unit="mm",
                     note="Örneklenen zamanlar arasındaki en büyük değer"),
        ResultMetric(metric_name="time_of_peak", value=round(peak["time_s"], 8), unit="s"),
    ]
    for row in curve:
        metrics.append(ResultMetric(
            metric_name="max_displacement_at_%gs" % row["time_s"],
            value=round(row["max_displacement_mm"], 6), unit="mm",
        ))

    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])
    solver_report = {
        "solver": solve_res["solver"], "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"], "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"], "solver_messages": solve_res["messages"],
        "mechdb": mechdb, "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "material": mat_report, "mesh": mesh_report,
        "boundary_conditions": setup["boundary_conditions"],
        "transient": {"end_time_s": t_end, "time_step_s": dt, "damping_alpha_per_s": alpha,
                      "damping_beta_s": beta, "curve": curve, "warnings": warnings},
        "executed_transcript": transcript,
    }
    manifest = AnalysisManifest(
        job_id=job_id, analysis_type=AnalysisType.TRANSIENT, created_at=job["created_at"],
        cad_file_path=job["cad_file_path"], cad_sha256=cad_sha256, material=material,
        mesh_config=mesh_cfg, boundary_conditions=bcs, results=metrics,
        artifacts=[transcript, mechdb] + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED, solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    return {
        "status": "completed", "job_id": job_id, "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"], "solve_time_s": solve_res["solve_time_s"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "peak": peak, "curve": curve, "warnings": warnings, "mechdb": mechdb,
        "message": "Gerçek ANSYS geçici çözümü tamamlandı. Değerler her örnek andaki modeldeki en büyük toplam yer değiştirmedir.",
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_random_vibration_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Rastgele titreşim (PSD) analizini GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Önce modal sistem, sonra ona bağlı rastgele titreşim sistemi çözülür. Dönen yer değiştirme ve
    von Mises değerleri 1 sigma (RMS) değerleridir; Gauss varsayımıyla yaklaşık yüzde 68.27
    olasılıkla aşılmazlar, 3 sigma (yüzde 99.73) değeri 1 sigma değerinin üç katıdır.
    'configure_random_vibration_analysis' ve 'validate_analysis' önceden çalıştırılmış olmalıdır.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Rastgele titreşim analizi bulunamadı veya konfigüre edilmedi."}
    if job["analysis_type"] != AnalysisType.RANDOM_VIBRATION.value:
        return {"status": "error", "message": "Bu job '%s' türünde; bu araç yalnızca 'random_vibration' içindir." % job["analysis_type"]}
    if not job["validation_json"]:
        return {"status": "error", "message": "Analiz henüz doğrulanmadı! Önce 'validate_analysis' çalıştırılmalıdır."}
    v_report = json.loads(job["validation_json"])
    if not v_report.get("valid", False):
        return {"status": "error", "message": f"Doğrulamadan geçemeyen analiz çözülemez! Hatalar: {v_report.get('errors')}"}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    needed = ("base_axis", "psd_frequencies_hz", "psd_values_g2_per_hz", "damping_ratio", "num_modes")
    if any(k not in cfg for k in needed):
        return {"status": "error", "message": "Konfigürasyonda PSD parametreleri yok; 'configure_random_vibration_analysis' çalıştırın."}
    axis, zeta, modes = cfg["base_axis"], cfg["damping_ratio"], cfg["num_modes"]
    freqs = cfg["psd_frequencies_hz"]
    asd_si = [v * STANDARD_GRAVITY_M_S2 ** 2 for v in cfg["psd_values_g2_per_hz"]]
    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg.get("boundary_conditions", [])]

    adapter = MechanicalAdapter(visible=show_gui)
    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)
        layered_report = adapter.apply_layered_section(cfg["layers"]) if cfg.get("layers") else None

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        setup = adapter.setup_random_vibration(bcs, axis, freqs, asd_si, zeta, modes)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve_random_vibration()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()
    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS rastgele titreşim çözümü başarısız oldu. Sentetik sonuç üretilmedi.",
        }

    def _to_si(entry: Dict[str, Any], kind: str) -> float:
        scale = {"m": 1.0, "mm": 1.0e-3, "pa": 1.0, "kpa": 1.0e3, "mpa": 1.0e6, "gpa": 1.0e9}
        unit = str(entry["unit"]).strip().lower()
        if unit not in scale:
            raise ValueError("Beklenmeyen %s birimi: %r" % (kind, entry["unit"]))
        return entry["value"] * scale[unit]

    res_si = solve_res["results_si"]
    try:
        disp_1s_mm = _to_si(res_si["DirectionalDeformation"], "yer değiştirme") * 1.0e3
        vm_1s_mpa = _to_si(res_si["EquivalentStressPSD"], "gerilme") / 1.0e6
    except (KeyError, ValueError) as exc:
        _mark_failed(job_id, exc)
        return {"status": "failed", "job_id": job_id, "error": str(exc),
                "message": "Rastgele titreşim sonuçları okunamadı. Sentetik sonuç üretilmedi."}

    mode_freqs = [m.get("frequency_hz") for m in solve_res.get("modal_frequencies", []) if m.get("frequency_hz") is not None]
    warnings = []
    if mode_freqs and max(mode_freqs) < freqs[-1]:
        warnings.append(
            "En yüksek hesaplanan mod (%.4g Hz) PSD'nin üst frekansından (%.4g Hz) küçük; bu bandın üstündeki "
            "modlar hesaba katılmadı. 'num_modes' artırın." % (max(mode_freqs), freqs[-1])
        )
    if not any(freqs[0] <= f <= freqs[-1] for f in mode_freqs):
        warnings.append("PSD bandında [%.4g, %.4g] Hz hiç mod yok; tepki yalnızca yarı-statik kalır." % (freqs[0], freqs[-1]))
    yield_mpa = material.yield_strength.value
    metrics = [
        ResultMetric(metric_name="displacement_1sigma_%s" % axis, value=round(disp_1s_mm, 6), unit="mm",
                     note="Taban ekseni doğrultusunda en büyük 1 sigma yer değiştirme (taban hareketine göre)"),
        ResultMetric(metric_name="displacement_3sigma_%s" % axis, value=round(3.0 * disp_1s_mm, 6), unit="mm"),
        ResultMetric(metric_name="von_mises_1sigma", value=round(vm_1s_mpa, 6), unit="MPa"),
        ResultMetric(metric_name="von_mises_3sigma", value=round(3.0 * vm_1s_mpa, 6), unit="MPa"),
        ResultMetric(metric_name="yield_margin_3sigma", value=round(yield_mpa / (3.0 * vm_1s_mpa), 4) if vm_1s_mpa > 0 else 0.0,
                     unit="", note="Akma dayanımı / 3 sigma von Mises"),
    ]

    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])
    solver_report = {
        "solver": solve_res["solver"], "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"], "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"], "solver_messages": solve_res["messages"],
        "mechdb": mechdb, "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "material": mat_report, "mesh": mesh_report,
        "boundary_conditions": setup["boundary_conditions"],
        "random_vibration": {"base_axis": axis, "damping_ratio": zeta, "psd_frequencies_hz": freqs,
                             "psd_values_g2_per_hz": cfg["psd_values_g2_per_hz"],
                             "modal_frequencies_hz": mode_freqs, "warnings": warnings},
        "executed_transcript": transcript,
    }
    manifest = AnalysisManifest(
        job_id=job_id, analysis_type=AnalysisType.RANDOM_VIBRATION, created_at=job["created_at"],
        cad_file_path=job["cad_file_path"], cad_sha256=cad_sha256, material=material,
        mesh_config=mesh_cfg, boundary_conditions=bcs, results=metrics,
        artifacts=[transcript, mechdb] + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED, solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    return {
        "status": "completed", "job_id": job_id, "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"], "solve_time_s": solve_res["solve_time_s"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "damping_ratio": zeta, "modal_frequencies_hz": mode_freqs,
        "displacement_1sigma_mm": round(disp_1s_mm, 6), "displacement_3sigma_mm": round(3.0 * disp_1s_mm, 6),
        "von_mises_1sigma_mpa": round(vm_1s_mpa, 6), "von_mises_3sigma_mpa": round(3.0 * vm_1s_mpa, 6),
        "warnings": warnings, "mechdb": mechdb,
        "message": "Gerçek ANSYS rastgele titreşim çözümü tamamlandı. Değerler 1 sigma (RMS) ve 3 sigma olarak verilmiştir.",
    }


@mcp.tool()
def evaluate_resonance_risk(
    job_id: str,
    operating_rpms: Optional[List[float]] = None,
    excitation_frequencies_hz: Optional[List[float]] = None,
    min_safe_margin_pct: float = 15.0
) -> Dict[str, Any]:
    """
    Doğal frekanslar ile çalışma devirleri (RPM) ve harici titreşim kaynakları arasındaki
    rezonans riskini (ISO / API ayrım kriteri) denetler.
    
    Args:
        job_id: Analiz kimliği.
        operating_rpms: Çalışma devirleri [1500, 3000].
        excitation_frequencies_hz: Bilinen harici frekanslar [50.0, 100.0].
        min_safe_margin_pct: Güvenli ayrım marjı yüzdesi (Varsayılan %15.0).
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış modal analiz bulunamadı."}

    manifest = json.loads(job["manifest_json"])

    # Effective mass ratios come from the participation table the solver wrote;
    # the resonance report must not invent them, because a mode that carries no
    # mass in the excitation direction is not the same risk as one that does.
    participation = (manifest.get("solver_report") or {}).get("participation_factors")
    by_mode = {}
    if participation:
        for row in participation.get("modes", []):
            by_mode[int(row["mode"])] = row

    modes_results = []
    for r in manifest["results"]:
        if "natural_frequency_mode_" in r["metric_name"]:
            m_num = int(r["metric_name"].split("_")[-1])
            pf = by_mode.get(m_num, {})
            modes_results.append(ModeShapeResult(
                mode_number=m_num,
                frequency_hz=r["value"],
                period_sec=round(1.0 / r["value"], 5) if r["value"] > 0 else 0.0,
                mode_type=r.get("note", "Vibration Mode"),
                effective_mass_ratio_x=float(pf.get("effective_mass_ratio_x", 0.0)),
                effective_mass_ratio_y=float(pf.get("effective_mass_ratio_y", 0.0)),
                effective_mass_ratio_z=float(pf.get("effective_mass_ratio_z", 0.0)),
                effective_mass_ratio_rotx=float(pf.get("effective_mass_ratio_rotx", 0.0)),
                effective_mass_ratio_roty=float(pf.get("effective_mass_ratio_roty", 0.0)),
                effective_mass_ratio_rotz=float(pf.get("effective_mass_ratio_rotz", 0.0)),
            ))

    if not modes_results:
        return {
            "status": "error",
            "message": "Manifest içinde doğal frekans bulunamadı. Önce 'solve_modal_analysis' çalıştırılmalıdır.",
        }

    try:
        res_report = ResonanceEvaluator.evaluate_resonance(
            modes=modes_results,
            operating_rpms=operating_rpms,
            discrete_excitation_frequencies_hz=excitation_frequencies_hz,
            min_safe_margin_pct=min_safe_margin_pct
        )
    except ValueError as exc:
        return {"status": "error", "job_id": job_id, "message": str(exc)}

    return {
        "status": "success",
        "job_id": job_id,
        "resonance_report": res_report.model_dump()
    }


@mcp.tool()
def render_modal_mode_shapes(job_id: str, num_modes: int = 3) -> Dict[str, Any]:
    """
    Çözülmüş modal sonuç dosyasından (.rst) gerçek mod şekli görsellerini üretir.

    Deformasyon alanı çözücünün özvektörüdür; sentetik alan kullanılmaz. Modal
    analiz çözülmemişse veya sonuç dosyası yoksa hata döner.

    Args:
        job_id: Tamamlanmış modal analiz kimliği.
        num_modes: Görselleştirilecek mod sayısı.
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış modal analiz bulunamadı."}

    manifest = json.loads(job["manifest_json"])
    rst_file = (manifest.get("solver_report") or {}).get("rst_file")
    if not rst_file or not os.path.exists(rst_file):
        return {
            "status": "error",
            "message": "Modal sonuç dosyası (.rst) bulunamadı; mod şekli üretilemez.",
        }

    out_dir = os.path.join(storage.get_job_dir(job_id), "plots", "modal")
    try:
        plots = ModeShapeVisualizer.render_mode_shapes(
            rst_file_path=rst_file,
            out_dir=out_dir,
            num_modes=num_modes,
            job_id=job_id,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return {
        "status": "success",
        "job_id": job_id,
        "rst_file": rst_file,
        "mode_plots": plots,
    }


# ---------------------------------------------------------------------------
# 1.6 Faz 9 Termal & Isı Transferi Analiz Araçları
# ---------------------------------------------------------------------------

@mcp.tool()
def configure_thermal_analysis(
    job_id: str,
    cad_file_path: str,
    material_name: str = "Aluminum 6061-T6",
    thermal_conductivity_w_m_k: float = 167.0,
    density_kg_m3: float = 2700.0,
    specific_heat_j_kg_k: float = 896.0,
    youngs_modulus_gpa: float = 68.9,
    poissons_ratio: float = 0.33,
    yield_strength_mpa: float = 276.0,
    thermal_expansion_per_k: float = 2.36e-5,
    mesh_global_size_mm: float = 5.0,
    heat_flow_w: Optional[float] = None,
    heat_face_normal: Optional[List[float]] = None,
    heat_face_axis_extreme: Optional[str] = None,
    internal_heat_generation_w_m3: Optional[float] = None,
    convection_coeff_w_m2_k: Optional[float] = 25.0,
    ambient_temp_c: float = 25.0,
    convection_face_normal: Optional[List[float]] = None,
    convection_face_axis_extreme: Optional[str] = None,
    fixed_temperature_c: Optional[float] = None,
    fixed_temp_face_normal: Optional[List[float]] = None,
    fixed_temp_face_axis_extreme: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ANSYS Mechanical kararlı durum termal analizini yapılandırır.

    Kararlı durumda enerji dengesi kurulabilmesi için en az bir ısı kaynağı
    (yüzey ısı akışı veya hacimsel iç ısı üretimi) ve en az bir ısı kuyusu
    (taşınım yüzeyi veya sabit sıcaklık yüzeyi) tanımlanmalıdır; aksi halde
    çözüm tekildir. Yüzeyler normal vektörüyle ya da eksen ekstremumuyla
    ('min_x', 'max_z' gibi) seçilir.

    Args:
        job_id: Analiz kimliği.
        cad_file_path: Katı CAD geometri yolu (STEP / IGES / Parasolid / SAT).
        material_name: Malzeme adı.
        thermal_conductivity_w_m_k: Isıl iletkenlik (W/m·K; Alüminyum ~167, Çelik ~50).
        density_kg_m3: Yoğunluk (kg/m³).
        specific_heat_j_kg_k: Özgül ısı (J/kg·K).
        youngs_modulus_gpa: Elastisite modülü (GPa) - termal gerilme denetimi için.
        poissons_ratio: Poisson oranı.
        yield_strength_mpa: Akma mukavemeti (MPa).
        thermal_expansion_per_k: Isıl genleşme katsayısı (1/K).
        mesh_global_size_mm: Global mesh eleman boyutu (mm).
        heat_flow_w: Bir yüzeye giren toplam ısı gücü (W).
        heat_face_normal: Isı giren yüzeyin normali [nx, ny, nz].
        heat_face_axis_extreme: Alternatif seçim, örn. 'min_x'.
        internal_heat_generation_w_m3: Hacimsel iç ısı üretimi (W/m³).
        convection_coeff_w_m2_k: Taşınım katsayısı (W/m²·K). None ise taşınım eklenmez.
        ambient_temp_c: Ortam sıcaklığı (°C).
        convection_face_normal: Taşınım yüzeyinin normali.
        convection_face_axis_extreme: Alternatif seçim, örn. 'max_x'.
        fixed_temperature_c: Sabit sıcaklık şartı (°C).
        fixed_temp_face_normal: Sabit sıcaklık yüzeyinin normali.
        fixed_temp_face_axis_extreme: Alternatif seçim, örn. 'max_z'.
    """
    job = storage.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job bulunamadı: {job_id}"}

    has_source = (heat_flow_w is not None and heat_flow_w != 0.0) or (
        internal_heat_generation_w_m3 is not None and internal_heat_generation_w_m3 != 0.0
    )
    has_sink = convection_coeff_w_m2_k is not None or fixed_temperature_c is not None
    if not has_source or not has_sink:
        return {
            "status": "error",
            "message": (
                "Kararlı durum termal analizi için hem ısı kaynağı "
                "(heat_flow_w veya internal_heat_generation_w_m3) hem de ısı kuyusu "
                "(convection_coeff_w_m2_k veya fixed_temperature_c) tanımlanmalıdır."
            ),
        }

    mat = Material(
        name=material_name,
        youngs_modulus=Quantity(value=youngs_modulus_gpa, unit="GPa"),
        poissons_ratio=poissons_ratio,
        yield_strength=Quantity(value=yield_strength_mpa, unit="MPa"),
        density=Quantity(value=density_kg_m3, unit="kg/m^3"),
        thermal_conductivity=Quantity(value=thermal_conductivity_w_m_k, unit="W/mK"),
        specific_heat=Quantity(value=specific_heat_j_kg_k, unit="J/kgK"),
        thermal_expansion=Quantity(value=thermal_expansion_per_k, unit="1/K"),
    )
    mesh_cfg = MeshConfig(global_size=Quantity(value=mesh_global_size_mm, unit="mm"))

    bcs: List[BoundaryCondition] = []

    if internal_heat_generation_w_m3:
        bcs.append(
            BoundaryCondition(
                name="Ic_Isi_Uretimi",
                type="internal_heat_generation",
                target_selector=SemanticSelector(name="Tum_Govde"),
                magnitude=Quantity(value=internal_heat_generation_w_m3, unit="W/m^3"),
            )
        )

    if heat_flow_w:
        heat_filter = _extreme_filter(heat_face_axis_extreme)
        if not heat_face_normal and not heat_filter:
            heat_face_normal = [-1.0, 0.0, 0.0]
        bcs.append(
            BoundaryCondition(
                name="Isi_Girisi",
                type="heat_flow",
                target_selector=SemanticSelector(
                    name="Isi_Yuzeyi",
                    normal_vector=heat_face_normal,
                    position_filter=heat_filter,
                    confidence_threshold=0.85,
                ),
                magnitude=Quantity(value=heat_flow_w, unit="W"),
            )
        )

    if convection_coeff_w_m2_k is not None:
        conv_filter = _extreme_filter(convection_face_axis_extreme)
        if not convection_face_normal and not conv_filter:
            convection_face_normal = [1.0, 0.0, 0.0]
        bcs.append(
            BoundaryCondition(
                name="Tasinim",
                type="convection",
                target_selector=SemanticSelector(
                    name="Sogutma_Yuzeyi",
                    normal_vector=convection_face_normal,
                    position_filter=conv_filter,
                    confidence_threshold=0.85,
                ),
                film_coefficient=Quantity(value=convection_coeff_w_m2_k, unit="W/m2K"),
                ambient_temperature=Quantity(value=ambient_temp_c, unit="C"),
            )
        )

    if fixed_temperature_c is not None:
        temp_filter = _extreme_filter(fixed_temp_face_axis_extreme)
        if not fixed_temp_face_normal and not temp_filter:
            fixed_temp_face_normal = [1.0, 0.0, 0.0]
        bcs.append(
            BoundaryCondition(
                name="Sabit_Sicaklik",
                type="temperature",
                target_selector=SemanticSelector(
                    name="Sabit_Sicaklik_Yuzeyi",
                    normal_vector=fixed_temp_face_normal,
                    position_filter=temp_filter,
                    confidence_threshold=0.85,
                ),
                magnitude=Quantity(value=fixed_temperature_c, unit="C"),
            )
        )

    thermal_cfg = {
        "analysis_type": "thermal_steady",
        "material": mat.model_dump(),
        "mesh_config": mesh_cfg.model_dump(),
        "boundary_conditions": [bc.model_dump() for bc in bcs],
    }

    storage.save_job_config(job_id=job_id, cad_path=cad_file_path, config=thermal_cfg)

    return {
        "status": "configured",
        "job_id": job_id,
        "material": material_name,
        "boundary_conditions": [bc.model_dump() for bc in bcs],
        "message": "Termal analiz yapılandırıldı. 'solve_thermal_analysis' ile gerçek ANSYS çözümünü başlatabilirsiniz.",
    }


@mcp.tool()
@exclusive_solver
@fresh_run_only
def solve_thermal_analysis(job_id: str, show_gui: bool = True) -> Dict[str, Any]:
    """
    Kararlı durum termal analizini GERÇEK ANSYS 2025 Mechanical oturumunda çözer.

    Sıcaklık dağılımı ve ısı akısı çözülmüş modelden okunur; kapalı form ısıl
    direnç tahmini kullanılmaz.

    Args:
        job_id: Analiz kimliği.
        show_gui: True ise Mechanical penceresi ekranda açılır ve adımlar izlenir.
    """
    job = storage.get_job(job_id)
    if not job or not job["config_json"]:
        return {"status": "error", "message": "Termal analiz bulunamadı veya konfigüre edilmedi."}

    job_dir = storage.get_job_dir(job_id)
    cfg = json.loads(job["config_json"])
    if "material" not in cfg or "boundary_conditions" not in cfg:
        return {
            "status": "error",
            "message": "Bu job eski biçimde konfigüre edilmiş. 'configure_thermal_analysis' yeniden çalıştırılmalıdır.",
        }

    material = Material(**cfg["material"])
    mesh_cfg = MeshConfig(**cfg["mesh_config"])
    bcs = [BoundaryCondition(**b) for b in cfg["boundary_conditions"]]

    adapter = MechanicalAdapter(visible=show_gui)

    try:
        storage.update_job_state(job_id, JobState.QUEUED)
        adapter.initialize_session(job_id, job_dir)

        storage.update_job_state(job_id, JobState.IMPORTING)
        geometry = adapter.load_geometry(job["cad_file_path"])
        mat_report = adapter.apply_material(material)

        storage.update_job_state(job_id, JobState.MESHING)
        mesh_report = adapter.generate_mesh(mesh_cfg)

        thermal_setup = adapter.setup_thermal(bcs)

        storage.update_job_state(job_id, JobState.SOLVING)
        solve_res = adapter.solve_thermal()

        storage.update_job_state(job_id, JobState.POST_PROCESSING)
        results = adapter.extract_thermal_results()
        t_plots = adapter.export_thermal_images()
        transcript = adapter.export_solve_script()
        mechdb = adapter.save_project()

    except Exception as exc:
        _mark_failed(job_id, exc)
        return {
            "status": "failed",
            "job_id": job_id,
            "state": JobState.FAILED.value,
            "error": str(exc),
            "step_log": adapter.step_log[-12:] if adapter._session else [],
            "message": "Gerçek ANSYS termal çözümü başarısız oldu. Sentetik sıcaklık üretilmedi.",
        }

    cad_sha256 = GeometryIntelligence.compute_file_sha256(job["cad_file_path"])

    solver_report = {
        "solver": solve_res["solver"],
        "solution_state": solve_res["solution_state"],
        "solve_time_s": solve_res["solve_time_s"],
        "rst_file": solve_res["rst_file"],
        "working_dir": solve_res["working_dir"],
        "solver_messages": solve_res["messages"],
        "mechdb": mechdb,
        "geometry": {k: v for k, v in geometry.items() if k != "faces"},
        "material": mat_report,
        "mesh": mesh_report,
        "boundary_conditions": thermal_setup["boundary_conditions"],
        "executed_transcript": transcript,
    }

    manifest = AnalysisManifest(
        job_id=job_id,
        analysis_type=AnalysisType.THERMAL_STEADY,
        created_at=job["created_at"],
        cad_file_path=job["cad_file_path"],
        cad_sha256=cad_sha256,
        material=material,
        mesh_config=mesh_cfg,
        boundary_conditions=bcs,
        results=results,
        artifacts=sorted(t_plots.values()) + [transcript, mechdb]
        + ([solve_res["rst_file"]] if solve_res.get("rst_file") else []),
        ansys_version=SystemConfig.get_ansys_summary().get("version", "Unknown"),
        status=JobState.COMPLETED,
        solver_report=solver_report,
    )
    storage.save_manifest(job_id, manifest)
    storage.update_job_state(job_id, JobState.COMPLETED)

    return {
        "status": "completed",
        "job_id": job_id,
        "state": JobState.COMPLETED.value,
        "solver": solve_res["solver"],
        "solve_time_s": solve_res["solve_time_s"],
        "mesh": {k: mesh_report[k] for k in ("nodes", "elements") if k in mesh_report},
        "results": [r.model_dump() for r in results],
        "plots": t_plots,
        "mechdb": mechdb,
        "message": "Gerçek ANSYS termal çözümü tamamlandı. Termal gerilme taraması için 'evaluate_thermal_stresses' kullanabilirsiniz.",
    }


@mcp.tool()
def evaluate_thermal_stresses(
    job_id: str,
    reference_temp_c: float,
    constraint_factor: float,
    cte_per_k: Optional[float] = None,
    youngs_modulus_gpa: Optional[float] = None,
    yield_strength_mpa: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Sıcaklık farkına bağlı termal genleşme gerilmesini ve akma emniyet faktörünü tarar.

    Bu araç bir FEA çözümü DEĞİLDİR: çözülmüş termal analizden okunan gerçek
    maksimum sıcaklığı kullanarak kapalı form sigma = C * E * alpha * DeltaT
    tahmini yapar. Gerçek termal gerilme dağılımı için sıcaklık alanını statik
    yapısal analize aktaran bir kuplaj gerekir.
    
    Malzeme değerleri (E, akma, ısıl genleşme) verilmezse job'un kayıtlı malzeme
    kartından okunur; kartta yoksa hata döner, varsayılan bir malzeme uydurulmaz.
    Referans sıcaklık ve kısıtlanma oranı mühendisin kararıdır ve zorunludur.

    Args:
        job_id: Analiz kimliği.
        reference_temp_c: Gerilmesiz montaj referans sıcaklığı (°C).
        constraint_factor: Mekanik kısıtlanma oranı (0.0 = Serbest, 1.0 = Tam Rijit).
        cte_per_k: Isıl genleşme katsayısı (1/K). Boşsa malzeme kartından.
        youngs_modulus_gpa: Elastisite Modülü (GPa). Boşsa malzeme kartından.
        yield_strength_mpa: Akma mukavemeti (MPa). Boşsa malzeme kartından.
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış termal analiz bulunamadı."}

    manifest = json.loads(job["manifest_json"])
    card = manifest.get("material") or {}
    try:
        if youngs_modulus_gpa is None and card.get("youngs_modulus"):
            youngs_modulus_gpa = UnitEngine.convert(Quantity(**card["youngs_modulus"]), "GPa").value
        if yield_strength_mpa is None and card.get("yield_strength"):
            yield_strength_mpa = UnitEngine.convert(Quantity(**card["yield_strength"]), "MPa").value
        if cte_per_k is None and card.get("thermal_expansion"):
            cte_per_k = UnitEngine.to_si(Quantity(**card["thermal_expansion"])).value
        missing = [n for n, v in (("cte_per_k", cte_per_k), ("youngs_modulus_gpa", youngs_modulus_gpa),
                                  ("yield_strength_mpa", yield_strength_mpa)) if v is None]
        if missing:
            return {"status": "error", "message": "Malzeme kartında yok ve verilmedi: %s" % ", ".join(missing)}
        values = dict(
            reference_temp_c=require_finite("reference_temp_c", reference_temp_c),
            cte_per_k=require_positive("cte_per_k", cte_per_k),
            youngs_modulus_gpa=require_positive("youngs_modulus_gpa", youngs_modulus_gpa),
            yield_strength_mpa=require_positive("yield_strength_mpa", yield_strength_mpa),
            constraint_factor=require_finite("constraint_factor", constraint_factor),
        )
        if not (0.0 <= values["constraint_factor"] <= 1.0):
            return {"status": "error", "message": "constraint_factor 0 ile 1 arasında olmalıdır."}
    except (InputError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}

    t_max = next(
        (r["value"] for r in manifest["results"] if r["metric_name"] == "maximum_temperature"),
        None,
    )
    if t_max is None:
        return {
            "status": "error",
            "message": (
                "Manifest içinde çözülmüş maksimum sıcaklık yok. "
                "Önce 'solve_thermal_analysis' çalıştırılmalıdır."
            ),
        }

    stress_report = ThermalStressCoupler.evaluate_thermal_stress(max_temp_c=t_max, **values)

    return {
        "status": "success",
        "job_id": job_id,
        "thermal_stress_report": stress_report.model_dump()
    }


@mcp.tool()
def render_structural_contours(job_id: str, deformation_scale: float = 0.0) -> Dict[str, Any]:
    """
    Çözülmüş statik yapısal sonuçtan von Mises gerilme ve toplam yer değiştirme konturlarını (PNG) üretir.

    Skaler değerler çözücünün sonuç dosyasından DPF ile okunur, hiçbir alan uydurulmaz. Dönen
    'field_maxima' değerleri, manifestodaki çözücü metrikleriyle karşılaştırılır; fark varsa
    'consistent_with_solver' false olur.

    Args:
        job_id: Tamamlanmış statik veya burkulma dışı yapısal analiz kimliği.
        deformation_scale: 0 ise şekil değiştirmemiş ağ, >0 ise yer değiştirmenin bu katı kadar kaydırılmış ağ.
    """
    from .adapters.structural_visualizer import StructuralRenderError, StructuralVisualizer

    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış analiz bulunamadı."}
    manifest = json.loads(job["manifest_json"])
    if manifest.get("analysis_type") != AnalysisType.STATIC_STRUCTURAL.value:
        return {"status": "error", "message": "Kontur yalnızca 'static_structural' sonuçları için üretilir."}
    try:
        scale = require_finite("deformation_scale", deformation_scale)
        if scale < 0:
            raise InputError("deformation_scale negatif olamaz.")
    except InputError as exc:
        return {"status": "error", "message": str(exc)}
    result_file = (manifest.get("solver_report") or {}).get("rst_file")
    out_dir = os.path.join(storage.get_job_dir(job_id), "plots", "structural")
    try:
        rendered = StructuralVisualizer.render(result_file, out_dir, job_id=job_id, deformation_scale=scale)
    except StructuralRenderError as exc:
        return {"status": "error", "message": str(exc)}

    reported = {r["metric_name"]: r["value"] for r in manifest.get("results", [])}
    maxima = rendered["field_maxima"]
    checks = {}
    for key, metric in (("von_mises_mpa", "maximum_von_mises_stress"),
                        ("total_displacement_mm", "maximum_total_deformation")):
        if metric in reported and reported[metric]:
            checks[metric] = abs(maxima[key] - reported[metric]) <= 0.01 * abs(reported[metric])
    return {
        "status": "success",
        "job_id": job_id,
        "result_file": result_file,
        "plots": rendered["plots"],
        "field_maxima": maxima,
        "consistent_with_solver": all(checks.values()) if checks else None,
        "consistency_checks": checks,
    }


@mcp.tool()
def export_result_field(job_id: str, result: str, set_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Çözülmüş bir yapısal/termal sonucu düğüm bazında CSV (ve VTU) olarak dışa aktarır.

    Değerler çözücünün sonuç dosyasından (.rst) DPF ile okunur; hiçbir değer üretilmez veya
    enterpole edilmez. Konumlar mm, yer değiştirme mm, gerilme MPa, kuvvet N cinsindendir.
    Dönen 'extrema' dosyadaki gerçek en küçük/en büyük değerlerdir; mümkünse manifestodaki
    çözücü metriğiyle karşılaştırılıp 'consistent_with_solver' verilir.

    Args:
        job_id: Tamamlanmış analiz kimliği (statik, modal, burkulma, harmonik, geçici, termal).
        result: 'displacement', 'displacement_magnitude', 'stress' (6 bileşen), 'von_mises',
            'principal_stress' (s1, s2, s3), 'elastic_strain', 'reaction_force' veya 'temperature'.
            Çözülen türde bulunmayan bir sonuç istenirse hata döner.
        set_number: Çözülmüş küme (1 tabanlı: zaman adımı, mod, frekans). Boşsa son küme.
    """
    from .adapters.result_exporter import SUPPORTED, ResultExportError, export_field, list_time_sets

    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış analiz bulunamadı."}
    if result not in SUPPORTED:
        return {"status": "error", "message": "Bilinmeyen sonuç %r. Desteklenenler: %s." % (result, SUPPORTED)}
    manifest = json.loads(job["manifest_json"])
    rst = (manifest.get("solver_report") or {}).get("rst_file")
    if not rst or not os.path.exists(rst):
        return {"status": "error", "message": "Bu işin çözücü sonuç dosyası (.rst) bulunamadı: %r." % rst}
    out_dir = os.path.join(storage.get_job_dir(job_id), "exports")
    try:
        summary = export_field(rst, result, out_dir, set_number=set_number)
    except ResultExportError as exc:
        return {"status": "error", "message": str(exc)}

    reported = {r["metric_name"]: r["value"] for r in manifest.get("results", [])}
    checks = {}
    if manifest.get("analysis_type") == AnalysisType.STATIC_STRUCTURAL.value:
        pairs = {"von_mises": ("von_mises", "maximum_von_mises_stress"),
                 "displacement_magnitude": ("displacement_magnitude", "maximum_total_deformation")}
        if result in pairs:
            column, metric = pairs[result]
            if metric in reported and reported[metric]:
                got = summary["extrema"][column]["max"]
                checks[metric] = abs(got - reported[metric]) <= 0.01 * abs(reported[metric])
    return {
        "status": "success", "job_id": job_id, "result_file": rst, **summary,
        "consistent_with_solver": all(checks.values()) if checks else None,
        "consistency_checks": checks,
    }


@mcp.tool()
def render_thermal_contours(job_id: str) -> Dict[str, Any]:
    """
    Çözülmüş termal sonuç dosyasından gerçek sıcaklık ve ısı akısı konturlarını üretir.

    Skaler alanlar çözücünün kendi sonucundan okunur; sentetik dağılım kullanılmaz.

    Args:
        job_id: Tamamlanmış termal analiz kimliği.
    """
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return {"status": "error", "message": "Tamamlanmış termal analiz bulunamadı."}

    manifest = json.loads(job["manifest_json"])
    result_file = (manifest.get("solver_report") or {}).get("rst_file")
    if not result_file or not os.path.exists(result_file):
        return {
            "status": "error",
            "message": "Termal sonuç dosyası bulunamadı; kontur üretilemez.",
        }

    out_dir = os.path.join(storage.get_job_dir(job_id), "plots", "thermal")
    try:
        plots = ThermalVisualizer.render_thermal_plots(
            result_file_path=result_file,
            out_dir=out_dir,
            job_id=job_id,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return {
        "status": "success",
        "job_id": job_id,
        "result_file": result_file,
        "plots": plots,
    }


# ---------------------------------------------------------------------------
# 2. MCP Resources (Okunabilir URI Akışları)
# ---------------------------------------------------------------------------

@mcp.resource("ansys://job/{job_id}/manifest")
def get_manifest_resource(job_id: str) -> str:
    """Returns the immutable SHA-256 analysis manifest for provenance."""
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return json.dumps({"error": "Manifest bulunamadı veya henüz oluşturulmadı."})
    return job["manifest_json"]


@mcp.resource("ansys://job/{job_id}/logs")
def get_logs_resource(job_id: str) -> str:
    """Returns the solver and orchestrator log trail."""
    logs = storage.get_job_logs(job_id)
    return json.dumps(logs, indent=2)


@mcp.resource("ansys://job/{job_id}/validation")
def get_validation_resource(job_id: str) -> str:
    """Returns the Gatekeeper validation report."""
    job = storage.get_job(job_id)
    if not job or not job["validation_json"]:
        return json.dumps({"error": "Doğrulama raporu mevcut değil."})
    return job["validation_json"]


@mcp.resource("ansys://job/{job_id}/mesh-convergence")
def get_mesh_convergence_resource(job_id: str) -> str:
    """Returns the stored ASME Grid Convergence Index (GCI) report; never starts a solve."""
    res = MeshStudyRunner.load_stored(storage, job_id)
    if res is None:
        res = {
            "error": "Bu job için kayıtlı bir mesh çalışması yok. Önce run_mesh_convergence_study çalıştırın."
        }
    return json.dumps(res, ensure_ascii=False, indent=2)


@mcp.resource("ansys://job/{job_id}/cfd-monitors")
def get_cfd_monitors_resource(job_id: str) -> str:
    """Returns the residual history Fluent recorded for a solved CFD job."""
    job = storage.get_job(job_id)
    if not job:
        return json.dumps({"error": "Job bulunamadı."}, ensure_ascii=False)

    info_path = os.path.join(storage.get_job_dir(job_id), "cfd_solver_info.json")
    if not os.path.exists(info_path):
        return json.dumps({
            "error": "Bu job için çözülmüş bir Fluent kaydı yok. Önce solve_cfd_simulation çalıştırın."
        }, ensure_ascii=False, indent=2)

    with open(info_path, encoding="utf-8") as handle:
        solve_info = json.load(handle)

    residuals = solve_info.get("final_residuals") or {}
    return json.dumps({
        "job_id": job_id,
        "solver": solve_info.get("solver"),
        "turbulence_model": solve_info.get("turbulence_model"),
        "iterations_run": solve_info.get("iterations_run"),
        "iterations_requested": solve_info.get("iterations_requested"),
        "residuals": residuals,
        "max_residual": max(residuals.values()) if residuals else None,
        "convergence_status": solve_info.get("status"),
        "convergence_note": solve_info.get("residual_convergence"),
        "solve_time_s": solve_info.get("solve_time_s"),
    }, ensure_ascii=False, indent=2)


@mcp.resource("ansys://job/{job_id}/provenance")
def get_provenance_resource(job_id: str) -> str:
    """Returns the cryptographic SHA-256 provenance and digital signature."""
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return json.dumps({"error": "Provenance kaydı bulunamadı."})

    prov = storage.load_provenance(job_id)
    if prov is None:
        return json.dumps({"error": "Bu job için kayıtlı bir provenance kaydı yok."})
    return prov.model_dump_json(indent=2)


@mcp.resource("ansys://optimization/{study_id}/pareto-front")
def get_pareto_front_resource(study_id: str) -> str:
    """Returns the Pareto optimal frontier for a parametric study."""
    try:
        report = OptimizationBatchRunner.get_report(storage=storage, study_id=study_id)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)
    return json.dumps(report, ensure_ascii=False, indent=2)


@mcp.resource("ansys://job/{job_id}/modal-frequencies")
def get_modal_frequencies_resource(job_id: str) -> str:
    """Returns natural frequencies and mode shape results."""
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return json.dumps({"error": "Modal analiz sonuçları bulunamadı."})
    return job["manifest_json"]


@mcp.resource("ansys://job/{job_id}/thermal-results")
def get_thermal_results_resource(job_id: str) -> str:
    """Returns steady-state thermal temperature and heat flux results."""
    job = storage.get_job(job_id)
    if not job or not job["manifest_json"]:
        return json.dumps({"error": "Termal analiz sonuçları bulunamadı."})
    return job["manifest_json"]


# ---------------------------------------------------------------------------
# 3. Main Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
