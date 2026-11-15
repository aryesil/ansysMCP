"""
Strict Validation Gatekeeper
----------------------------
Validates analysis models before submission to solvers.
Blocks incomplete, unconstrained, or unphysical simulations from wasting compute and license.
"""

import math
import os
from typing import List
from ..domain.models import (
    AnalysisType,
    Material,
    BoundaryCondition,
    BoundaryConditionType,
    MeshConfig,
    ValidationReport,
    ValidationCheck,
)
from ..domain.units import UnitEngine, UnitError
from .input_checks import check_material_card


class GatekeeperValidator:
    """Performs rigorous pre-solve engineering and consistency validation."""

    # Solid formats a structural solver can import. A faceted STL carries no
    # surfaces to attach a support or load to, so it is allowed for CFD only,
    # where the body is only ever wrapped in a tunnel.
    ALLOWED_CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".sat", ".x_t", ".x_b"}
    CFD_ONLY_CAD_EXTENSIONS = {".stl"}

    @classmethod
    def validate_setup(
        cls,
        job_id: str,
        analysis_type: AnalysisType,
        cad_file_path: str,
        material: Material,
        mesh_config: MeshConfig,
        boundary_conditions: List[BoundaryCondition],
    ) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        checks: List[ValidationCheck] = []

        # -------------------------------------------------------------
        # 1. CAD File Validation
        # -------------------------------------------------------------
        if not os.path.exists(cad_file_path):
            errors.append(f"CAD dosyası bulunamadı: {cad_file_path}")
            checks.append(ValidationCheck(
                check_name="CAD_EXISTS",
                passed=False,
                message=f"CAD dosyası mevcut değil: {cad_file_path}"
            ))
        else:
            ext = os.path.splitext(cad_file_path)[1].lower()
            allowed = set(cls.ALLOWED_CAD_EXTENSIONS)
            if analysis_type in (AnalysisType.CFD_STEADY, AnalysisType.CFD_TRANSIENT):
                allowed |= cls.CFD_ONLY_CAD_EXTENSIONS
            if ext not in allowed:
                errors.append(f"Desteklenmeyen CAD formatı: '{ext}'. İzin verilenler: {sorted(allowed)}")
                checks.append(ValidationCheck(
                    check_name="CAD_FORMAT",
                    passed=False,
                    message=f"Desteklenmeyen dosya uzantısı: {ext}"
                ))
            else:
                checks.append(ValidationCheck(
                    check_name="CAD_EXISTS_AND_VALID",
                    passed=True,
                    message=f"CAD dosyası doğrulandı ({ext})"
                ))

        # -------------------------------------------------------------
        # 2. Material Properties & Units Validation
        # -------------------------------------------------------------
        mat_passed = True
        problems = check_material_card(
            material.youngs_modulus,
            material.poissons_ratio,
            material.yield_strength,
            material.ultimate_strength,
            material.density,
            orthotropic=material.orthotropic is not None,
        )
        if problems:
            errors.extend(problems)
            mat_passed = False
        if material.density is None and analysis_type in (
            AnalysisType.STATIC_STRUCTURAL, AnalysisType.MODAL, AnalysisType.BUCKLING, AnalysisType.HARMONIC, AnalysisType.TRANSIENT, AnalysisType.RANDOM_VIBRATION
        ):
            errors.append("Yoğunluk tanımlı değil; yapısal ve modal analiz için gereklidir.")
            mat_passed = False

        checks.append(ValidationCheck(
            check_name="MATERIAL_PROPERTIES",
            passed=mat_passed,
            message="Malzeme özellikleri geçerli." if mat_passed else "Malzeme özelliklerinde hata tespit edildi."
        ))

        # -------------------------------------------------------------
        # 3. Mesh Configuration Validation
        # -------------------------------------------------------------
        mesh_passed = True
        try:
            global_size_si = UnitEngine.to_si(mesh_config.global_size).value
            if not math.isfinite(global_size_si) or global_size_si <= 0:
                errors.append("Global mesh boyutu pozitif olmalıdır.")
                mesh_passed = False
        except UnitError as ue:
            errors.append(f"Mesh boyutu birim hatası: {str(ue)}")
            mesh_passed = False

        if mesh_config.growth_rate < 1.05 or mesh_config.growth_rate > 2.0:
            errors.append(f"Mesh büyüme oranı 1.05 ile 2.0 arasında olmalıdır. Verilen: {mesh_config.growth_rate}")
            mesh_passed = False

        if mesh_config.target_quality < 0.15:
            warnings.append(f"Hedef ortogonal kalite hedefi düşük ({mesh_config.target_quality}). Önerilen: >= 0.20")

        checks.append(ValidationCheck(
            check_name="MESH_CONFIG",
            passed=mesh_passed,
            message="Mesh konfigürasyonu geçerli." if mesh_passed else "Mesh konfigürasyonunda hata var."
        ))

        # -------------------------------------------------------------
        # 4. Boundary Conditions & Rigid Body Motion Validation (FEA)
        # -------------------------------------------------------------
        if analysis_type in [AnalysisType.STATIC_STRUCTURAL, AnalysisType.MODAL, AnalysisType.BUCKLING, AnalysisType.HARMONIC, AnalysisType.TRANSIENT, AnalysisType.RANDOM_VIBRATION]:
            # A body is held against rigid motion by a fixed support, or by
            # displacement supports that together prescribe all three
            # translations. A frictionless support only blocks the normal
            # direction and cannot hold a body on its own.
            prescribed_axes = set()
            for bc in boundary_conditions:
                if bc.type == BoundaryConditionType.DISPLACEMENT and bc.components_mm:
                    prescribed_axes.update(i for i, c in enumerate(bc.components_mm) if c is not None)
            has_fixed_support = any(
                bc.type == BoundaryConditionType.FIXED_SUPPORT
                or (bc.type == BoundaryConditionType.CYLINDRICAL_SUPPORT
                    and bc.dofs and all(v == 'fixed' for v in bc.dofs.values()))
                for bc in boundary_conditions
            ) or prescribed_axes == {0, 1, 2}
            has_load = any(
                bc.type in [
                    BoundaryConditionType.FORCE, BoundaryConditionType.PRESSURE,
                    BoundaryConditionType.GRAVITY, BoundaryConditionType.REMOTE_FORCE,
                    BoundaryConditionType.MOMENT,
                ]
                for bc in boundary_conditions
            ) or any(
                bc.type == BoundaryConditionType.DISPLACEMENT and bc.components_mm
                and any(c not in (None, 0.0) for c in bc.components_mm)
                for bc in boundary_conditions
            )

            if not has_fixed_support:
                errors.append(
                    "Rijit Gövde Hareketi (Rigid Body Motion) Riski: Analizde bir 'fixed_support' ya da üç ötelemeyi birlikte tanımlayan 'displacement' mesnetleri bulunmalıdır; sürtünmesiz mesnet tek başına yeterli değildir."
                )
                checks.append(ValidationCheck(
                    check_name="CONSTRAINT_CHECK",
                    passed=False,
                    message="Sabit mesnet veya kısıtlayıcı sınır şartı eksik!"
                ))
            else:
                checks.append(ValidationCheck(
                    check_name="CONSTRAINT_CHECK",
                    passed=True,
                    message="Mesnetleme ve kısıtlar mevcut."
                ))

            if analysis_type == AnalysisType.TRANSIENT and not has_load:
                errors.append(
                    "Geçici (transient) analiz bir yük gerektirir: tanımlı yük t=0'da uygulanır."
                )
            if analysis_type == AnalysisType.HARMONIC and not has_load:
                errors.append(
                    "Harmonik analiz bir uyarım yükü gerektirir: tanımlı yük genlik olarak uygulanır."
                )
            if analysis_type == AnalysisType.BUCKLING and not has_load:
                errors.append(
                    "Burkulma analizi bir referans yük gerektirir: yük çarpanı bu yüke göre hesaplanır."
                )
            if analysis_type == AnalysisType.STATIC_STRUCTURAL and not has_load:
                warnings.append("Statik analizde tanımlı bir yük (Kuvvet, Basınç, Yerçekimi) bulunamadı. Yalnızca sıfır gerilme çözülecek.")

        # -------------------------------------------------------------
        # 5. Load definitions: finite magnitude, usable direction
        # -------------------------------------------------------------
        load_passed = True
        for bc in boundary_conditions:
            if bc.type in (
                BoundaryConditionType.FORCE, BoundaryConditionType.PRESSURE,
                BoundaryConditionType.REMOTE_FORCE, BoundaryConditionType.MOMENT,
                BoundaryConditionType.GRAVITY,
            ):
                if bc.magnitude is None or not math.isfinite(bc.magnitude.value) or bc.magnitude.value <= 0.0:
                    errors.append(f"'{bc.name}' için büyüklük sonlu ve sıfırdan büyük olmalıdır: {bc.magnitude}")
                    load_passed = False
            if bc.type == BoundaryConditionType.DISPLACEMENT:
                comps = bc.components_mm or []
                if len(comps) != 3 or all(c is None for c in comps) or any(
                    c is not None and not math.isfinite(c) for c in comps
                ):
                    errors.append(f"'{bc.name}' yer değiştirmesi için sonlu en az bir bileşen gerekir: {bc.components_mm}")
                    load_passed = False
            if bc.type == BoundaryConditionType.REMOTE_FORCE:
                origin = bc.origin_mm or []
                if len(origin) != 3 or not all(math.isfinite(c) for c in origin):
                    errors.append(f"'{bc.name}' uzak kuvvetinin 'origin_mm' noktası tanımsız: {bc.origin_mm}")
                    load_passed = False
            if bc.type in (
                BoundaryConditionType.FORCE, BoundaryConditionType.REMOTE_FORCE,
                BoundaryConditionType.MOMENT, BoundaryConditionType.GRAVITY,
            ):
                direction = bc.direction or []
                if len(direction) != 3 or not all(math.isfinite(c) for c in direction) or math.sqrt(sum(c * c for c in direction)) < 1.0e-9:
                    errors.append(f"'{bc.name}' yönü tanımsız veya sıfır vektör: {bc.direction}")
                    load_passed = False
        checks.append(ValidationCheck(
            check_name="LOAD_DEFINITIONS",
            passed=load_passed,
            message="Yük tanımları geçerli." if load_passed else "Yük tanımlarında hata var."
        ))

        is_valid = len(errors) == 0

        return ValidationReport(
            job_id=job_id,
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks=checks
        )
