"""
Cryptographic Provenance & Tamper Verification Engine
-----------------------------------------------------
Generates SHA-256 audit trails and digital provenance signatures for simulation runs.
Ensures simulation reproducibility and detects post-analysis data tampering.
"""

import hashlib
import json
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from ..domain.models import JobState


class ProvenanceRecord(BaseModel):
    job_id: str
    timestamp_utc: str
    server_version: str = "2.0.0"
    target_ansys_version: str = "ANSYS 2025 R1 (v251)"
    cad_sha256: str
    input_config_sha256: str
    results_sha256: str
    digital_signature: str
    artifacts: List[str]


class ProvenanceEngine:
    """Manages immutable audit trails and simulation reproducibility checks."""

    @staticmethod
    def compute_sha256(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @classmethod
    def generate_provenance(
        cls,
        job_id: str,
        cad_file_path: str,
        config_dict: Dict[str, Any],
        results_list: List[Dict[str, Any]],
        artifacts: List[str],
        timestamp_utc: str
    ) -> ProvenanceRecord:
        """
        Creates an immutable provenance record with a digital audit signature.
        """
        # 1. CAD Hash
        cad_hash = "no_cad"
        if os.path.exists(cad_file_path):
            with open(cad_file_path, "rb") as f:
                cad_hash = hashlib.sha256(f.read()).hexdigest()

        # 2. Config & Results Hash
        cfg_str = json.dumps(config_dict, sort_keys=True)
        res_str = json.dumps(results_list, sort_keys=True)

        cfg_hash = cls.compute_sha256(cfg_str)
        res_hash = cls.compute_sha256(res_str)

        # 3. Master Digital Signature
        sig_raw = f"{job_id}|{cad_hash}|{cfg_hash}|{res_hash}|{timestamp_utc}"
        digital_signature = cls.compute_sha256(sig_raw)

        return ProvenanceRecord(
            job_id=job_id,
            timestamp_utc=timestamp_utc,
            cad_sha256=cad_hash,
            input_config_sha256=cfg_hash,
            results_sha256=res_hash,
            digital_signature=digital_signature,
            artifacts=artifacts
        )

    @classmethod
    def verify_against_record(
        cls,
        record: ProvenanceRecord,
        cad_file_path: str,
        current_config_dict: Dict[str, Any],
        current_results_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check present data against a record that was written when the job finished.

        The record must come from storage, not be rebuilt from the data being
        checked: a record regenerated from current values always matches them.
        Beyond config and results this also re-hashes the CAD file and
        recomputes the master signature, so an edited record is caught too.
        """
        report = cls.verify_tampering(record, current_config_dict, current_results_list)

        cad_hash = "no_cad"
        if os.path.exists(cad_file_path):
            with open(cad_file_path, "rb") as handle:
                cad_hash = hashlib.sha256(handle.read()).hexdigest()
        cad_intact = cad_hash == record.cad_sha256

        expected_signature = cls.compute_sha256(
            f"{record.job_id}|{record.cad_sha256}|{record.input_config_sha256}|"
            f"{record.results_sha256}|{record.timestamp_utc}"
        )
        signature_intact = expected_signature == record.digital_signature

        is_valid = report["provenance_valid"] and cad_intact and signature_intact
        report.update({
            "provenance_valid": is_valid,
            "cad_intact": cad_intact,
            "signature_intact": signature_intact,
            "status": "PASS (Değişmezlik Doğrulandı)" if is_valid
            else "FAIL (Veri tahrif edildi veya değiştirildi!)",
        })
        return report

    @classmethod
    def verify_tampering(
        cls,
        record: ProvenanceRecord,
        current_config_dict: Dict[str, Any],
        current_results_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifies whether simulation inputs or results have been modified after run.
        """
        cfg_str = json.dumps(current_config_dict, sort_keys=True)
        res_str = json.dumps(current_results_list, sort_keys=True)

        curr_cfg_hash = cls.compute_sha256(cfg_str)
        curr_res_hash = cls.compute_sha256(res_str)

        config_tampered = (curr_cfg_hash != record.input_config_sha256)
        results_tampered = (curr_res_hash != record.results_sha256)

        is_valid = not config_tampered and not results_tampered

        return {
            "job_id": record.job_id,
            "provenance_valid": is_valid,
            "config_intact": not config_tampered,
            "results_intact": not results_tampered,
            "digital_signature": record.digital_signature,
            "status": "PASS (Değişmezlik Doğrulandı)" if is_valid else "FAIL (Veri tahrif edildi veya değiştirildi!)"
        }
