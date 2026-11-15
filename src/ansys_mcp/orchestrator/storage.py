"""
Job Persistence & Storage Layer
-------------------------------
SQLite and file-system backed persistence for analysis jobs, logs, and artifacts.
"""

import sqlite3
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from ..domain.models import JobState, AnalysisType, AnalysisManifest, ValidationReport
from .state_machine import check_update


class JobStorage:
    """Manages persistent job records in SQLite and workspace artifacts on disk."""

    def __init__(self, workspace_root: Optional[str] = None):
        # Default: a ``runs`` folder at the project root (next to server.py).
        if workspace_root is None:
            workspace_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "runs")
        self.workspace_root = workspace_root
        os.makedirs(self.workspace_root, exist_ok=True)
        self.db_path = os.path.join(self.workspace_root, "orchestrator.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    cad_file_path TEXT,
                    config_json TEXT,
                    validation_json TEXT,
                    manifest_json TEXT,
                    error_message TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                )
            """)
            conn.commit()

    def get_job_dir(self, job_id: str) -> str:
        """Returns isolated directory for job files, meshes, results, and plots."""
        job_dir = os.path.join(self.workspace_root, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def create_job(self, job_id: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (job_id, analysis_type, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, analysis_type.value, JobState.CREATED.value, now, now))
            conn.commit()
        
        self.get_job_dir(job_id)
        self.add_log(job_id, "INFO", f"Job oluşturuldu: {job_id} ({analysis_type.value})")
        return {"job_id": job_id, "state": JobState.CREATED.value, "created_at": now}

    def update_job_state(self, job_id: str, new_state: JobState, error_message: Optional[str] = None):
        current = self.get_job(job_id)
        if current is not None:
            check_update(JobState(current["state"]), new_state)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET state = ?, updated_at = ?, error_message = COALESCE(?, error_message)
                WHERE job_id = ?
            """, (new_state.value, now, error_message, job_id))
            conn.commit()
        self.add_log(job_id, "INFO" if not error_message else "ERROR", f"Durum değişti -> {new_state.value}")

    def save_job_config(self, job_id: str, cad_path: str, config: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET cad_file_path = ?, config_json = ?, state = ?, updated_at = ?
                WHERE job_id = ?
            """, (cad_path, json.dumps(config), JobState.CONFIGURED.value, now, job_id))
            conn.commit()
        self.add_log(job_id, "INFO", f"Analiz konfigürasyonu kaydedildi (CAD: {cad_path})")

    def save_validation_report(self, job_id: str, report: ValidationReport):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET validation_json = ?, updated_at = ?
                WHERE job_id = ?
            """, (report.model_dump_json(), now, job_id))
            conn.commit()
        self.add_log(job_id, "INFO" if report.valid else "WARNING", f"Doğrulama tamamlandı (Geçerli: {report.valid})")

    def save_manifest(self, job_id: str, manifest: AnalysisManifest):
        now = datetime.now(timezone.utc).isoformat()
        manifest_json = manifest.model_dump_json(indent=2)
        
        # Save to DB
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET manifest_json = ?, updated_at = ?
                WHERE job_id = ?
            """, (manifest_json, now, job_id))
            conn.commit()
            
        # Save file in job dir
        manifest_file = os.path.join(self.get_job_dir(job_id), "manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(manifest_json)

        self._sign_job(job_id, manifest)

    def provenance_path(self, job_id: str) -> str:
        return os.path.join(self.get_job_dir(job_id), "provenance.json")

    def _sign_job(self, job_id: str, manifest: AnalysisManifest) -> None:
        """Write the provenance record at the moment the result is stored.

        Verification later compares the live data with this file; building the
        record at verification time would compare the data with itself.
        """
        from ..security.provenance import ProvenanceEngine

        job = self.get_job(job_id)
        cfg = json.loads(job["config_json"]) if job and job["config_json"] else {}
        record = ProvenanceEngine.generate_provenance(
            job_id=job_id,
            cad_file_path=(job or {}).get("cad_file_path") or "",
            config_dict=cfg,
            results_list=[r.model_dump() for r in manifest.results],
            artifacts=list(manifest.artifacts),
            timestamp_utc=manifest.created_at,
        )
        with open(self.provenance_path(job_id), "w", encoding="utf-8") as handle:
            handle.write(record.model_dump_json(indent=2))

    def load_provenance(self, job_id: str):
        """The record written when the job finished, or ``None`` if there is none."""
        from ..security.provenance import ProvenanceRecord

        path = self.provenance_path(job_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return ProvenanceRecord.model_validate_json(handle.read())

    def add_log(self, job_id: str, level: str, message: str):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO job_logs (job_id, timestamp, level, message)
                VALUES (?, ?, ?, ?)
            """, (job_id, now, level, message))
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def get_job_logs(self, job_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, level, message FROM job_logs WHERE job_id = ? ORDER BY id ASC", (job_id,))
            return [dict(r) for r in cursor.fetchall()]

    def list_jobs(
        self,
        state: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Jobs newest first, with the total that matches the filters."""
        clauses: List[str] = []
        params: List[Any] = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if analysis_type:
            clauses.append("analysis_type = ?")
            params.append(analysis_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM jobs" + where, params).fetchone()[0]
            rows = conn.execute(
                "SELECT job_id, analysis_type, state, created_at, updated_at, cad_file_path, error_message "
                "FROM jobs" + where + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params + [int(limit), int(offset)],
            ).fetchall()
        return {"total": total, "jobs": [dict(r) for r in rows]}

    def delete_job(self, job_id: str, delete_files: bool = False) -> bool:
        """Remove a job's records; optionally its working directory too."""
        import shutil

        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not exists:
                return False
            conn.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
        if delete_files:
            job_dir = os.path.join(self.workspace_root, job_id)
            # Only ever remove a directory that really sits inside the workspace.
            if os.path.isdir(job_dir) and os.path.abspath(job_dir).startswith(
                os.path.abspath(self.workspace_root) + os.sep
            ):
                shutil.rmtree(job_dir, ignore_errors=True)
        return True

    def clone_job(self, job_id: str, new_job_id: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a CONFIGURED copy of ``job_id`` carrying ``config``; results are not copied."""
        source = self.get_job(job_id)
        if not source:
            return None
        self.create_job(new_job_id, AnalysisType(source["analysis_type"]))
        self.save_job_config(new_job_id, source.get("cad_file_path") or "", config)
        self.add_log(new_job_id, "INFO", "Kaynak job: %s" % job_id)
        return self.get_job(new_job_id)
