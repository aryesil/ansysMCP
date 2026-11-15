"""
ANSYS AI-MCP Configuration & Environment Detector
--------------------------------------------------
Auto-detects ANSYS 2025 (2025 R1 / 2025 R2) and prior installations (2024/2023).
Configures binary paths, gRPC ports, working directories, and solver environment variables.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AnsysVersionInfo(BaseModel):
    version_str: str  # e.g. "2025 R1"
    version_code: str  # e.g. "251"
    env_var_name: str  # e.g. "AWP_ROOT251"
    root_path: str
    ansys_exe: Optional[str] = None
    mechanical_exe: Optional[str] = None
    fluent_exe: Optional[str] = None
    dpf_server_exe: Optional[str] = None


class SystemConfig:
    """Detects and provides paths for ANSYS 2025 installations."""

    # Prioritized list: 2025 R2 -> 2025 R1 -> 2024 R2 -> 2024 R1 -> 2023
    SUPPORTED_VERSIONS = [
        {"version_str": "2025 R2", "code": "252", "env": "AWP_ROOT252", "default_dir": r"C:\Program Files\ANSYS Inc\v252"},
        {"version_str": "2025 R1", "code": "251", "env": "AWP_ROOT251", "default_dir": r"C:\Program Files\ANSYS Inc\v251"},
        {"version_str": "2024 R2", "code": "242", "env": "AWP_ROOT242", "default_dir": r"C:\Program Files\ANSYS Inc\v242"},
        {"version_str": "2024 R1", "code": "241", "env": "AWP_ROOT241", "default_dir": r"C:\Program Files\ANSYS Inc\v241"},
    ]

    @classmethod
    def detect_ansys_installation(cls) -> Optional[AnsysVersionInfo]:
        """
        Scans environment variables and standard Program Files paths for ANSYS 2025/2024.
        """
        for v in cls.SUPPORTED_VERSIONS:
            root = os.environ.get(v["env"])
            if not root or not os.path.exists(root):
                if os.path.exists(v["default_dir"]):
                    root = v["default_dir"]

            if root and os.path.exists(root):
                code = v["code"]
                ansys_exe = os.path.join(root, "ansys", "bin", "win64", f"ansys{code}.exe")
                mech_exe = os.path.join(root, "aisol", "bin", "winx64", "AnsysWBU.exe")
                fluent_exe = os.path.join(root, "fluent", "ntbin", "win64", "fluent.exe")
                dpf_exe = os.path.join(root, "dpf", "bin", "winx64", "Ans.Dpf.Grpc.bat")

                return AnsysVersionInfo(
                    version_str=v["version_str"],
                    version_code=code,
                    env_var_name=v["env"],
                    root_path=root,
                    ansys_exe=ansys_exe if os.path.exists(ansys_exe) else None,
                    mechanical_exe=mech_exe if os.path.exists(mech_exe) else None,
                    fluent_exe=fluent_exe if os.path.exists(fluent_exe) else None,
                    dpf_server_exe=dpf_exe if os.path.exists(dpf_exe) else None
                )
        return None

    @classmethod
    def get_ansys_summary(cls) -> Dict[str, Any]:
        """Returns detected environment status for MCP tool inspection."""
        info = cls.detect_ansys_installation()
        if info:
            return {
                "detected": True,
                "version": info.version_str,
                "version_code": info.version_code,
                "root_path": info.root_path,
                "mechanical_ready": info.mechanical_exe is not None,
                "fluent_ready": info.fluent_exe is not None,
                "dpf_ready": info.dpf_server_exe is not None,
            }
        else:
            return {
                "detected": False,
                "version": "Not Found",
                "target_version": "2025 R1 (v251)",
                "note": "ANSYS kurulumu bulunamadı; analiz araçları çalıştırılamaz. ANSYS 2025 R1 kurulduğunda otomatik olarak algılanacaktır."
            }
