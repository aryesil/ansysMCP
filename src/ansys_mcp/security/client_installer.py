"""
Client Configuration Packages Generator
---------------------------------------
Generates ready-to-use MCP configuration files for Claude Code, Antigravity, and Cursor.
The interpreter and project folder are the ones this server is running from, so the
generated entries work on any machine without editing paths.
"""

import os
import sys
from typing import Dict, Any


class ClientConfigGenerator:
    """Generates standard MCP client connection configs."""

    @staticmethod
    def get_python_path() -> str:
        return os.path.abspath(sys.executable).replace("\\", "/")

    @staticmethod
    def get_project_root() -> str:
        here = os.path.abspath(__file__)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
        return root.replace("\\", "/")

    @classmethod
    def generate_all_configs(cls) -> Dict[str, Any]:
        py_exe = cls.get_python_path()
        root = cls.get_project_root()
        entry = f"{root}/server.py"

        claude_config = {
            "mcpServers": {
                "ansys-workbench": {"command": py_exe, "args": [entry], "cwd": root}
            }
        }
        antigravity_config = {
            "mcpServers": {
                "ansys": {"command": py_exe, "args": [entry], "cwd": root}
            }
        }
        cursor_config = {
            "mcp": {
                "servers": [
                    {"name": "ansys-mcp", "command": py_exe, "args": [entry], "cwd": root}
                ]
            }
        }

        return {
            "claude_desktop_config": claude_config,
            "antigravity_config": antigravity_config,
            "cursor_config": cursor_config,
        }
