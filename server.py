"""
ANSYS AI-MCP Server - Main Application Entrypoint
"""

import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from ansys_mcp.preload import preload_native_libraries

if __name__ == "__main__":
    # Heavy native libraries are imported here, on the main thread, because
    # their first import hangs on a worker thread of a stdio server (Windows).
    preload_native_libraries()

    from ansys_mcp.server import mcp

    mcp.run()
