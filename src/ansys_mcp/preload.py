"""
Native library preload
----------------------
The server answers tool calls from worker threads. Several libraries the tools
need (scipy's BLAS bindings, VTK, h5py, the OpenCASCADE bindings) load compiled
extensions the first time they are imported, and on Windows that first import
can block forever when it happens on a worker thread of a stdio server: the
first call to a geometry tool never returned, while the same import on the
main thread takes under a second.

Importing them here, on the main thread before the server starts listening,
moves that cost to start-up and keeps every later call a plain cache hit.
"""

import importlib
import sys
import time
from typing import Dict

# Ordered so that dependencies load before the packages built on them.
_MODULES = (
    "numpy",
    "scipy.linalg",
    "scipy.spatial",
    "scipy.sparse",
    "networkx",
    "trimesh",
    "cascadio",
    "h5py",
    "matplotlib",
    "matplotlib.pyplot",
    "pyvista",
    "ansys.dpf.core",
)


def preload_native_libraries() -> Dict[str, str]:
    """Import every heavy dependency now; return a status string per module.

    A module that is missing or fails to import is reported, not fatal: the
    tool that needs it will raise its own clear error when it is called.
    """
    # Select the non-interactive backend before pyplot is touched.
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass

    report: Dict[str, str] = {}
    for name in _MODULES:
        started = time.time()
        try:
            importlib.import_module(name)
            report[name] = "ok %.1fs" % (time.time() - started)
        except Exception as exc:  # noqa: BLE001 - any import failure is reported the same way
            report[name] = "unavailable: %s" % exc
    # stdout is the MCP protocol channel; diagnostics go to stderr only.
    print("[ansys-mcp] preload: " + ", ".join("%s=%s" % kv for kv in report.items()), file=sys.stderr)
    return report
