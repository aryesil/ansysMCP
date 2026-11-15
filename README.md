# ansysMCP

> 🇹🇷 [Türkçe README](README.tr.md)

**An MCP (Model Context Protocol) server that drives a real ANSYS 2025 R1 installation.**
An AI client such as Claude Code, Claude Desktop, Cursor or Google Antigravity can work with ANSYS Mechanical and Fluent in plain language: it reads a CAD file, sets up the analysis, solves it and checks the result.

ansysMCP runs Mechanical through PyMechanical, Fluent through PyFluent and result files through PyDPF. It covers static, modal, buckling, harmonic, transient and random-vibration structural analysis, steady thermal analysis, aerodynamic CFD, layered composite shells, mesh-convergence studies and design-of-experiments planning. Nothing is simulated or made up: when the solver is missing or a solve fails, the tool returns an error instead of a number.

> **Note:** tool descriptions and error messages are written in Turkish. The AI client translates them for you.

---

## Contents

1. [What it does and does not do](#what-it-does-and-does-not-do)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Connecting a client](#connecting-a-client)
5. [Your first analysis](#your-first-analysis)
6. [Configuration](#configuration)
7. [Tool catalogue](#tool-catalogue)
8. [Verification status](#verification-status)
9. [Design principles](#design-principles)
10. [Known limits](#known-limits)
11. [Troubleshooting](#troubleshooting)
12. [Project layout](#project-layout)
13. [License](#license)

---

## What it does and does not do

**Does**

| Area | Scope |
|------|-------|
| Structural (Mechanical) | Static, modal, linear buckling, harmonic, transient, random vibration (PSD) |
| Loads and supports | Force, pressure, moment, remote force, gravity, prescribed displacement, cylindrical support, compression-only support, extra supports and loads |
| Materials | Library cards, isotropic and orthotropic elasticity, thermal properties |
| Composites | Layered section on surface (shell) bodies (ply thickness and angle), classical laminate theory pre-check |
| Thermal | Steady-state conduction, convection, thermal stress evaluation |
| Aerodynamics (Fluent) | Virtual wind tunnel, prism boundary layer sized from a target y+, Cd / Cl / Cm, angle-of-attack sweep, sideslip, free-stream direction (by rotating a STEP body) |
| Mesh | Three-level mesh independence (GCI, ASME V&V 20), stress-singularity diagnosis, mesh-quality gate |
| Assemblies | Contact pairs (bonded, frictional, frictionless) |
| Results | Contour images (PNG), per-node CSV/VTU export, Markdown/HTML report, hand-calculation comparison |
| Design exploration | Latin Hypercube / full-factorial design of experiments and a Pareto front (it does not call the solver; it works with real responses you submit) |
| Security | Three-level AST sandbox for scripts, SHA-256 provenance signature and tamper check |

**Does not**

* Produce fake results. Without a working solver it returns an error.
* Provide an ANSYS license. You need an installed and licensed ANSYS.
* Extract a mid-surface from solid CAD. Layered composites need a surface (shell) STEP file.
* Run an ACP (Composite PrepPost) workflow or ply failure criteria.

---

## Requirements

| Component | Version / note |
|-----------|----------------|
| Operating system | Windows 10 / 11 (paths and the Mechanical and Fluent processes are Windows-specific) |
| ANSYS | **2025 R1 (v251)** is the target version; 2025 R2, 2024 R2 and 2024 R1 installs are also looked up |
| ANSYS license | A valid license for Mechanical (structural / thermal) and/or Fluent (CFD) solves |
| Python | 3.10 or newer (developed and tested on 3.11) |
| Disk | A few GB free for the install and job output (CFD meshes and result files are large) |
| Optional | SolidWorks + `pywin32` (only to convert `.sldprt` / `.sldasm` to STEP) |

The ANSYS install is found through the `AWP_ROOT251` environment variable or the standard folder `C:\Program Files\ANSYS Inc\v251`.

---

## Installation

### 1. Get the code

```powershell
git clone https://github.com/aryesil/ansysMCP.git
cd ansysMCP
```

### 2. Create a virtual environment and install dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs: `mcp`, `fastmcp`, `pydantic`, `ansys-mechanical-core` (PyMechanical), `ansys-fluent-core` (PyFluent), `ansys-dpf-core` (PyDPF), `ansys-geometry-core`, `pyvista`, `trimesh`, `cascadio` (STEP reading), `h5py`, `networkx`, `scipy`, `matplotlib` and `numpy`.

To convert SolidWorks files you also need:

```powershell
pip install pywin32
```

### 3. Check the installation

Confirm that ANSYS is found and the Python packages load:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from ansys_mcp.config import SystemConfig; print(SystemConfig.get_ansys_summary())"
```

The output should show `'detected': True` and the ANSYS version. If it shows `False`, see [Troubleshooting](#troubleshooting).

You can also start the server by hand. It speaks over stdio, so without a client it simply waits silently:

```powershell
python server.py
```

---

## Connecting a client

The server entry point is `server.py` in the repository root. The client runs it with the Python from your virtual environment. Replace the example paths below with your own folder.

> Once connected, the `get_client_configurations` tool returns ready-made configuration blocks that use the running server's own paths.

### Claude Code

```powershell
claude mcp add ansys -- C:\path\to\ansysMCP\venv\Scripts\python.exe C:\path\to\ansysMCP\server.py
```

Or put this in a project `.mcp.json`:

```json
{
  "mcpServers": {
    "ansys": {
      "type": "stdio",
      "command": "C:/path/to/ansysMCP/venv/Scripts/python.exe",
      "args": ["C:/path/to/ansysMCP/server.py"]
    }
  }
}
```

### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ansys-workbench": {
      "command": "C:/path/to/ansysMCP/venv/Scripts/python.exe",
      "args": ["C:/path/to/ansysMCP/server.py"],
      "cwd": "C:/path/to/ansysMCP"
    }
  }
}
```

### Cursor

`~/.cursor/mcp.json` (or `.cursor/mcp.json` inside a project):

```json
{
  "mcpServers": {
    "ansys": {
      "command": "C:/path/to/ansysMCP/venv/Scripts/python.exe",
      "args": ["C:/path/to/ansysMCP/server.py"],
      "cwd": "C:/path/to/ansysMCP"
    }
  }
}
```

### Google Antigravity

Use the same `command` / `args` pair in Antigravity's MCP settings.

Restart the client after changing its configuration; new tools only appear after a restart.

---

## Your first analysis

Just describe the job to the client in plain language. Example request:

> "The carbon tube in `C:\cad\tube.step` is fixed at one end and loaded with a 50 N lateral force at the other. Run a static analysis and compare the tip deflection with the closed-form solution."

The typical tool sequence behind the scenes:

1. `get_ansys_environment_info`: ANSYS and license status
2. `inspect_cad` / `inspect_geometry_details`: volume, area, bounding box, face list
3. `create_analysis`: creates a job record and returns a `job_id`
4. `configure_static_analysis`: material, mesh size, support and load (faces are named by normal or by an extreme selector such as `min_x` / `max_x`)
5. `validate_analysis`: the input gate; missing or non-physical input is rejected here
6. `submit_analysis`: the live Mechanical solve
7. `get_analysis_results`, `render_structural_contours`, `hand_calc_check`, `export_report`

Pass `show_gui=True` to watch Mechanical on screen; the window stays open and every step is visible.

For Fluent CFD the flow is `configure_aerodynamics_cfd`, then `solve_cfd_simulation` (single point) or `solve_cfd_polar` (angle-of-attack sweep).

---

## Configuration

| Setting | Meaning | Default |
|---------|---------|---------|
| `AMPC_RUNS_DIR` | Where job folders and the job database (`orchestrator.db`) live | `<repo>/runs` |
| `AWP_ROOT251` (or 252 / 242 / 241) | ANSYS install folder | Searched automatically (`C:\Program Files\ANSYS Inc\v251` ...) |

* Mechanical is driven over the local gRPC port **10000**. The server connects to a Mechanical already listening there, or starts a new one.
* There is one shared Mechanical session. Long solves do not queue: while one runs, a second one gets a "busy" error. A Workbench session opened by hand on the same machine can interfere with that shared session.
* The `runs/` folder (job output, `.rst`, `.cas.h5`, images, CAD copies) is local data and is not part of the repository.

You can set the variable in the client configuration:

```json
"env": { "AMPC_RUNS_DIR": "E:/ansys_runs" }
```

---

## Tool catalogue

The server exposes 56 tools and 9 resources.

**Session and environment**
`get_ansys_environment_info`, `open_ansys_session`, `close_ansys_session`, `get_live_session_steps`, `get_client_configurations`

**Job management**
`create_analysis`, `validate_analysis`, `list_jobs`, `get_job_status`, `clone_job`, `delete_job`, `cancel_job`

**Geometry and CAD**
`inspect_cad`, `inspect_geometry_details`, `inspect_live_geometry_faces`, `find_semantic_faces`, `render_geometry_preview`, `convert_cad_to_step` (needs SolidWorks)

**Structural analysis**
`configure_static_analysis` + `submit_analysis`,
`configure_modal_analysis` + `solve_modal_analysis`,
`configure_buckling_analysis` + `solve_buckling_analysis`,
`configure_harmonic_analysis` + `solve_harmonic_analysis`,
`configure_transient_analysis` + `solve_transient_analysis`,
`configure_random_vibration_analysis` + `solve_random_vibration_analysis`,
`configure_assembly_contacts`

**Thermal**
`configure_thermal_analysis` + `solve_thermal_analysis`, `evaluate_thermal_stresses`, `render_thermal_contours`

**Aerodynamics / CFD**
`configure_aerodynamics_cfd`, `solve_cfd_simulation`, `solve_cfd_polar`, `get_cfd_aerodynamic_coefficients`

**Mesh and accuracy**
`run_mesh_convergence_study`, `detect_stress_singularities`, `get_static_equilibrium_status`, `hand_calc_check`

**Results and reports**
`get_analysis_results`, `render_structural_contours`, `render_modal_mode_shapes`, `export_result_field`, `export_report`, `export_mechanical_script`, `evaluate_resonance_risk`

**Materials and units**
`lookup_material`, `convert_units`

**Optimization (DoE)**
`generate_optimization_design_points`, `submit_design_point_results`, `evaluate_optimization_study`, `get_pareto_optimal_designs`

**Security**
`execute_safe_python_script`, `verify_analysis_provenance`

**Resources**
`ansys://job/{job_id}/manifest`, `.../logs`, `.../validation`, `.../mesh-convergence`, `.../cfd-monitors`, `.../provenance`, `.../modal-frequencies`, `.../thermal-results`, `ansys://optimization/{study_id}/pareto-front`

Each tool's parameters and details appear as its description in the client's tool list.

---

## Verification status

The solve paths were run against a real ANSYS 2025 R1 and compared with closed-form solutions. Reference body: a 16/14 mm x 585 mm cantilevered carbon tube (E = 135 GPa).

| Analysis | Result | Reference |
|----------|--------|-----------|
| Static, 50 N at the tip | tip deflection 18.574 mm | 18.57 mm (`F L³ / 3EI`) |
| Static reaction | 50.000 N | 50 N |
| Modal | first bending frequency 79.79 Hz | 79.8 Hz |
| Transient (step load) | peak 0.7101 mm | 0.7198 mm (damped single mode) |
| Random vibration | flat-PSD response | Miles' formula, within 6 % |
| Layered shell (unidirectional) | tip deflection 15.54 mm | 15.36 mm (laminate theory) |
| Layered shell (cross-ply 0/90/90/0) | 28.52 mm | 28.47 mm |
| Layered shell in modal, buckling, harmonic, transient, PSD | agree with closed forms | thin-wall cantilever |
| CFD, reversed and cross flow | Cd inside the reference range | cylinder cross-section, Re ~ 3e4 |

These numbers were measured on the development machine; your license, mesh size and geometry can give different values. Unverified paths are listed explicitly under [Known limits](#known-limits).

---

## Design principles

* **No made-up numbers.** If the solver is not running, the license is missing or a solve fails, the tool returns an error. No default value is used to manufacture a result.
* **Explicit input.** Material, mesh size, support face, load, damping and PSD are all taken from the user. Missing or non-physical input is rejected at the `validate_analysis` stage.
* **Verifiability.** Every solve is reported with an equilibrium check (applied load + reaction = 0) and an optional hand-calculation comparison. Solved results are read from the `.rst` file.
* **Traceability.** Each job carries a provenance record with a SHA-256 signature of its inputs and results; `verify_analysis_provenance` checks for tampering.
* **Safe script execution.** `execute_safe_python_script` runs code that passes an AST policy in a separate, isolated interpreter with a time limit. The highest level needs an explicit confirmation token.

---

## Known limits

* **Windows only** is targeted.
* Layered composites need a **surface (shell) body**. There is no mid-surface extraction from solid CAD; a solid body gives an error.
* **There are no ply failure criteria and no ACP workflow.** The stress tool and safety factor are invalid on a layered section and are not added; total deformation, von Mises and the reaction are reported.
* Free-stream direction is verified live for STEP bodies. The STL/OBJ/PLY and assembly STEP paths are coded but have not been tried live with Fluent.
* CFD with Parasolid / SAT files has not been tried.
* Mechanical runs through one shared session; parallel solves are not supported.
* Tool descriptions and error messages are in Turkish.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `get_ansys_summary()` shows `detected: False` | Set `AWP_ROOT251` to the install folder (`C:\Program Files\ANSYS Inc\v251`) |
| "No live Mechanical session" | Call `open_ansys_session`, or check that the ANSYS license server is up |
| Connection lost during a solve | Another Mechanical / Workbench session may be using port 10000; close it and retry. The server opens a fresh session on the next call |
| A tool says "busy" | Another solve is running; wait for it to finish |
| Tools do not show up in the client | Check the paths to `server.py` and Python; restart the client completely |
| The first geometry call is very slow | Heavy libraries (VTK, scipy, OpenCASCADE) load at start-up; retry once the server is up |
| A CFD solve takes very long | Increase the mesh size (`mesh_size_mm`) or lower the iteration count; real Fluent solves take minutes |
| The disk fills up | Move the job folder to another drive with `AMPC_RUNS_DIR` and remove old jobs with `delete_job` |

---

## Project layout

```text
ansysMCP/
├── server.py                  # Entry point (preloads heavy libraries on the main thread)
├── pyproject.toml
├── requirements.txt
└── src/ansys_mcp/
    ├── server.py              # MCP tools and resources
    ├── config.py              # ANSYS install detection
    ├── preload.py             # Loads native libraries at start-up
    ├── adapters/              # Mechanical (live), Fluent, DPF, result export, visualisation
    ├── analysis/              # GCI, mesh study, stress singularities
    ├── cfd/                   # Wind tunnel, prism layers, volume mesh
    ├── composites/            # Classical laminate theory
    ├── domain/                # Data models, units, material library, hand calculations
    ├── geometry/              # CAD parsing, semantic face matching, STEP rotation, contacts
    ├── modal/                 # Modal engine, resonance evaluation, mode shapes
    ├── optimization/          # DoE and Pareto
    ├── orchestrator/          # Job state machine, SQLite storage, process management
    ├── security/              # Sandbox, provenance, client configuration generator
    ├── thermal/               # Thermal stress and visualisation
    └── validation/            # Input gates, load and support schemas, equilibrium check
```

---

## License

Released under the [MIT License](LICENSE). ANSYS, Mechanical, Fluent and related names are registered trademarks of ANSYS, Inc.; this project is not affiliated with or endorsed by ANSYS.
