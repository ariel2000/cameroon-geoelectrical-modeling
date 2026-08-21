"""Run the complete reproducible synthetic workflow from the repository root."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import argparse
from pathlib import Path

import discretize
import matplotlib
import numpy
import scipy
import simpeg


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "plot_reference_model.py",
    "dc_forward_2d.py",
    "ip_forward_2d.py",
    "mesh_convergence.py",
    "sensitivity_analysis.py",
    "dcip_parameter_recovery.py",
    "petroleum_parameter_recovery.py",
    "compare_with_field.py",
]


def write_metadata():
    parameter_files = [
        ROOT / "scripts" / "model_parameters.json",
        ROOT / "scripts" / "model_scenarios.json",
        ROOT / "scripts" / "petroleum_scenario.json",
    ]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "discretize": discretize.__version__,
        "simpeg": simpeg.__version__,
        "parameter_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in parameter_files
        },
        "scripts": SCRIPTS,
    }
    output = ROOT / "outputs" / "run_metadata.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Refresh provenance metadata without rerunning simulations.",
    )
    args = parser.parse_args()

    if not args.metadata_only:
        for script in SCRIPTS:
            print(f"\n=== Running {script} ===", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script)],
                cwd=ROOT,
                check=True,
            )

    output = write_metadata()
    print(f"\nWorkflow complete. Saved {output}")


if __name__ == "__main__":
    main()
