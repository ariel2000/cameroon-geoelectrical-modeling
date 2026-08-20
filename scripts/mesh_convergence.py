"""Quantify DC/IP response convergence on nested adaptive meshes."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from common_model import OUTPUT_DIR
from sensitivity_analysis import (
    BASE_LATERITE_THICKNESS,
    BASE_RHO_LATERITE,
    BASE_TARGET_HEIGHT,
    BASE_TARGET_TOP,
    BASE_TARGET_WIDTH,
    build_dc_ip_mesh_and_surveys,
    relative_rms_change,
    run_dc_ip_variant,
)


def main():
    cell_sizes = [20.0, 10.0, 5.0, 2.5]
    responses = []
    rows = []

    for cell_size in cell_sizes:
        setup = build_dc_ip_mesh_and_surveys(cell_size=cell_size)
        mesh = setup[0]
        dc_data, ip_data = run_dc_ip_variant(
            *setup,
            laterite_thickness=BASE_LATERITE_THICKNESS,
            laterite_resistivity=BASE_RHO_LATERITE,
            target_top=BASE_TARGET_TOP,
            target_width=BASE_TARGET_WIDTH,
            target_height=BASE_TARGET_HEIGHT,
        )
        responses.append((cell_size, dc_data, ip_data))
        rows.append([cell_size, mesh.nC, np.nan, np.nan])

    finest_dc = responses[-1][1]
    finest_ip = responses[-1][2]
    for index, (_, dc_data, ip_data) in enumerate(responses):
        rows[index][2] = relative_rms_change(finest_dc, dc_data)
        rows[index][3] = relative_rms_change(finest_ip, ip_data)

    csv_path = OUTPUT_DIR / "mesh_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "minimum_cell_size_m", "mesh_cells",
            "dc_relative_rms_vs_2p5m_percent", "ip_relative_rms_vs_2p5m_percent",
        ])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = np.asarray([row[0] for row in rows])
    ax.plot(sizes, [row[2] for row in rows], "o-", label="DC")
    ax.plot(sizes, [row[3] for row in rows], "s-", label="IP")
    ax.set_xlabel("Minimum cell size (m)")
    ax.set_ylabel("Relative RMS difference from 2.5 m mesh (%)")
    ax.set_title("Nested-mesh convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mesh_convergence.png", dpi=200)
    plt.close(fig)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
