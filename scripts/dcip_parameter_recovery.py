"""Controlled grid-search recovery of 2D target depth and width from DC/IP."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from common_model import OUTPUT_DIR, P, RANDOM_SEED
from sensitivity_analysis import (
    BASE_LATERITE_THICKNESS,
    BASE_RHO_LATERITE,
    BASE_TARGET_HEIGHT,
    build_dc_ip_mesh_and_surveys,
    run_dc_ip_variant,
)
from uncertainty import add_gaussian_noise, normalized_rms


def main():
    setup = build_dc_ip_mesh_and_surveys(cell_size=10.0)
    fixed = dict(
        laterite_thickness=BASE_LATERITE_THICKNESS,
        laterite_resistivity=BASE_RHO_LATERITE,
        target_height=BASE_TARGET_HEIGHT,
    )

    true_depth, true_width = 95.0, 70.0
    true_dc, true_ip = run_dc_ip_variant(
        *setup, **fixed, target_top=true_depth, target_width=true_width
    )
    observed_dc, dc_std = add_gaussian_noise(
        true_dc, float(P["dc_relative_error"]),
        float(P["dc_error_floor_ohm_m"]), RANDOM_SEED,
    )
    observed_ip, ip_std = add_gaussian_noise(
        true_ip, float(P["ip_relative_error"]),
        float(P["ip_error_floor_v_v"]), RANDOM_SEED + 1,
    )

    depths = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
    widths = np.array([40.0, 60.0, 80.0, 100.0, 120.0])
    objective = np.empty((depths.size, widths.size))
    rows = []

    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            predicted_dc, predicted_ip = run_dc_ip_variant(
                *setup, **fixed, target_top=depth, target_width=width
            )
            dc_nrms = normalized_rms(predicted_dc - observed_dc, dc_std)
            ip_nrms = normalized_rms(predicted_ip - observed_ip, ip_std)
            combined = np.sqrt((dc_nrms**2 + ip_nrms**2) / 2.0)
            objective[i, j] = combined
            rows.append([depth, width, dc_nrms, ip_nrms, combined])

    best = np.unravel_index(np.argmin(objective), objective.shape)
    best_depth, best_width = depths[best[0]], widths[best[1]]

    path = OUTPUT_DIR / "dcip_parameter_recovery.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["depth_m", "width_m", "dc_nrms", "ip_nrms", "combined_nrms"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        objective, origin="lower", aspect="auto",
        extent=[widths[0], widths[-1], depths[0], depths[-1]],
    )
    ax.scatter(true_width, true_depth, marker="*", s=130, c="white", label="Synthetic truth")
    ax.scatter(best_width, best_depth, marker="x", s=90, c="red", label="Grid estimate")
    ax.set_xlabel("Target width (m)")
    ax.set_ylabel("Target top depth (m)")
    ax.set_title("Joint DC/IP parametric recovery")
    fig.colorbar(image, ax=ax, label="Combined normalized RMS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dcip_parameter_recovery.png", dpi=200)
    plt.close(fig)
    print(f"Best grid estimate: top={best_depth:.0f} m, width={best_width:.0f} m")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
