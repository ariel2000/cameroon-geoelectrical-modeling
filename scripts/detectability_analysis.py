"""Estimate target anomaly-to-noise ratios as a function of depth."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from common_model import OUTPUT_DIR, P
from sensitivity_analysis import (
    BASE_LATERITE_THICKNESS,
    BASE_RHO_LATERITE,
    BASE_TARGET_HEIGHT,
    BASE_TARGET_WIDTH,
    build_dc_ip_mesh_and_surveys,
    run_dc_ip_variant,
    run_tdem_variant,
)
from uncertainty import normalized_rms, standard_deviation


def main():
    depths = np.array([60.0, 80.0, 100.0, 140.0, 180.0])
    setup = build_dc_ip_mesh_and_surveys(cell_size=10.0)
    common = dict(
        laterite_thickness=BASE_LATERITE_THICKNESS,
        laterite_resistivity=BASE_RHO_LATERITE,
        target_top=100.0,
        target_width=BASE_TARGET_WIDTH,
        target_height=BASE_TARGET_HEIGHT,
    )
    dc_background, ip_background = run_dc_ip_variant(
        *setup, **common, include_target=False
    )
    _, tdem_background = run_tdem_variant(
        laterite_thickness=BASE_LATERITE_THICKNESS,
        laterite_resistivity=BASE_RHO_LATERITE,
        target_top=100.0,
        target_height=BASE_TARGET_HEIGHT,
        include_target=False,
    )

    dc_std = standard_deviation(
        dc_background, float(P["dc_relative_error"]),
        float(P["dc_error_floor_ohm_m"]),
    )
    ip_std = standard_deviation(
        ip_background, float(P["ip_relative_error"]),
        float(P["ip_error_floor_v_v"]),
    )
    tdem_std = standard_deviation(
        tdem_background, float(P["tdem_relative_error"]),
        float(P["tdem_error_floor_t_s"]),
    )

    rows = []
    for depth in depths:
        dc_target, ip_target = run_dc_ip_variant(
            *setup, **{**common, "target_top": depth}
        )
        _, tdem_target = run_tdem_variant(
            laterite_thickness=BASE_LATERITE_THICKNESS,
            laterite_resistivity=BASE_RHO_LATERITE,
            target_top=depth,
            target_height=BASE_TARGET_HEIGHT,
        )
        rows.append([
            depth,
            normalized_rms(dc_target - dc_background, dc_std),
            normalized_rms(ip_target - ip_background, ip_std),
            normalized_rms(tdem_target - tdem_background, tdem_std),
        ])

    path = OUTPUT_DIR / "detectability_by_depth.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["target_top_depth_m", "dc_rms_snr", "ip_rms_snr", "tdem_rms_snr"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for column, label, marker in [(1, "DC", "o"), (2, "IP", "s"), (3, "TDEM 1D equivalent", "^")]:
        ax.semilogy(depths, [row[column] for row in rows], marker + "-", label=label)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="RMS SNR = 1")
    ax.set_xlabel("Target top depth (m)")
    ax.set_ylabel("RMS anomaly-to-noise ratio")
    ax.set_title("Synthetic detectability under declared noise model")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "detectability_by_depth.png", dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
