"""Conditional parameter recovery for documented Cameroonian deposit models.

For each scenario, synthetic DC/IP data are generated from known parameters,
Gaussian noise is added, and parameters are recovered on controlled candidate
grids.  Detection is not the objective: the output quantifies how closely the
synthetic truth can be estimated and supports comparison with field literature.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common_model import OUTPUT_DIR, P, RANDOM_SEED
from sensitivity_analysis import (
    BASE_LATERITE_THICKNESS,
    BASE_RHO_LATERITE,
    build_dc_ip_mesh_and_surveys,
    run_dc_ip_variant,
)
from uncertainty import add_gaussian_noise, normalized_rms


SCENARIO_FILE = Path(__file__).resolve().with_name("model_scenarios.json")
N_REALIZATIONS = 20

SEARCH_GRIDS = {
    "gold_sulphide": {
        "depths": [10.0, 20.0, 25.0, 30.0],
        "widths": [50.0, 80.0, 110.0],
        "resistivities": [300.0, 600.0, 900.0],
        "chargeabilities": [0.030, 0.050, 0.0671],
    },
    "iron_channel": {
        "depths": [5.0, 15.0, 20.0, 30.0],
        "widths": [100.0, 150.0, 200.0],
        "resistivities": [750.0, 1500.0, 2500.0],
        "chargeabilities": [0.016, 0.018, 0.025],
    },
}


def _predict(setup, scenario, **updates):
    parameters = {
        "laterite_thickness": BASE_LATERITE_THICKNESS,
        "laterite_resistivity": BASE_RHO_LATERITE,
        "target_top": scenario["target_top_depth_m"],
        "target_width": scenario["target_width_m"],
        "target_height": scenario["target_height_m"],
        "target_resistivity": scenario["target_resistivity_ohm_m"],
        "target_chargeability": scenario["target_chargeability"],
    }
    parameters.update(updates)
    return run_dc_ip_variant(*setup, **parameters)


def _best_geometry(library, observed_dc, observed_ip, dc_std, ip_std):
    scores = []
    for depth, width, predicted_dc, predicted_ip in library:
        dc_score = normalized_rms(predicted_dc - observed_dc, dc_std)
        ip_score = normalized_rms(predicted_ip - observed_ip, ip_std)
        scores.append((np.hypot(dc_score, ip_score) / np.sqrt(2.0), depth, width))
    return min(scores)


def _best_property(library, observed, std):
    return min(
        (normalized_rms(predicted - observed, std), value)
        for value, predicted in library
    )


def _relative_error(estimate, truth):
    return 100.0 * (float(estimate) - float(truth)) / float(truth)


def run_scenario(name, scenario, setup):
    grid = SEARCH_GRIDS[name]
    true_dc, true_ip = _predict(setup, scenario)

    geometry_library = []
    for depth in grid["depths"]:
        for width in grid["widths"]:
            dc_data, ip_data = _predict(
                setup, scenario, target_top=depth, target_width=width
            )
            geometry_library.append((depth, width, dc_data, ip_data))

    resistivity_library = []
    for resistivity in grid["resistivities"]:
        dc_data, _ = _predict(setup, scenario, target_resistivity=resistivity)
        resistivity_library.append((resistivity, dc_data))

    chargeability_library = []
    for chargeability in grid["chargeabilities"]:
        _, ip_data = _predict(setup, scenario, target_chargeability=chargeability)
        chargeability_library.append((chargeability, ip_data))

    rows = []
    scenario_offset = 1000 * list(SEARCH_GRIDS).index(name)
    for realization in range(N_REALIZATIONS):
        observed_dc, dc_std = add_gaussian_noise(
            true_dc,
            float(P["dc_relative_error"]),
            float(P["dc_error_floor_ohm_m"]),
            RANDOM_SEED + scenario_offset + realization,
        )
        observed_ip, ip_std = add_gaussian_noise(
            true_ip,
            float(P["ip_relative_error"]),
            float(P["ip_error_floor_v_v"]),
            RANDOM_SEED + 5000 + scenario_offset + realization,
        )

        geometry_score, estimated_depth, estimated_width = _best_geometry(
            geometry_library, observed_dc, observed_ip, dc_std, ip_std
        )
        rho_score, estimated_resistivity = _best_property(
            resistivity_library, observed_dc, dc_std
        )
        eta_score, estimated_chargeability = _best_property(
            chargeability_library, observed_ip, ip_std
        )
        rows.append({
            "scenario": name,
            "realization": realization,
            "estimated_top_depth_m": estimated_depth,
            "estimated_width_m": estimated_width,
            "estimated_resistivity_ohm_m": estimated_resistivity,
            "estimated_chargeability_v_v": estimated_chargeability,
            "geometry_nrms": geometry_score,
            "resistivity_nrms": rho_score,
            "chargeability_nrms": eta_score,
        })

    return rows


def _summarize(scenario, rows):
    definitions = [
        ("top_depth_m", "estimated_top_depth_m", scenario["target_top_depth_m"]),
        ("width_m", "estimated_width_m", scenario["target_width_m"]),
        ("resistivity_ohm_m", "estimated_resistivity_ohm_m", scenario["target_resistivity_ohm_m"]),
        ("chargeability_v_v", "estimated_chargeability_v_v", scenario["target_chargeability"]),
    ]
    summary = {}
    for label, key, truth in definitions:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[label] = {
            "true": float(truth),
            "median_estimate": float(np.median(values)),
            "p10_estimate": float(np.percentile(values, 10)),
            "p90_estimate": float(np.percentile(values, 90)),
            "median_relative_error_percent": float(_relative_error(np.median(values), truth)),
            "exact_grid_recovery_fraction": float(np.mean(values == float(truth))),
        }
    return {
        "label": scenario["label"],
        "field_sites": scenario["field_sites"],
        "conditional_parameters": summary,
        "number_of_noise_realizations": len(rows),
    }


def _write_plot(summary):
    labels = []
    errors = []
    colors = []
    palette = {"gold_sulphide": "#2d6a4f", "iron_channel": "#b5651d"}
    for scenario_name, scenario_summary in summary.items():
        for parameter, stats in scenario_summary["conditional_parameters"].items():
            labels.append(f"{scenario_name}\n{parameter}")
            errors.append(abs(stats["median_relative_error_percent"]))
            colors.append(palette[scenario_name])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(np.arange(len(labels)), errors, color=colors)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Absolute median recovery error (%)")
    ax.set_title("Synthetic parameter recovery for Cameroonian deposit models")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dcip_parameter_recovery.png", dpi=200)
    plt.close(fig)


def main():
    scenarios = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
    setup = build_dc_ip_mesh_and_surveys(cell_size=float(P["surface_cell_size_m"]))
    all_rows = []
    summary = {}

    for name, scenario in scenarios.items():
        rows = run_scenario(name, scenario, setup)
        all_rows.extend(rows)
        summary[name] = _summarize(scenario, rows)

    csv_path = OUTPUT_DIR / "dcip_parameter_recovery.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = OUTPUT_DIR / "dcip_parameter_recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plot(summary)
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
