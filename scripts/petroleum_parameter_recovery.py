"""Synthetic resistivity-log recovery for a Cameroonian oil/gas reservoir.

This particular case uses the electrical deep-resistivity log because the
documented Douala and Rio del Rey reservoirs lie far below the practical
investigation depth of surface DC/IP.  The workflow recovers reservoir top,
thickness and formation resistivity, then derives water saturation with a
shaly-sand Simandoux relation.  It does not identify oil versus gas by
resistivity alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
SCENARIO_FILE = Path(__file__).resolve().with_name("petroleum_scenario.json")
N_REALIZATIONS = 20
RANDOM_SEED = 20260821

SEARCH_GRID = {
    "top_depths_m": [4896.0, 4900.0, 4902.0],
    "thicknesses_m": [32.0, 36.0, 38.0],
    "resistivities_ohm_m": [2.8, 3.5, 4.2],
}


def simandoux_water_saturation(
    formation_resistivity,
    porosity,
    shale_volume,
    water_resistivity,
    shale_resistivity,
    archie_a=1.0,
    archie_m=2.0,
    saturation_exponent=2.0,
):
    """Return Sw from a simplified Simandoux equation for n=2."""
    if not np.isclose(saturation_exponent, 2.0):
        raise ValueError("The analytic solver currently requires n=2")
    values = (
        formation_resistivity,
        porosity,
        water_resistivity,
        shale_resistivity,
        archie_a,
    )
    if any(float(value) <= 0.0 for value in values):
        raise ValueError("Resistivities, porosity and Archie a must be positive")
    if not 0.0 <= float(shale_volume) < 1.0:
        raise ValueError("shale_volume must be in [0, 1)")

    coefficient_a = porosity**archie_m / (archie_a * water_resistivity)
    coefficient_b = shale_volume / shale_resistivity
    discriminant = coefficient_b**2 + 4.0 * coefficient_a / formation_resistivity
    saturation = (-coefficient_b + np.sqrt(discriminant)) / (2.0 * coefficient_a)
    return float(np.clip(saturation, 0.0, 1.0))


def synthetic_log(depths, top_depth, thickness, reservoir_rho, background_rho):
    values = np.full_like(depths, float(background_rho), dtype=float)
    reservoir = (depths >= top_depth) & (depths <= top_depth + thickness)
    values[reservoir] = float(reservoir_rho)
    return values


def add_noise(values, relative_noise, floor, seed):
    standard_deviation = np.sqrt((relative_noise * values) ** 2 + floor**2)
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, standard_deviation), standard_deviation


def normalized_rms(residual, standard_deviation):
    return float(np.sqrt(np.mean((residual / standard_deviation) ** 2)))


def recover_one(observed, standard_deviation, depths, scenario):
    candidates = []
    for top in SEARCH_GRID["top_depths_m"]:
        for thickness in SEARCH_GRID["thicknesses_m"]:
            for resistivity in SEARCH_GRID["resistivities_ohm_m"]:
                predicted = synthetic_log(
                    depths,
                    top,
                    thickness,
                    resistivity,
                    scenario["background_resistivity_ohm_m"],
                )
                candidates.append(
                    (normalized_rms(predicted - observed, standard_deviation), top, thickness, resistivity)
                )
    score, top, thickness, resistivity = min(candidates)
    water_saturation = simandoux_water_saturation(
        resistivity,
        scenario["effective_porosity"],
        scenario["shale_volume"],
        scenario["formation_water_resistivity_ohm_m"],
        scenario["shale_resistivity_ohm_m"],
        scenario["archie_a"],
        scenario["archie_m"],
        scenario["saturation_exponent_n"],
    )
    return score, top, thickness, resistivity, water_saturation


def relative_error(estimate, truth):
    return 100.0 * (float(estimate) - float(truth)) / float(truth)


def summarize(rows, scenario):
    true_sw = simandoux_water_saturation(
        scenario["formation_resistivity_ohm_m"],
        scenario["effective_porosity"],
        scenario["shale_volume"],
        scenario["formation_water_resistivity_ohm_m"],
        scenario["shale_resistivity_ohm_m"],
        scenario["archie_a"],
        scenario["archie_m"],
        scenario["saturation_exponent_n"],
    )
    definitions = [
        ("top_depth_m", "estimated_top_depth_m", scenario["reservoir_top_depth_m"]),
        ("thickness_m", "estimated_thickness_m", scenario["reservoir_thickness_m"]),
        ("resistivity_ohm_m", "estimated_resistivity_ohm_m", scenario["formation_resistivity_ohm_m"]),
        ("water_saturation", "estimated_water_saturation", true_sw),
    ]
    parameters = {}
    for label, key, truth in definitions:
        values = np.asarray([row[key] for row in rows], dtype=float)
        median = float(np.median(values))
        parameters[label] = {
            "true": float(truth),
            "median_estimate": median,
            "p10_estimate": float(np.percentile(values, 10)),
            "p90_estimate": float(np.percentile(values, 90)),
            "median_relative_error_percent": relative_error(median, truth),
        }
    parameters["top_depth_m"]["median_absolute_error_m"] = (
        parameters["top_depth_m"]["median_estimate"] - parameters["top_depth_m"]["true"]
    )
    return {
        "label": scenario["label"],
        "field_sites": scenario["field_sites"],
        "method": "deep_resistivity_log_and_Simandoux",
        "conditional_parameters": parameters,
        "fixed_parameters": {
            "effective_porosity": scenario["effective_porosity"],
            "shale_volume": scenario["shale_volume"],
            "formation_water_resistivity_ohm_m": scenario["formation_water_resistivity_ohm_m"],
        },
        "number_of_noise_realizations": len(rows),
        "interpretive_limit": "resistivity alone does not distinguish oil from gas",
    }


def write_plot(depths, truth, observed, summary, scenario):
    stats = summary["conditional_parameters"]
    recovered = synthetic_log(
        depths,
        stats["top_depth_m"]["median_estimate"],
        stats["thickness_m"]["median_estimate"],
        stats["resistivity_ohm_m"]["median_estimate"],
        scenario["background_resistivity_ohm_m"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axes[0].plot(truth, depths, color="black", linewidth=2, label="Synthetic truth")
    axes[0].scatter(observed, depths, s=6, color="#457b9d", alpha=0.5, label="Noisy log")
    axes[0].plot(recovered, depths, color="#d95f02", linewidth=2, linestyle="--", label="Median estimate")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Deep resistivity (ohm m)")
    axes[0].set_ylabel("Measured depth (m)")
    axes[0].set_title("Rio del Rey synthetic resistivity log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    names = ["Top\ndepth", "Thickness", "Resistivity", "Water\nsaturation"]
    errors = [
        abs(stats["top_depth_m"]["median_absolute_error_m"]) / scenario["reservoir_thickness_m"] * 100.0,
        abs(stats["thickness_m"]["median_relative_error_percent"]),
        abs(stats["resistivity_ohm_m"]["median_relative_error_percent"]),
        abs(stats["water_saturation"]["median_relative_error_percent"]),
    ]
    bars = axes[1].bar(names, errors, color="#264653")
    axes[1].set_ylabel("Error (%)")
    axes[1].set_title("Conditional recovery error")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, errors):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "petroleum_parameter_recovery.png", dpi=200)
    plt.close(fig)


def main():
    scenario = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
    depths = np.arange(
        scenario["log_depth_start_m"],
        scenario["log_depth_end_m"] + scenario["log_sample_interval_m"] / 2.0,
        scenario["log_sample_interval_m"],
    )
    truth = synthetic_log(
        depths,
        scenario["reservoir_top_depth_m"],
        scenario["reservoir_thickness_m"],
        scenario["formation_resistivity_ohm_m"],
        scenario["background_resistivity_ohm_m"],
    )

    rows = []
    first_observed = None
    for realization in range(N_REALIZATIONS):
        observed, standard_deviation = add_noise(
            truth,
            scenario["relative_noise"],
            scenario["absolute_noise_floor_ohm_m"],
            RANDOM_SEED + realization,
        )
        if first_observed is None:
            first_observed = observed
        score, top, thickness, resistivity, water_saturation = recover_one(
            observed, standard_deviation, depths, scenario
        )
        rows.append({
            "realization": realization,
            "estimated_top_depth_m": top,
            "estimated_thickness_m": thickness,
            "estimated_resistivity_ohm_m": resistivity,
            "estimated_water_saturation": water_saturation,
            "hydrocarbon_saturation": 1.0 - water_saturation,
            "nrms": score,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "petroleum_parameter_recovery.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, scenario)
    summary_path = OUTPUT_DIR / "petroleum_parameter_recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_plot(depths, truth, first_observed, summary, scenario)
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
