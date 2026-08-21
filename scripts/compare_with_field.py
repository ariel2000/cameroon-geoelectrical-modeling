"""Compare recovered synthetic parameters with published Cameroon intervals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common_model import OUTPUT_DIR, ROOT


SUMMARY = OUTPUT_DIR / "dcip_parameter_recovery_summary.json"
PETROLEUM_SUMMARY = OUTPUT_DIR / "petroleum_parameter_recovery_summary.json"
SCENARIOS = Path(__file__).resolve().with_name("model_scenarios.json")
BENCHMARKS = ROOT / "data" / "field_benchmarks.csv"


def _read_benchmarks():
    with BENCHMARKS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _inside(value, lower, upper, role=""):
    if role == "reference_value":
        reference = float(lower)
        return abs(value - reference) / reference <= 0.10
    if lower != "" and value < float(lower):
        return False
    if upper != "" and value > float(upper):
        return False
    return True


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    petroleum_summary = json.loads(PETROLEUM_SUMMARY.read_text(encoding="utf-8"))
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    benchmarks = _read_benchmarks()

    comparisons = [
        ("gold_sulphide", "Bindiba", "target_top_depth", "top_depth_m", "estimated"),
        ("gold_sulphide", "Bindiba", "target_resistivity", "resistivity_ohm_m", "estimated"),
        ("gold_sulphide", "Bindiba", "target_chargeability", "chargeability_v_v", "estimated"),
        ("iron_channel", "Messondo", "target_resistivity", "resistivity_ohm_m", "estimated"),
        ("iron_channel", "Messondo", "target_chargeability", "chargeability_v_v", "estimated"),
        ("iron_channel", "Messondo", "overburden_thickness", "top_depth_m", "estimated"),
        ("iron_channel", "Messondo", "mineralized_thickness", "target_height_m", "fixed"),
    ]

    rows = []
    for scenario_name, site, field_parameter, synthetic_parameter, status in comparisons:
        benchmark = next(
            row for row in benchmarks
            if row["site"] == site and row["parameter"] == field_parameter
        )
        if status == "estimated":
            stats = summary[scenario_name]["conditional_parameters"][synthetic_parameter]
            truth = stats["true"]
            estimate = stats["median_estimate"]
            error = stats["median_relative_error_percent"]
        else:
            truth = float(scenarios[scenario_name][synthetic_parameter])
            estimate = truth
            error = 0.0
        rows.append({
            "scenario": scenario_name,
            "site": site,
            "parameter": field_parameter,
            "estimation_status": status,
            "synthetic_true_value": truth,
            "median_estimated_value": estimate,
            "relative_error_percent": error,
            "field_lower_bound": benchmark["lower_bound"],
            "field_upper_bound": benchmark["upper_bound"],
            "unit": benchmark["unit"],
            "field_role": benchmark["role"],
            "estimate_consistent_with_field": _inside(
                estimate, benchmark["lower_bound"], benchmark["upper_bound"], benchmark["role"]
            ),
            "source": benchmark["source"],
        })

    petroleum_comparisons = [
        ("RioDelRey", "reservoir_top_depth", "top_depth_m"),
        ("RioDelRey", "reservoir_thickness", "thickness_m"),
        ("RioDelRey", "formation_resistivity", "resistivity_ohm_m"),
        ("RioDelRey", "water_saturation", "water_saturation"),
    ]
    for site, field_parameter, synthetic_parameter in petroleum_comparisons:
        benchmark = next(
            row for row in benchmarks
            if row["site"] == site and row["parameter"] == field_parameter
        )
        stats = petroleum_summary["conditional_parameters"][synthetic_parameter]
        estimate = stats["median_estimate"]
        rows.append({
            "scenario": "petroleum_gas_reservoir",
            "site": site,
            "parameter": field_parameter,
            "estimation_status": "estimated",
            "synthetic_true_value": stats["true"],
            "median_estimated_value": estimate,
            "relative_error_percent": stats["median_relative_error_percent"],
            "field_lower_bound": benchmark["lower_bound"],
            "field_upper_bound": benchmark["upper_bound"],
            "unit": benchmark["unit"],
            "field_role": benchmark["role"],
            "estimate_consistent_with_field": _inside(
                estimate, benchmark["lower_bound"], benchmark["upper_bound"], benchmark["role"]
            ),
            "source": benchmark["source"],
        })

    output = OUTPUT_DIR / "field_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
