import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _benchmarks():
    with (ROOT / "data" / "field_benchmarks.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _inside(value, row):
    return (
        (row["lower_bound"] == "" or value >= float(row["lower_bound"]))
        and (row["upper_bound"] == "" or value <= float(row["upper_bound"]))
    )


def test_reference_scenarios_are_anchored_to_field_intervals():
    scenarios = json.loads(
        (ROOT / "scripts" / "model_scenarios.json").read_text(encoding="utf-8")
    )
    rows = _benchmarks()
    checks = [
        ("gold_sulphide", "target_top_depth_m", "Bindiba", "target_top_depth"),
        ("gold_sulphide", "target_resistivity_ohm_m", "Bindiba", "target_resistivity"),
        ("gold_sulphide", "target_chargeability", "Bindiba", "target_chargeability"),
        ("iron_channel", "target_resistivity_ohm_m", "Messondo", "target_resistivity"),
        ("iron_channel", "target_chargeability", "Messondo", "target_chargeability"),
        ("iron_channel", "target_top_depth_m", "Messondo", "overburden_thickness"),
        ("iron_channel", "target_height_m", "Messondo", "mineralized_thickness"),
    ]
    for scenario, key, site, parameter in checks:
        benchmark = next(
            row for row in rows if row["site"] == site and row["parameter"] == parameter
        )
        assert _inside(float(scenarios[scenario][key]), benchmark)


def test_field_benchmarks_have_traceable_sources():
    assert all(row["source"] for row in _benchmarks())


def test_petroleum_scenario_is_anchored_to_rio_del_rey_data():
    scenario = json.loads(
        (ROOT / "scripts" / "petroleum_scenario.json").read_text(encoding="utf-8")
    )
    rows = _benchmarks()
    checks = [
        ("reservoir_top_depth_m", "reservoir_top_depth"),
        ("reservoir_thickness_m", "reservoir_thickness"),
        ("formation_resistivity_ohm_m", "formation_resistivity"),
        ("effective_porosity", "effective_porosity"),
    ]
    for key, parameter in checks:
        benchmark = next(
            row for row in rows
            if row["site"] == "RioDelRey" and row["parameter"] == parameter
        )
        assert _inside(float(scenario[key]), benchmark)
