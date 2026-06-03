from __future__ import annotations

"""
Sensitivity analysis for DC, IP and TDEM synthetic responses.

This script is intended for Chapter 4 of the report:
- influence of the near-surface layer thickness;
- influence of the near-surface layer resistivity;
- influence of target depth;
- influence of target size.

The figures are saved in English in the outputs/ directory.

Important physical convention:
- the ground surface is h = 0 m;
- depths are represented by negative z values in the 2D DC/IP model;
- for TDEM, the receiver and transmitter are placed at h = 0 m.

For the TDEM simulation, SimPEG returns the magnetic flux time derivative
for the chosen receiver. In a non-magnetic medium, mu = mu0 is constant,
therefore dB/dt = mu0 dH/dt. The shape of the transient curve is not changed
by this constant factor.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
)
from simpeg.electromagnetics import time_domain as tdem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PARAM_FILE = ROOT / "model_parameters.json"
with open(PARAM_FILE, "r", encoding="utf-8") as f:
    BASE = json.load(f)


def get_value(name: str, default: float) -> float:
    return float(BASE.get(name, default))


BASE_LATERITE_THICKNESS = get_value("laterite_thickness_m", 15.0)
BASE_SAPROLITE_THICKNESS = get_value("saprolite_thickness_m", 30.0)
BASE_TARGET_TOP = get_value("target_top_depth_m", 100.0)
BASE_TARGET_WIDTH = get_value("target_width_m", 80.0)
BASE_TARGET_HEIGHT = get_value("target_height_m", 40.0)

BASE_RHO_LATERITE = get_value("laterite_resistivity_ohm_m", 80.0)
BASE_RHO_SAPROLITE = get_value("saprolite_resistivity_ohm_m", 200.0)
BASE_RHO_BASEMENT = get_value("basement_resistivity_ohm_m", 1000.0)
BASE_RHO_TARGET = get_value("target_resistivity_ohm_m", 10.0)
BASE_CHARGEABILITY = get_value("target_chargeability", 0.10)

DOMAIN_WIDTH = get_value("model_width_m", 800.0)
SURFACE_CELL_SIZE = get_value("surface_cell_size_m", 10.0)


# -----------------------------------------------------------------------------
# Parameterized 2D model
# -----------------------------------------------------------------------------

def conductivity_2d_variant(x, z, laterite_thickness, laterite_resistivity,
                            target_top, target_width, target_height):
    """Conductivity model for DC/IP sensitivity tests.

    Surface: z = 0 m.
    Depth: z < 0 m.
    """
    sigma_laterite = 1.0 / laterite_resistivity
    sigma_saprolite = 1.0 / BASE_RHO_SAPROLITE
    sigma_basement = 1.0 / BASE_RHO_BASEMENT
    sigma_target = 1.0 / BASE_RHO_TARGET

    sigma = np.full_like(x, sigma_basement, dtype=float)

    saprolite_base = -(laterite_thickness + BASE_SAPROLITE_THICKNESS)
    laterite_base = -laterite_thickness

    sigma[z > saprolite_base] = sigma_saprolite
    sigma[z > laterite_base] = sigma_laterite

    x1 = -target_width / 2.0
    x2 = target_width / 2.0
    z1 = -target_top
    z2 = -(target_top + target_height)

    target = (x >= x1) & (x <= x2) & (z <= z1) & (z >= z2)
    sigma[target] = sigma_target

    return sigma


def chargeability_2d_variant(x, z, target_top, target_width, target_height):
    eta = np.zeros_like(x, dtype=float)

    x1 = -target_width / 2.0
    x2 = target_width / 2.0
    z1 = -target_top
    z2 = -(target_top + target_height)

    target = (x >= x1) & (x <= x2) & (z <= z1) & (z >= z2)
    eta[target] = BASE_CHARGEABILITY

    return eta


def build_dc_ip_mesh_and_surveys():
    topo_x = np.linspace(-DOMAIN_WIDTH / 2.0, DOMAIN_WIDTH / 2.0, 401)
    topo_xyz = np.c_[topo_x, np.zeros_like(topo_x)]

    source_list = generate_dcip_sources_line(
        survey_type="dipole-dipole",
        data_type="volt",
        dimension_type="2D",
        end_points=np.r_[-400.0, 400.0],
        topo=topo_xyz,
        num_rx_per_src=8,
        station_spacing=20.0,
    )

    dc_survey = dc.survey.Survey(source_list)
    ip_survey = ip.survey.from_dc_to_ip_survey(dc_survey)

    dh = SURFACE_CELL_SIZE
    mesh = TreeMesh([[(dh, 256)], [(dh, 128)]], x0="CN")

    mesh.refine_surface(topo_xyz, padding_cells_by_level=[0, 0, 3, 3], finalize=False)

    electrode_locations = np.c_[
        dc_survey.locations_a,
        dc_survey.locations_b,
        dc_survey.locations_m,
        dc_survey.locations_n,
    ]

    unique_locations = np.unique(
        electrode_locations.reshape((4 * dc_survey.nD, 2)),
        axis=0,
    )

    mesh.refine_points(unique_locations, padding_cells_by_level=[4, 4, 4], finalize=False)
    mesh.finalize()

    ind_active = active_from_xyz(mesh, topo_xyz)
    active_map = maps.InjectActiveCells(mesh, ind_active, 1e-8)
    eta_map = maps.InjectActiveCells(mesh, ind_active, 0.0) * maps.IdentityMap(
        nP=int(ind_active.sum())
    )

    cc = mesh.cell_centers[ind_active]

    return mesh, ind_active, active_map, eta_map, cc, dc_survey, ip_survey


def run_dc_ip_variant(mesh, ind_active, active_map, eta_map, cc, dc_survey, ip_survey,
                      laterite_thickness, laterite_resistivity,
                      target_top, target_width, target_height):
    sigma_active = conductivity_2d_variant(
        cc[:, 0], cc[:, 1],
        laterite_thickness=laterite_thickness,
        laterite_resistivity=laterite_resistivity,
        target_top=target_top,
        target_width=target_width,
        target_height=target_height,
    )

    sigma_map = active_map * maps.IdentityMap(nP=int(ind_active.sum()))

    dc_sim = dc.Simulation2DNodal(
        mesh=mesh,
        survey=dc_survey,
        sigmaMap=sigma_map,
    )

    dc_pred = dc_sim.dpred(sigma_active)
    rho_app = apparent_resistivity_from_voltage(dc_survey, dc_pred)

    eta_active = chargeability_2d_variant(
        cc[:, 0], cc[:, 1],
        target_top=target_top,
        target_width=target_width,
        target_height=target_height,
    )

    sigma_background = active_map * sigma_active

    ip_sim = ip.Simulation2DNodal(
        mesh=mesh,
        survey=ip_survey,
        etaMap=eta_map,
        sigma=sigma_background,
    )

    ip_pred = ip_sim.dpred(eta_active)

    return rho_app, ip_pred


# -----------------------------------------------------------------------------
# Parameterized 1D TDEM model
# -----------------------------------------------------------------------------

def tdem_layers_variant(laterite_thickness, laterite_resistivity,
                        target_top, target_height):
    """Build a simple 1D equivalent TDEM model.

    The 2D target is represented as a conductive layer. This is a simplification,
    but it allows the sensitivity of the TDEM transient to target depth and
    thickness to be studied in a first approximation.
    """
    saprolite_base = laterite_thickness + BASE_SAPROLITE_THICKNESS

    thicknesses = []
    resistivities = []

    thicknesses.append(laterite_thickness)
    resistivities.append(laterite_resistivity)

    thicknesses.append(BASE_SAPROLITE_THICKNESS)
    resistivities.append(BASE_RHO_SAPROLITE)

    if target_top > saprolite_base:
        thicknesses.append(target_top - saprolite_base)
        resistivities.append(BASE_RHO_BASEMENT)

    thicknesses.append(target_height)
    resistivities.append(BASE_RHO_TARGET)

    resistivities.append(BASE_RHO_BASEMENT)

    return np.asarray(thicknesses, dtype=float), np.asarray(resistivities, dtype=float)


def run_tdem_variant(laterite_thickness, laterite_resistivity,
                     target_top, target_height):
    times = np.logspace(-5, -2, 31)

    receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        locations=np.array([[0.0, 0.0, 0.0]]),
        times=times,
        orientation="z",
    )

    source = tdem.sources.CircularLoop(
        receiver_list=[receiver],
        location=np.array([0.0, 0.0, 0.0]),
        radius=50.0,
        current=1.0,
        waveform=tdem.sources.StepOffWaveform(),
    )

    survey = tdem.Survey([source])

    thicknesses, resistivities = tdem_layers_variant(
        laterite_thickness=laterite_thickness,
        laterite_resistivity=laterite_resistivity,
        target_top=target_top,
        target_height=target_height,
    )

    sigmas = 1.0 / resistivities

    sim = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=thicknesses,
        sigmaMap=maps.ExpMap(),
    )

    pred = sim.dpred(np.log(sigmas))

    return times, np.abs(pred)


# -----------------------------------------------------------------------------
# Sensitivity metrics and plotting
# -----------------------------------------------------------------------------

def relative_rms_change(reference, variant):
    reference = np.asarray(reference, dtype=float)
    variant = np.asarray(variant, dtype=float)

    denominator = np.sqrt(np.mean(reference ** 2))
    if denominator == 0:
        return np.nan

    return 100.0 * np.sqrt(np.mean((variant - reference) ** 2)) / denominator


def plot_sensitivity_curve(values, dc_scores, ip_scores, tdem_scores,
                           xlabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(values, dc_scores, "o-", label="DC apparent resistivity")
    plt.plot(values, ip_scores, "s-", label="IP response")
    plt.plot(values, tdem_scores, "^-", label="TDEM transient")
    plt.xlabel(xlabel)
    plt.ylabel("Relative RMS change (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200)
    plt.close()


def plot_tdem_transients(curves, title, filename):
    plt.figure(figsize=(8, 5))
    for label, times, response in curves:
        plt.loglog(times, response, "o-", lw=1.3, ms=4, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("|dBz/dt| or constant-scaled |dHz/dt|")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200)
    plt.close()


def main():
    mesh, ind_active, active_map, eta_map, cc, dc_survey, ip_survey = (
        build_dc_ip_mesh_and_surveys()
    )

    base_params = dict(
        laterite_thickness=BASE_LATERITE_THICKNESS,
        laterite_resistivity=BASE_RHO_LATERITE,
        target_top=BASE_TARGET_TOP,
        target_width=BASE_TARGET_WIDTH,
        target_height=BASE_TARGET_HEIGHT,
    )

    base_dc, base_ip = run_dc_ip_variant(
        mesh, ind_active, active_map, eta_map, cc, dc_survey, ip_survey,
        **base_params,
    )

    base_times, base_tdem = run_tdem_variant(
        laterite_thickness=BASE_LATERITE_THICKNESS,
        laterite_resistivity=BASE_RHO_LATERITE,
        target_top=BASE_TARGET_TOP,
        target_height=BASE_TARGET_HEIGHT,
    )

    experiments = [
        {
            "name": "laterite_thickness",
            "values": np.array([5.0, 10.0, 15.0, 20.0, 30.0]),
            "xlabel": "Laterite thickness (m)",
            "title": "Sensitivity to near-surface layer thickness",
            "filename": "sensitivity_laterite_thickness.png",
        },
        {
            "name": "laterite_resistivity",
            "values": np.array([40.0, 80.0, 150.0, 300.0]),
            "xlabel": "Laterite resistivity (Ohm m)",
            "title": "Sensitivity to near-surface layer resistivity",
            "filename": "sensitivity_laterite_resistivity.png",
        },
        {
            "name": "target_top",
            "values": np.array([60.0, 80.0, 100.0, 140.0, 180.0]),
            "xlabel": "Target top depth (m)",
            "title": "Sensitivity to target depth",
            "filename": "sensitivity_target_depth.png",
        },
        {
            "name": "target_width",
            "values": np.array([40.0, 80.0, 120.0, 160.0]),
            "xlabel": "Target width (m)",
            "title": "Sensitivity to target width",
            "filename": "sensitivity_target_width.png",
        },
    ]

    summary_rows = []
    tdem_depth_curves = []

    for exp in experiments:
        dc_scores = []
        ip_scores = []
        tdem_scores = []

        for value in exp["values"]:
            params = dict(base_params)
            params[exp["name"]] = float(value)

            dc_data, ip_data = run_dc_ip_variant(
                mesh, ind_active, active_map, eta_map, cc, dc_survey, ip_survey,
                **params,
            )

            dc_score = relative_rms_change(base_dc, dc_data)
            ip_score = relative_rms_change(base_ip, ip_data)

            if exp["name"] == "target_width":
                # A 1D TDEM model cannot represent target width.
                # Therefore, target-width sensitivity is not applicable to TDEM.
                tdem_score = np.nan
            else:
                times, tdem_data = run_tdem_variant(
                    laterite_thickness=params["laterite_thickness"],
                    laterite_resistivity=params["laterite_resistivity"],
                    target_top=params["target_top"],
                    target_height=params["target_height"],
                )
                tdem_score = relative_rms_change(base_tdem, tdem_data)

                if exp["name"] == "target_top":
                    tdem_depth_curves.append(
                        (f"Target top = {int(value)} m", times, tdem_data)
                    )

            dc_scores.append(dc_score)
            ip_scores.append(ip_score)
            tdem_scores.append(tdem_score)

            summary_rows.append(
                [exp["name"], float(value), dc_score, ip_score, tdem_score]
            )

        plot_sensitivity_curve(
            exp["values"],
            dc_scores,
            ip_scores,
            tdem_scores,
            xlabel=exp["xlabel"],
            title=exp["title"],
            filename=exp["filename"],
        )

    # Additional TDEM figure for Chapter 4: transient curves for different target depths.
    if tdem_depth_curves:
        plot_tdem_transients(
            tdem_depth_curves,
            "TDEM transient response for different target depths",
            "tdem_target_depth_comparison.png",
        )

    summary_path = OUT / "sensitivity_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("parameter,value,dc_relative_rms_percent,ip_relative_rms_percent,tdem_relative_rms_percent\n")
        for row in summary_rows:
            f.write(
                f"{row[0]},{row[1]:.6g},{row[2]:.6g},{row[3]:.6g},{row[4]:.6g}\n"
            )

    print("Sensitivity analysis completed.")
    print(f"Figures and CSV file saved in: {OUT}")


if __name__ == "__main__":
    main()
