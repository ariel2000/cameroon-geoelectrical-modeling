from __future__ import annotations

"""
Sensitivity analysis for the primary DC and IP parameter-estimation workflow.

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

The TDEM helper functions below are retained only for reproducibility of the
earlier exploratory scripts. They are not executed by this primary analysis.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.utils import get_default_solver
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
)
from simpeg.electromagnetics import time_domain as tdem

from common_model import tree_mesh_shape


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PARAM_FILE = Path(__file__).resolve().with_name("model_parameters.json")
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
                            target_top, target_width, target_height,
                            target_resistivity=BASE_RHO_TARGET,
                            include_target=True):
    """Conductivity model for DC/IP sensitivity tests.

    Surface: z = 0 m.
    Depth: z < 0 m.
    """
    sigma_laterite = 1.0 / laterite_resistivity
    sigma_saprolite = 1.0 / BASE_RHO_SAPROLITE
    sigma_basement = 1.0 / BASE_RHO_BASEMENT
    sigma_target = 1.0 / float(target_resistivity)

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
    if include_target:
        sigma[target] = sigma_target

    return sigma


def chargeability_2d_variant(x, z, target_top, target_width, target_height,
                             target_chargeability=BASE_CHARGEABILITY,
                             include_target=True):
    eta = np.zeros_like(x, dtype=float)

    x1 = -target_width / 2.0
    x2 = target_width / 2.0
    z1 = -target_top
    z2 = -(target_top + target_height)

    target = (x >= x1) & (x <= x2) & (z <= z1) & (z >= z2)
    if include_target:
        eta[target] = float(target_chargeability)

    return eta


def build_dc_ip_mesh_and_surveys(cell_size=SURFACE_CELL_SIZE):
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

    dh = float(cell_size)
    nx, nz = tree_mesh_shape(dh)
    mesh = TreeMesh(
        [[(dh, nx)], [(dh, nz)]], x0="CN", diagonal_balance=True
    )

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

    # Resolve the geological interfaces and the target explicitly. Without
    # this refinement, deep cells alias the target geometry and IP responses
    # do not converge even when the nominal base-cell size is reduced.
    interface_half_width = 2.0 * dh
    for interface_depth in (
        BASE_LATERITE_THICKNESS,
        BASE_LATERITE_THICKNESS + BASE_SAPROLITE_THICKNESS,
    ):
        mesh.refine_box(
            [[-DOMAIN_WIDTH / 2.0, -interface_depth - interface_half_width]],
            [[DOMAIN_WIDTH / 2.0, -interface_depth + interface_half_width]],
            levels=mesh.max_level,
            finalize=False,
        )

    mesh.refine_box(
        [[-220.0 - 2.0 * dh, -220.0 - 2.0 * dh]],
        [[220.0 + 2.0 * dh, 0.0]],
        levels=mesh.max_level,
        finalize=False,
    )
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
                      target_top, target_width, target_height,
                      target_resistivity=BASE_RHO_TARGET,
                      target_chargeability=BASE_CHARGEABILITY,
                      include_target=True):
    sigma_active = conductivity_2d_variant(
        cc[:, 0], cc[:, 1],
        laterite_thickness=laterite_thickness,
        laterite_resistivity=laterite_resistivity,
        target_top=target_top,
        target_width=target_width,
        target_height=target_height,
        target_resistivity=target_resistivity,
        include_target=include_target,
    )

    sigma_map = active_map * maps.IdentityMap(nP=int(ind_active.sum()))

    dc_sim = dc.Simulation2DNodal(
        mesh=mesh,
        survey=dc_survey,
        sigmaMap=sigma_map,
        solver=get_default_solver(),
    )

    dc_pred = dc_sim.dpred(sigma_active)
    rho_app = apparent_resistivity_from_voltage(dc_survey, dc_pred)

    eta_active = chargeability_2d_variant(
        cc[:, 0], cc[:, 1],
        target_top=target_top,
        target_width=target_width,
        target_height=target_height,
        target_chargeability=target_chargeability,
        include_target=include_target,
    )

    sigma_background = active_map * sigma_active

    ip_sim = ip.Simulation2DNodal(
        mesh=mesh,
        survey=ip_survey,
        etaMap=eta_map,
        sigma=sigma_background,
        solver=get_default_solver(),
    )

    ip_pred = ip_sim.dpred(eta_active)

    return rho_app, ip_pred


# -----------------------------------------------------------------------------
# Parameterized 1D TDEM model
# -----------------------------------------------------------------------------

def tdem_layers_variant(laterite_thickness, laterite_resistivity,
                        target_top, target_height,
                        target_resistivity=BASE_RHO_TARGET,
                        include_target=True):
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

    if not include_target:
        resistivities.append(BASE_RHO_BASEMENT)
        return np.asarray(thicknesses, dtype=float), np.asarray(resistivities, dtype=float)

    if target_top > saprolite_base:
        thicknesses.append(target_top - saprolite_base)
        resistivities.append(BASE_RHO_BASEMENT)

    thicknesses.append(target_height)
    resistivities.append(float(target_resistivity))

    resistivities.append(BASE_RHO_BASEMENT)

    return np.asarray(thicknesses, dtype=float), np.asarray(resistivities, dtype=float)


def run_tdem_variant(laterite_thickness, laterite_resistivity,
                     target_top, target_height,
                     target_resistivity=BASE_RHO_TARGET,
                     include_target=True,
                     times=None):
    if times is None:
        times = np.logspace(-5, -2, 31)
    times = np.asarray(times, dtype=float)

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
        target_resistivity=target_resistivity,
        include_target=include_target,
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


def plot_sensitivity_curve(values, dc_scores, ip_scores, xlabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(values, dc_scores, "o-", label="DC apparent resistivity")
    plt.plot(values, ip_scores, "s-", label="IP response")
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
    plt.ylabel(r"$|dB_z/dt|$ (T/s)")
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

    experiments = [
        {
            "name": "laterite_thickness",
            "values": np.array([2.5, 5.0, 7.5]),
            "xlabel": "Laterite thickness (m)",
            "title": "Sensitivity to near-surface layer thickness",
            "filename": "sensitivity_laterite_thickness.png",
        },
        {
            "name": "laterite_resistivity",
            "values": np.array([2100.0, 3000.0, 4200.0]),
            "xlabel": "Laterite resistivity (Ohm m)",
            "title": "Sensitivity to near-surface layer resistivity",
            "filename": "sensitivity_laterite_resistivity.png",
        },
        {
            "name": "target_top",
            "values": np.array([10.0, 15.0, 20.0, 25.0, 30.0]),
            "xlabel": "Target top depth (m)",
            "title": "Sensitivity to target depth",
            "filename": "sensitivity_target_depth.png",
        },
        {
            "name": "target_width",
            "values": np.array([50.0, 75.0, 100.0, 150.0]),
            "xlabel": "Target width (m)",
            "title": "Sensitivity to target width",
            "filename": "sensitivity_target_width.png",
        },
    ]

    summary_rows = []
    for exp in experiments:
        dc_scores = []
        ip_scores = []

        for value in exp["values"]:
            params = dict(base_params)
            params[exp["name"]] = float(value)

            dc_data, ip_data = run_dc_ip_variant(
                mesh, ind_active, active_map, eta_map, cc, dc_survey, ip_survey,
                **params,
            )

            dc_score = relative_rms_change(base_dc, dc_data)
            ip_score = relative_rms_change(base_ip, ip_data)

            dc_scores.append(dc_score)
            ip_scores.append(ip_score)

            summary_rows.append([exp["name"], float(value), dc_score, ip_score])

        plot_sensitivity_curve(
            exp["values"],
            dc_scores,
            ip_scores,
            xlabel=exp["xlabel"],
            title=exp["title"],
            filename=exp["filename"],
        )

    summary_path = OUT / "sensitivity_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("parameter,value,dc_relative_rms_percent,ip_relative_rms_percent\n")
        for row in summary_rows:
            f.write(
                f"{row[0]},{row[1]:.6g},{row[2]:.6g},{row[3]:.6g}\n"
            )

    print("Sensitivity analysis completed.")
    print(f"Figures and CSV file saved in: {OUT}")


if __name__ == "__main__":
    main()
