from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAM_FILE = ROOT / "model_parameters.json"

with open(PARAM_FILE, "r", encoding="utf-8") as f:
    P = json.load(f)

# Resistivity
RHO_LATERITE = float(P["laterite_resistivity_ohm_m"])
RHO_SAPROLITE = float(P["saprolite_resistivity_ohm_m"])
RHO_BASEMENT = float(P["basement_resistivity_ohm_m"])
RHO_TARGET = float(P["target_resistivity_ohm_m"])

# Conductivity
SIGMA_LATERITE = float(P["laterite_conductivity_s_m"])
SIGMA_SAPROLITE = float(P["saprolite_conductivity_s_m"])
SIGMA_BASEMENT = float(P["basement_conductivity_s_m"])
SIGMA_TARGET = float(P["target_conductivity_s_m"])

# Geometry
LATERITE_THICKNESS = float(P["laterite_thickness_m"])
SAPROLITE_THICKNESS = float(P["saprolite_thickness_m"])
TARGET_TOP = float(P["target_top_depth_m"])
TARGET_WIDTH = float(P["target_width_m"])
TARGET_HEIGHT = float(P["target_height_m"])

# Chargeability
TARGET_CHARGEABILITY = float(P["target_chargeability"])

# Domain
DOMAIN_WIDTH = float(P["model_width_m"])
DOMAIN_DEPTH = float(P["model_depth_m"])
SURFACE_CELL_SIZE = float(P["surface_cell_size_m"])
DEEP_CELL_SIZE = float(P["deep_cell_size_m"])


def target_bounds():
    x1 = -TARGET_WIDTH / 2.0
    x2 = TARGET_WIDTH / 2.0
    z1 = -TARGET_TOP
    z2 = -(TARGET_TOP + TARGET_HEIGHT)
    return x1, x2, z1, z2


def conductivity_2d(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    x, z are arrays of cell-center coordinates.
    Surface is at z = 0, depth is negative.
    """
    sigma = np.full_like(x, SIGMA_BASEMENT, dtype=float)

    # Saprolite
    sigma[z > -(LATERITE_THICKNESS + SAPROLITE_THICKNESS)] = SIGMA_SAPROLITE

    # Laterite
    sigma[z > -LATERITE_THICKNESS] = SIGMA_LATERITE

    # Conductive target
    x1, x2, z1, z2 = target_bounds()
    mask = (x >= x1) & (x <= x2) & (z <= z1) & (z >= z2)
    sigma[mask] = SIGMA_TARGET

    return sigma


def chargeability_2d(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Background chargeability is taken as near zero.
    The target is highly chargeable.
    """
    eta = np.zeros_like(x, dtype=float)
    x1, x2, z1, z2 = target_bounds()
    mask = (x >= x1) & (x <= x2) & (z <= z1) & (z >= z2)
    eta[mask] = TARGET_CHARGEABILITY
    return eta


def tdem_layered_model():
    """
    1D layered model used for the baseline TDEM simulation.
    """
    thicknesses = np.r_[LATERITE_THICKNESS, SAPROLITE_THICKNESS]
    resistivities = np.r_[RHO_LATERITE, RHO_SAPROLITE, RHO_BASEMENT]
    return thicknesses, resistivities
