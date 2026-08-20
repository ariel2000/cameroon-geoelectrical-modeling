from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAM_FILE = Path(__file__).resolve().with_name("model_parameters.json")

with open(PARAM_FILE, "r", encoding="utf-8") as f:
    P = json.load(f)


def _positive(name: str) -> float:
    value = float(P[name])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive; got {value!r}")
    return value

# Resistivity
RHO_LATERITE = _positive("laterite_resistivity_ohm_m")
RHO_SAPROLITE = _positive("saprolite_resistivity_ohm_m")
RHO_BASEMENT = _positive("basement_resistivity_ohm_m")
RHO_TARGET = _positive("target_resistivity_ohm_m")

# Conductivity
SIGMA_LATERITE = 1.0 / RHO_LATERITE
SIGMA_SAPROLITE = 1.0 / RHO_SAPROLITE
SIGMA_BASEMENT = 1.0 / RHO_BASEMENT
SIGMA_TARGET = 1.0 / RHO_TARGET

# Geometry
LATERITE_THICKNESS = _positive("laterite_thickness_m")
SAPROLITE_THICKNESS = _positive("saprolite_thickness_m")
TARGET_TOP = _positive("target_top_depth_m")
TARGET_WIDTH = _positive("target_width_m")
TARGET_HEIGHT = _positive("target_height_m")

# Chargeability
TARGET_CHARGEABILITY = float(P["target_chargeability"])
if not 0.0 <= TARGET_CHARGEABILITY < 1.0:
    raise ValueError("target_chargeability must be in [0, 1)")

# Domain
DOMAIN_WIDTH = _positive("model_width_m")
DOMAIN_DEPTH = _positive("model_depth_m")
SURFACE_CELL_SIZE = _positive("surface_cell_size_m")
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DCIP_LINE_START = float(P["dcip_line_start_m"])
DCIP_LINE_END = float(P["dcip_line_end_m"])
DCIP_STATION_SPACING = _positive("dcip_station_spacing_m")
DCIP_RECEIVERS_PER_SOURCE = int(P["dcip_receivers_per_source"])
TDEM_LOOP_RADIUS = _positive("tdem_loop_radius_m")
TDEM_CURRENT = _positive("tdem_current_a")
TDEM_TIME_MIN = _positive("tdem_time_min_s")
TDEM_TIME_MAX = _positive("tdem_time_max_s")
TDEM_TIME_CHANNELS = int(P["tdem_time_channels"])
RANDOM_SEED = int(P["random_seed"])


def target_bounds():
    x1 = -TARGET_WIDTH / 2.0
    x2 = TARGET_WIDTH / 2.0
    z1 = -TARGET_TOP
    z2 = -(TARGET_TOP + TARGET_HEIGHT)
    return x1, x2, z1, z2


def tree_mesh_shape(cell_size: float) -> tuple[int, int]:
    """Return power-of-two base-grid counts covering the declared domain.

    The vertical TreeMesh is centred on z=0, hence it must span twice the
    requested subsurface depth. Adaptive refinement keeps the active problem
    substantially smaller than the full tensor grid.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    nx = 2 ** int(np.ceil(np.log2(DOMAIN_WIDTH / cell_size)))
    nz = 2 ** int(np.ceil(np.log2(2.0 * DOMAIN_DEPTH / cell_size)))
    return max(nx, 16), max(nz, 16)


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


def validate_reciprocal_parameters(rtol: float = 1e-10) -> None:
    """Reject inconsistent duplicated resistivity/conductivity values in JSON."""
    pairs = (
        ("laterite", RHO_LATERITE, P.get("laterite_conductivity_s_m")),
        ("saprolite", RHO_SAPROLITE, P.get("saprolite_conductivity_s_m")),
        ("basement", RHO_BASEMENT, P.get("basement_conductivity_s_m")),
        ("target", RHO_TARGET, P.get("target_conductivity_s_m")),
    )
    for label, rho, configured_sigma in pairs:
        if configured_sigma is None:
            continue
        if not np.isclose(1.0 / rho, float(configured_sigma), rtol=rtol):
            raise ValueError(f"Inconsistent resistivity/conductivity for {label}")


validate_reciprocal_parameters()
