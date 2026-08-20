"""Parametric recovery of a 1D conductive layer from noisy TDEM data."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from common_model import OUTPUT_DIR, P, RANDOM_SEED
from sensitivity_analysis import (
    BASE_LATERITE_THICKNESS,
    BASE_RHO_LATERITE,
    run_tdem_variant,
)
from uncertainty import add_gaussian_noise


def response(parameters, times):
    top, thickness, resistivity = np.exp(parameters)
    _, predicted = run_tdem_variant(
        laterite_thickness=BASE_LATERITE_THICKNESS,
        laterite_resistivity=BASE_RHO_LATERITE,
        target_top=top,
        target_height=thickness,
        target_resistivity=resistivity,
        times=times,
    )
    return predicted


def main():
    times = np.logspace(-5, -2, 41)
    truth = np.array([95.0, 35.0, 12.0])
    synthetic = response(np.log(truth), times)
    observed, std = add_gaussian_noise(
        synthetic, float(P["tdem_relative_error"]),
        float(P["tdem_error_floor_t_s"]), RANDOM_SEED + 2,
    )

    def residual(log_parameters):
        return (response(log_parameters, times) - observed) / std

    lower = np.log([50.0, 10.0, 3.0])
    upper = np.log([180.0, 100.0, 50.0])
    starts = ([70.0, 20.0, 8.0], [100.0, 40.0, 15.0], [150.0, 70.0, 30.0])
    solutions = [
        least_squares(residual, np.log(start), bounds=(lower, upper), max_nfev=150)
        for start in starts
    ]
    result = min(solutions, key=lambda item: np.sum(item.fun**2))
    estimate = np.exp(result.x)

    dof = max(times.size - estimate.size, 1)
    covariance_log = np.linalg.pinv(result.jac.T @ result.jac) * np.sum(result.fun**2) / dof
    standard_error = estimate * np.sqrt(np.maximum(np.diag(covariance_log), 0.0))
    predicted = response(result.x, times)

    summary = {
        "model_scope": "1D laterally infinite conductive-layer equivalent",
        "random_seed": RANDOM_SEED + 2,
        "true_parameters": dict(zip(["top_depth_m", "thickness_m", "resistivity_ohm_m"], truth)),
        "estimated_parameters": dict(zip(["top_depth_m", "thickness_m", "resistivity_ohm_m"], estimate)),
        "linearized_standard_errors": dict(zip(["top_depth_m", "thickness_m", "resistivity_ohm_m"], standard_error)),
        "normalized_rms": float(np.sqrt(np.mean(result.fun**2))),
        "success": bool(result.success),
        "message": result.message,
    }
    path = OUTPUT_DIR / "tdem_parameter_recovery.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(times, np.abs(observed), "o", ms=4, label="Noisy synthetic data")
    ax.loglog(times, synthetic, "-", label="Noise-free truth")
    ax.loglog(times, predicted, "--", label="Recovered model")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|dB_z/dt|$ (T/s)")
    ax.set_title("TDEM 1D parametric recovery")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tdem_parameter_recovery.png", dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
