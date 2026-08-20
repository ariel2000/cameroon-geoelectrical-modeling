"""Deterministic uncertainty and misfit utilities for synthetic experiments."""

from __future__ import annotations

import numpy as np


def standard_deviation(data, relative_error: float, error_floor: float):
    data = np.asarray(data, dtype=float)
    if relative_error < 0 or error_floor < 0:
        raise ValueError("Uncertainty terms must be non-negative")
    return np.sqrt((relative_error * np.abs(data)) ** 2 + error_floor**2)


def add_gaussian_noise(data, relative_error: float, error_floor: float, seed: int):
    data = np.asarray(data, dtype=float)
    std = standard_deviation(data, relative_error, error_floor)
    rng = np.random.default_rng(seed)
    return data + rng.normal(0.0, std, data.shape), std


def normalized_rms(residual, standard_deviation_values):
    residual = np.asarray(residual, dtype=float)
    std = np.asarray(standard_deviation_values, dtype=float)
    if np.any(std <= 0):
        raise ValueError("All standard deviations must be positive")
    return float(np.sqrt(np.mean((residual / std) ** 2)))
