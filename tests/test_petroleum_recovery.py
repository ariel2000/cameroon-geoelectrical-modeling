import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from petroleum_parameter_recovery import (  # noqa: E402
    simandoux_water_saturation,
    synthetic_log,
)


def test_synthetic_log_places_reservoir_at_declared_depth():
    import numpy as np

    depths = np.arange(0.0, 11.0, 1.0)
    values = synthetic_log(depths, 3.0, 4.0, 8.0, 1.0)
    assert all(values[(depths >= 3.0) & (depths <= 7.0)] == 8.0)
    assert all(values[(depths < 3.0) | (depths > 7.0)] == 1.0)


def test_simandoux_solution_matches_rio_del_rey_reference():
    saturation = simandoux_water_saturation(3.4, 0.25, 0.43, 0.045, 1.2)
    assert saturation == pytest.approx(0.3489180954)
    assert 0.03 <= saturation <= 0.63


def test_simandoux_rejects_nonphysical_resistivity():
    with pytest.raises(ValueError):
        simandoux_water_saturation(0.0, 0.25, 0.43, 0.045, 1.2)
