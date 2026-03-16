from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from common_model import tdem_layered_model

# 1. Same laterite-saprolite-basement profile
thicknesses, resistivities = tdem_layered_model()
sigmas = 1.0 / resistivities

# 2. Time channels
times = np.logspace(-5, -2, 31)

# 3. Receiver and source
receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
    locations=np.array([[0.0, 0.0, 30.0]]),
    times=times,
    orientation="z",
)

waveform = tdem.sources.StepOffWaveform()
source = tdem.sources.CircularLoop(
    receiver_list=[receiver],
    location=np.array([0.0, 0.0, 30.0]),
    radius=50.0,
    current=1.0,
    waveform=waveform,
)
survey = tdem.Survey([source])

# 4. 1D layered simulation
simulation = tdem.Simulation1DLayered(
    survey=survey,
    thicknesses=thicknesses,
    sigmaMap=maps.ExpMap(),
)

m = np.log(sigmas)
dpred = simulation.dpred(m)

# 5. Plot
plt.figure(figsize=(7, 5))
plt.loglog(times, np.abs(dpred), "o-", lw=1.5)
plt.xlabel("Time (s)")
plt.ylabel("|dBz/dt|")
plt.title("Synthetic TDEM transient response")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/tdem_transient.png", dpi=200)
print("Saved outputs/tdem_transient.png")
