from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from common_model import conductivity_2d, DOMAIN_WIDTH, DOMAIN_DEPTH

x = np.linspace(-DOMAIN_WIDTH / 2, DOMAIN_WIDTH / 2, 400)
z = np.linspace(-DOMAIN_DEPTH, 0, 200)
X, Z = np.meshgrid(x, z)

sigma = conductivity_2d(X, Z)
rho = 1.0 / sigma

plt.figure(figsize=(9, 4.8))
im = plt.pcolormesh(X, -Z, rho, shading="auto")
cb = plt.colorbar(im)
cb.set_label("Resistivity (Ohm·m)")
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.title("Synthetic geoelectrical reference model")
plt.tight_layout()
plt.savefig("outputs/reference_model.png", dpi=200)
print("Saved outputs/reference_model.png")
