from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    apparent_resistivity_from_voltage,
    plot_pseudosection,
)

from common_model import DOMAIN_WIDTH, SURFACE_CELL_SIZE, conductivity_2d

# 1. Flat topography
topo_x = np.linspace(-DOMAIN_WIDTH / 2, DOMAIN_WIDTH / 2, 401)
topo_xyz = np.c_[topo_x, np.zeros_like(topo_x)]

# 2. Dipole-dipole survey
source_list = generate_dcip_sources_line(
    survey_type="dipole-dipole",
    data_type="volt",
    dimension_type="2D",
    end_points=np.r_[-400.0, 400.0],
    topo=topo_xyz,
    num_rx_per_src=8,
    station_spacing=20.0,
)
survey = dc.survey.Survey(source_list)

# 3. Mesh
dh = SURFACE_CELL_SIZE
mesh = TreeMesh([[(dh, 256)], [(dh, 128)]], x0="CN")
mesh.refine_surface(topo_xyz, padding_cells_by_level=[0, 0, 3, 3], finalize=False)

electrode_locations = np.c_[
    survey.locations_a, survey.locations_b, survey.locations_m, survey.locations_n
]
unique_locations = np.unique(electrode_locations.reshape((4 * survey.nD, 2)), axis=0)
mesh.refine_points(unique_locations, padding_cells_by_level=[4, 4, 4], finalize=False)
mesh.finalize()

# 4. Active cells and conductivity model
ind_active = active_from_xyz(mesh, topo_xyz)
active_map = maps.InjectActiveCells(mesh, ind_active, 1e-8)

cc = mesh.cell_centers[ind_active]
sigma_active = conductivity_2d(cc[:, 0], cc[:, 1])
sigma_map = active_map * maps.IdentityMap(nP=int(ind_active.sum()))

# 5. Forward problem
simulation = dc.Simulation2DNodal(
    mesh=mesh,
    survey=survey,
    sigmaMap=sigma_map,
)

dpred = simulation.dpred(sigma_active)
rho_app = apparent_resistivity_from_voltage(survey, dpred)

# 6. Plot
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
plot_pseudosection(
    survey,
    rho_app,
    plot_type="contourf",
    scale="log",
    ax=ax,
    contourf_opts={"levels": 25},
)
ax.set_title("Synthetic DC apparent resistivity")
plt.tight_layout()
plt.savefig("outputs/dc_pseudosection.png", dpi=200)
print("Saved outputs/dc_pseudosection.png")
