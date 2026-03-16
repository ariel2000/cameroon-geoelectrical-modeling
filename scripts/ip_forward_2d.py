from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from discretize import TreeMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static import induced_polarization as ip
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
    plot_pseudosection,
)

from common_model import DOMAIN_WIDTH, SURFACE_CELL_SIZE, conductivity_2d, chargeability_2d

# 1. Flat topography
topo_x = np.linspace(-DOMAIN_WIDTH / 2, DOMAIN_WIDTH / 2, 401)
topo_xyz = np.c_[topo_x, np.zeros_like(topo_x)]

# 2. Survey geometry
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

# 3. Mesh
dh = SURFACE_CELL_SIZE
mesh = TreeMesh([[(dh, 256)], [(dh, 128)]], x0="CN")
mesh.refine_surface(topo_xyz, padding_cells_by_level=[0, 0, 3, 3], finalize=False)

electrode_locations = np.c_[
    dc_survey.locations_a, dc_survey.locations_b, dc_survey.locations_m, dc_survey.locations_n
]
unique_locations = np.unique(electrode_locations.reshape((4 * dc_survey.nD, 2)), axis=0)
mesh.refine_points(unique_locations, padding_cells_by_level=[4, 4, 4], finalize=False)
mesh.finalize()

# 4. Active cells and models
ind_active = active_from_xyz(mesh, topo_xyz)
active_map = maps.InjectActiveCells(mesh, ind_active, 1e-8)
cc = mesh.cell_centers[ind_active]

sigma_active = conductivity_2d(cc[:, 0], cc[:, 1])
eta_active = chargeability_2d(cc[:, 0], cc[:, 1])

eta_map = maps.InjectActiveCells(mesh, ind_active, 0.0) * maps.IdentityMap(nP=int(ind_active.sum()))
sigma_background = active_map * sigma_active

# 5. Forward problem
simulation = ip.Simulation2DNodal(
    mesh=mesh,
    survey=ip_survey,
    etaMap=eta_map,
    sigma=sigma_background,
)

dpred = simulation.dpred(eta_active)

# 6. Plot
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
plot_pseudosection(
    ip_survey,
    dpred,
    plot_type="contourf",
    scale="linear",
    ax=ax,
    contourf_opts={"levels": 25},
)
ax.set_title("Synthetic IP response")
plt.tight_layout()
plt.savefig("outputs/ip_pseudosection.png", dpi=200)
print("Saved outputs/ip_pseudosection.png")
