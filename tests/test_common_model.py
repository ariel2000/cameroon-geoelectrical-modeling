import numpy as np

from scripts import common_model as model


def test_surface_and_depth_convention():
    x = np.array([0.0, 0.0, 0.0, 0.0])
    z = np.array([-1.0, -20.0, -60.0, -110.0])
    sigma = model.conductivity_2d(x, z)
    expected = np.array([
        model.SIGMA_LATERITE,
        model.SIGMA_SAPROLITE,
        model.SIGMA_BASEMENT,
        model.SIGMA_TARGET,
    ])
    np.testing.assert_allclose(sigma, expected)


def test_target_bounds_use_top_depth():
    x1, x2, z_top, z_bottom = model.target_bounds()
    assert x1 == -model.TARGET_WIDTH / 2.0
    assert x2 == model.TARGET_WIDTH / 2.0
    assert z_top == -model.TARGET_TOP
    assert z_bottom == -(model.TARGET_TOP + model.TARGET_HEIGHT)


def test_resistivity_conductivity_reciprocity():
    np.testing.assert_allclose(
        [model.SIGMA_LATERITE, model.SIGMA_SAPROLITE,
         model.SIGMA_BASEMENT, model.SIGMA_TARGET],
        1.0 / np.array([
            model.RHO_LATERITE, model.RHO_SAPROLITE,
            model.RHO_BASEMENT, model.RHO_TARGET,
        ]),
    )


def test_tree_mesh_covers_declared_domain():
    nx, nz = model.tree_mesh_shape(model.SURFACE_CELL_SIZE)
    assert nx * model.SURFACE_CELL_SIZE >= model.DOMAIN_WIDTH
    assert nz * model.SURFACE_CELL_SIZE / 2.0 >= model.DOMAIN_DEPTH
