import matplotlib.pyplot as plt
import numpy as np

from kcwi_pipeline.apertures import (
    WhiteLightRangeController,
    _white_light_two_panel,
)


def test_white_light_range_controller_matches_direct_cube_sum() -> None:
    cube = np.arange(6 * 3 * 2, dtype=np.float32).reshape(6, 3, 2)
    cube[2, 1, 1] = np.nan
    wavelength = np.arange(5000.0, 5006.0)
    controller = WhiteLightRangeController(
        cube,
        wavelength,
        minimum=5001.0,
        maximum=5004.0,
    )

    expected_initial = np.nansum(cube[1:5], axis=0)
    assert np.allclose(controller.image(), expected_initial)

    controller.set_bounds(5002.0, 5003.0)
    expected_narrow = np.nansum(cube[2:4], axis=0)
    assert np.allclose(controller.image(), expected_narrow)
    assert controller.bounds == (5002.0, 5003.0)


def test_wavelength_slider_updates_both_white_light_panels() -> None:
    rng = np.random.default_rng(27)
    cube = rng.normal(size=(6, 4, 3)).astype(np.float32)
    wavelength = np.arange(6000.0, 6006.0)
    controller = WhiteLightRangeController(
        cube,
        wavelength,
        minimum=6000.0,
        maximum=6005.0,
    )
    fig, ax_left, ax_right = _white_light_two_panel(
        controller.image(),
        "test white light",
        wavelength_controller=controller,
    )

    wavelength_slider = fig._kcwi_wavelength_widgets[0]
    wavelength_slider.set_val((6002.0, 6004.0))
    expected = np.sum(cube[2:5], axis=0)

    assert np.allclose(np.asarray(ax_left.images[0].get_array()), expected)
    assert np.allclose(np.asarray(ax_right.images[0].get_array()), expected)
    assert controller.bounds == (6002.0, 6004.0)
    assert "6002.0-6004.0 A" in ax_left.get_title()
    plt.close(fig)
