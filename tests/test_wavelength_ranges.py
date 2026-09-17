import json

import numpy as np
from astropy.io import fits
from matplotlib.backend_bases import KeyEvent

from kcwi_pipeline import calibration as calibration_module
from kcwi_pipeline import object_workflow
from kcwi_pipeline.calibration import plot_o2_correction_diagnostic
from kcwi_pipeline.object_workflow import (
    TELLURIC_WINDOWS,
    _common_wavelength_coverage,
    _resolve_wavelength_range,
    _trim_side_arrays,
    interactive_continuum_spline,
)


def _write_cube(path, start: float, stop: float, samples: int = 11) -> None:
    data = np.zeros((samples, 2, 2), dtype=np.float32)
    header = fits.Header()
    header["CRVAL3"] = start
    header["CRPIX3"] = 1.0
    header["CDELT3"] = (stop - start) / (samples - 1)
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def test_common_coverage_uses_intersection_of_all_exposures(tmp_path) -> None:
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fits"
    _write_cube(first, 3000.0, 6000.0)
    _write_cube(second, 3100.0, 5900.0)

    assert np.allclose(
        _common_wavelength_coverage([first, second], "BLUE"),
        (3100.0, 5900.0),
    )


def test_standard_range_is_suggested_saved_and_reused(monkeypatch, tmp_path) -> None:
    standard_cube = tmp_path / "standard.fits"
    science_cube = tmp_path / "science.fits"
    _write_cube(standard_cube, 3000.0, 6000.0)
    _write_cube(science_cube, 3000.0, 6000.0)
    calib_dir = tmp_path / "calibrations"
    prompts = []

    def accept_default(question, default=None):
        prompts.append((question, default))
        return default

    monkeypatch.setattr(object_workflow, "prompt", accept_default)
    approved = _resolve_wavelength_range(
        tmp_path / "STD",
        "BLUE",
        [standard_cube],
        calib_dir,
        standard=True,
        override=None,
    )

    assert np.allclose(approved, (3300.0, 5700.0))
    assert prompts == [("BLUE usable wavelength range (A)", "3300.0:5700.0")]
    with open(calib_dir / "wavelength_ranges.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["ranges"]["BLUE"]["approved_range_A"] == [3300.0, 5700.0]

    monkeypatch.setattr(
        object_workflow,
        "prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must reuse saved range")),
    )
    reused = _resolve_wavelength_range(
        tmp_path / "SCIENCE",
        "BLUE",
        [science_cube],
        calib_dir,
        standard=False,
        override=None,
    )
    assert reused == approved


def test_trim_uses_approved_dynamic_range() -> None:
    wavelength = np.arange(5000.0, 11001.0, 100.0)
    values = np.arange(wavelength.size)

    trimmed_wavelength, trimmed_values = _trim_side_arrays(
        "RED",
        wavelength,
        values,
        wavelength_range=(6200.0, 10300.0),
    )

    assert trimmed_wavelength[0] == 6200.0
    assert trimmed_wavelength[-1] == 10300.0
    assert np.array_equal(trimmed_values, values[12:54])


def test_explicit_range_allows_cube_narrower_than_default_edge_trims(tmp_path) -> None:
    narrow_cube = tmp_path / "narrow.fits"
    _write_cube(narrow_cube, 5000.0, 5600.0)

    approved = _resolve_wavelength_range(
        tmp_path / "STD",
        "RED",
        [narrow_cube],
        tmp_path / "calibrations",
        standard=True,
        override=(5100.0, 5500.0),
    )

    assert approved == (5100.0, 5500.0)


def test_telluric_windows_extend_through_11000_angstroms() -> None:
    assert (5875.0, 6000.0) not in TELLURIC_WINDOWS
    assert (8900.0, 9260.0) in TELLURIC_WINDOWS
    assert (9265.0, 9630.0) in TELLURIC_WINDOWS
    assert (9635.0, 10000.0) in TELLURIC_WINDOWS
    assert (10700.0, 11000.0) in TELLURIC_WINDOWS


def test_spline_editor_shows_trimmed_data_with_margin_and_overlapping_windows(
    monkeypatch,
) -> None:
    wavelength = np.linspace(6500.0, 7000.0, 101)
    counts = np.linspace(10.0, 12.0, wavelength.size)
    reference = np.ones_like(wavelength)

    def inspect_and_accept() -> None:
        figure = object_workflow.plt.gcf()
        observed_axis = figure.axes[0]
        assert np.allclose(observed_axis.get_xlim(), (6400.0, 7100.0))
        assert len(observed_axis.patches) == 1
        figure.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", figure.canvas, key="a"),
        )

    monkeypatch.setattr(object_workflow.plt, "show", inspect_and_accept)
    interactive_continuum_spline(
        wavelength,
        counts,
        reference,
        title="trimmed RED standard",
        show=True,
        exclude_windows=TELLURIC_WINDOWS,
        initial_points=[
            (5000.0, 9.0),
            (6550.0, 10.2),
            (6950.0, 11.8),
            (10500.0, 13.0),
        ],
    )


def test_telluric_detail_uses_compact_grid_and_only_overlapping_windows(
    monkeypatch,
    tmp_path,
) -> None:
    wavelength = np.linspace(6800.0, 8400.0, 1601)
    transmission = np.ones_like(wavelength)
    telluric_mask = np.zeros_like(wavelength, dtype=bool)
    for lo, hi in TELLURIC_WINDOWS:
        telluric_mask |= (wavelength >= lo) & (wavelength <= hi)
    transmission[telluric_mask] = 0.85
    before = np.ones_like(wavelength)
    captured = {}
    original_close = calibration_module.plt.close

    def capture_figure(figure) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(calibration_module.plt, "close", capture_figure)
    plot_o2_correction_diagnostic(
        "test",
        wavelength,
        before,
        before / transmission,
        transmission,
        transmission,
        telluric_mask,
        TELLURIC_WINDOWS,
        tmp_path / "telluric_detail.png",
        show=False,
    )

    figure = captured["figure"]
    try:
        titled_axes = {axis.get_title(): axis for axis in figure.axes if axis.get_title()}
        zoom_titles = [title for title in titled_axes if title.startswith("Telluric window")]
        assert len(zoom_titles) == 4
        assert not any("8900" in title or "10700" in title for title in zoom_titles)
        assert np.allclose(
            titled_axes["test RED: full spectrum"].get_xlim(),
            (6800.0, 8400.0),
        )
        assert np.isclose(figure.get_size_inches()[0], 15.0)
    finally:
        original_close(figure)
