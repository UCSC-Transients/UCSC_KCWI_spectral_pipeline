import json

import numpy as np
from astropy.io import fits

from kcwi_pipeline.spectral_cr import (
    detect_cr_like_narrow_features,
    interpolate_rejected_candidate,
    resolving_power_from_header,
)


def test_resolving_power_uses_grating_and_slicer_header_values() -> None:
    header = fits.Header()
    header["BGRATNAM"] = "BM"
    header["RGRATNAM"] = "RM2"
    header["IFUNAM"] = "Small"

    blue = resolving_power_from_header(header, "BLUE")
    red = resolving_power_from_header(header, "RED")

    assert blue is not None
    assert blue.value == 8000.0
    assert blue.grating == "BM"
    assert blue.slicer == "Small"
    assert not blue.approximate
    assert red is not None
    assert red.value == 5600.0
    assert red.approximate


def test_resolving_power_override_and_unknown_configuration() -> None:
    override = resolving_power_from_header({}, "BLUE", override=1750.0)

    assert override is not None
    assert override.value == 1750.0
    assert override.source == "command-line override"
    assert resolving_power_from_header({}, "BLUE") is None


def test_detector_flags_narrow_spike_but_not_lsf_width_line() -> None:
    rng = np.random.default_rng(81)
    wavelength = np.arange(4000.0, 4200.0)
    flux = rng.normal(0.0, 0.35, wavelength.size)
    pixels = np.arange(wavelength.size, dtype=float)
    expected_fwhm_pixels = 4.1
    gaussian_sigma = expected_fwhm_pixels / 2.355
    flux += 8.0 * np.exp(-0.5 * ((pixels - 60.0) / gaussian_sigma) ** 2)
    flux[130] += 8.0
    flux[170] -= 8.0

    detection = detect_cr_like_narrow_features(
        wavelength,
        flux,
        np.full(flux.shape, 0.35),
        resolving_power=1000.0,
    )

    assert [candidate.peak_index for candidate in detection.candidates] == [130, 170]
    assert [candidate.polarity for candidate in detection.candidates] == [
        "positive",
        "negative",
    ]
    assert detection.candidates[0].snr > 0
    assert detection.candidates[1].snr < 0
    for candidate in detection.candidates:
        assert candidate.width_ratio < 0.65
        assert candidate.start_index <= candidate.peak_index < candidate.stop_index
        json.dumps(candidate.to_dict())


def test_rejected_candidate_is_interpolated_with_inflated_uncertainty() -> None:
    rng = np.random.default_rng(14)
    wavelength = np.arange(4000.0, 4200.0)
    flux = rng.normal(0.0, 0.2, wavelength.size)
    flux[100] += 9.0
    sigma = np.full(flux.shape, 0.2)
    detection = detect_cr_like_narrow_features(
        wavelength,
        flux,
        sigma,
        resolving_power=1000.0,
    )
    candidate = detection.candidates[0]

    cleaned_flux, cleaned_sigma, details = interpolate_rejected_candidate(
        wavelength,
        flux,
        sigma,
        candidate,
        continuum=detection.continuum,
        noise=detection.noise,
    )

    start = candidate.start_index
    stop = candidate.stop_index
    expected = np.interp(
        wavelength[start:stop],
        [wavelength[start - 1], wavelength[stop]],
        [flux[start - 1], flux[stop]],
    )
    assert np.allclose(cleaned_flux[start:stop], expected)
    assert np.all(cleaned_sigma[start:stop] > sigma[start:stop])
    assert np.array_equal(cleaned_flux[:start], flux[:start])
    assert details["local_interpolation_scatter"] > 0

    _flux_without_input_sigma, estimated_sigma, estimated_details = (
        interpolate_rejected_candidate(
            wavelength,
            flux,
            None,
            candidate,
            continuum=detection.continuum,
            noise=detection.noise,
        )
    )
    assert np.all(np.isfinite(estimated_sigma))
    assert estimated_details["uncertainty_source"] == "local robust noise estimate"


def test_sequential_review_can_preview_redo_and_accept_result(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    from kcwi_pipeline.object_workflow import _review_spectral_cr_candidates

    wavelength = np.arange(4000.0, 4200.0)
    flux = np.zeros(wavelength.shape)
    flux[70] = 8.0
    flux[130] = -9.0
    sigma = np.ones(wavelength.shape)
    detection = detect_cr_like_narrow_features(
        wavelength,
        flux,
        sigma,
        resolving_power=1000.0,
    )
    assert len(detection.candidates) == 2

    def click_review_buttons() -> None:
        figure = plt.gcf()
        accept_button, remove_button = figure._kcwi_spectral_cr_widgets
        zoom_axis, overview_axis = figure._kcwi_spectral_cr_axes

        assert len(zoom_axis.lines[0].get_xdata()) < wavelength.size
        assert len(overview_axis.lines[0].get_xdata()) == wavelength.size
        assert any(
            patch.get_label() == "Upper-panel wavelength range"
            for patch in overview_axis.patches
        )
        assert any(
            line.get_label() == "Current candidate"
            for line in overview_axis.lines
        )

        # First pass: accept the first feature and remove the second.
        accept_button._observers.process("clicked", None)
        remove_button._observers.process("clicked", None)
        assert accept_button.label.get_text() == "Redo review"
        assert remove_button.label.get_text() == "Accept result"
        assert len(overview_axis.lines[0].get_xdata()) == wavelength.size

        # Redo discards the first pass. Remove the first, accept the second,
        # then approve the full resultant spectrum.
        accept_button._observers.process("clicked", None)
        assert accept_button.label.get_text() == "Accept line"
        remove_button._observers.process("clicked", None)
        accept_button._observers.process("clicked", None)
        remove_button._observers.process("clicked", None)

    monkeypatch.setattr(plt, "show", click_review_buttons)
    cleaned_flux, cleaned_sigma, decisions = _review_spectral_cr_candidates(
        "test",
        wavelength,
        flux,
        sigma,
        detection,
    )

    assert [item["decision"] for item in decisions] == ["removed", "accepted"]
    assert cleaned_flux[70] == 0.0
    assert cleaned_flux[130] == flux[130]
    assert cleaned_sigma is not None
    assert cleaned_sigma[70] > sigma[70]
