import numpy as np

from kcwi_pipeline.config import ApertureShape, TargetBackgroundApertures
from kcwi_pipeline.calibration import apply_sensitivity_with_uncertainty
from kcwi_pipeline.object_workflow import (
    _coadd_1d_spectra,
    _extract_counts_with_uncert,
)
from kcwi_pipeline.uncertainty import linear_resample_with_uncertainty


def test_linear_resampling_propagates_variance_not_sigma() -> None:
    wavelength_in = np.array([0.0, 1.0])
    flux_in = np.array([10.0, 20.0])
    sigma_in = np.array([2.0, 4.0])

    flux_out, sigma_out = linear_resample_with_uncertainty(
        np.array([0.25]),
        wavelength_in,
        flux_in,
        sigma_in,
    )

    assert np.allclose(flux_out, [12.5])
    assert sigma_out is not None
    expected_sigma = np.sqrt((0.75 * 2.0) ** 2 + (0.25 * 4.0) ** 2)
    assert np.allclose(sigma_out, [expected_sigma])


def test_sensitivity_application_uses_propagated_resampling_uncertainty() -> None:
    wavelength_out, flux_out, sigma_out = apply_sensitivity_with_uncertainty(
        np.array([0.25]),
        np.array([2.0]),
        np.array([0.0, 1.0]),
        np.array([10.0, 20.0]),
        np.array([2.0, 4.0]),
    )

    assert np.allclose(wavelength_out, [0.25])
    assert np.allclose(flux_out, [25.0])
    assert sigma_out is not None
    assert np.allclose(sigma_out, [2.0 * np.sqrt(3.25)])


def test_extraction_uses_surviving_area_and_empirical_background_error(monkeypatch) -> None:
    target_mask = np.array([[1.0, 1.0], [0.0, 0.0]])
    background_mask = np.array([[0.0, 0.0], [1.0, 1.0]])

    def fake_aperture_mask(_ny, _nx, shape):
        return target_mask if shape.shape == "circle" else background_mask

    monkeypatch.setattr(
        "kcwi_pipeline.object_workflow.aperture_weight_mask",
        fake_aperture_mask,
    )
    apertures = TargetBackgroundApertures(
        target=ApertureShape("circle", (0.0, 0.0, 1.0)),
        background=ApertureShape("rect", (0.0, 0.0, 1.0, 1.0, 0.0)),
    )
    cube = np.array([[[10.0, 10.0], [0.0, 2.0]]])
    uncertainty = np.ones_like(cube)
    flags = np.zeros_like(cube, dtype=np.uint8)
    flags[0, 0, 1] = 1

    counts, sigma = _extract_counts_with_uncert(
        cube,
        uncertainty,
        flags,
        apertures,
    )

    # One target pixel survives: 10 - (background mean 1)*effective area 1.
    assert np.allclose(counts, [9.0])
    assert sigma is not None
    # Target variance is 1; empirical variance of the background mean is 1.
    assert np.allclose(sigma, [np.sqrt(2.0)])


def test_coadd_uncertainty_inflates_when_exposures_disagree() -> None:
    wavelength = np.array([5000.0])
    spectra = [
        (wavelength, np.array([0.0]), np.array([1.0])),
        (wavelength, np.array([10.0]), np.array([1.0])),
    ]

    _, flux, sigma, n_good = _coadd_1d_spectra(
        spectra,
        sigma_clip_value=100.0,
    )

    assert np.allclose(flux, [5.0])
    assert sigma is not None
    assert np.allclose(sigma, [5.0])
    assert np.array_equal(n_good, [2])


def test_coadd_keeps_formal_uncertainty_without_excess_scatter() -> None:
    wavelength = np.array([5000.0])
    spectra = [
        (wavelength, np.array([3.0]), np.array([1.0])),
        (wavelength, np.array([3.0]), np.array([1.0])),
    ]

    _, _flux, sigma, _n_good = _coadd_1d_spectra(spectra)

    assert sigma is not None
    assert np.allclose(sigma, [1.0 / np.sqrt(2.0)])
