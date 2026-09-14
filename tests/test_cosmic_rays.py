import json
import warnings
from dataclasses import replace

import numpy as np
from astropy.io import fits

from kcwi_pipeline.cosmic_rays import (
    CosmicRayRejectionConfig,
    reject_cosmic_rays,
    resolve_cr_workers,
    write_cr_cleaned_fits,
)


def _moving_spike_cube(*, repeat_across_slices: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    cube = rng.normal(0.0, 1.0, size=(42, 30, 3)).astype(np.float32)
    affected = np.zeros(cube.shape, dtype=bool)
    slice_indices = range(cube.shape[2]) if repeat_across_slices else (1,)

    for wavelength_pixel in range(8, 29):
        center = 6 + int(round(0.45 * (wavelength_pixel - 8)))
        for slice_index in slice_indices:
            cube[wavelength_pixel, center - 1, slice_index] += 2.8
            cube[wavelength_pixel, center, slice_index] += 14.0
            cube[wavelength_pixel, center + 1, slice_index] += 11.0
            cube[wavelength_pixel, center + 2, slice_index] += 2.8
            affected[wavelength_pixel, center - 1 : center + 3, slice_index] = True
    return cube, affected


def _config() -> CosmicRayRejectionConfig:
    return CosmicRayRejectionConfig(
        sigma=7.0,
        motion_sigma=4.0,
        grow_sigma=1.8,
        max_spatial_footprint=6,
        min_track_length=5,
        slice_shift_pixels=2,
        neighbor_veto_fraction=0.6,
        workers=1,
    )


def test_masks_full_moving_spike_track_in_one_slice() -> None:
    cube, affected = _moving_spike_cube(repeat_across_slices=False)
    result = reject_cosmic_rays(cube, np.ones_like(cube), config=_config())

    recovered = np.count_nonzero(result.mask & affected)
    assert recovered / np.count_nonzero(affected) > 0.9
    assert not np.any(result.mask[:, :, 0])
    assert not np.any(result.mask[:, :, 2])
    assert result.diagnostics["tracks_accepted"] == 1
    assert result.diagnostics["slices_with_tracks"] == 1
    assert result.runtime_seconds > 0
    assert np.nanmax(result.cleaned_cube[affected]) < np.nanmax(cube[affected])
    json.dumps(result.diagnostics)


def test_neighboring_slice_coherence_protects_tilted_emission() -> None:
    cube, _ = _moving_spike_cube(repeat_across_slices=True)
    result = reject_cosmic_rays(cube, np.ones_like(cube), config=_config())

    assert not np.any(result.mask)
    assert result.diagnostics["tracks_accepted"] == 0
    assert result.diagnostics["tracks_rejected_neighbor_coherence"] >= 3


def test_unrelated_neighbor_spikes_do_not_veto_a_cr_track() -> None:
    cube, affected = _moving_spike_cube(repeat_across_slices=False)
    for wavelength_pixel in range(8, 29):
        center = 6 + int(round(0.45 * (wavelength_pixel - 8)))
        offset = -3 if wavelength_pixel % 2 == 0 else 3
        cube[wavelength_pixel, center + offset, 0] += 14.0

    result = reject_cosmic_rays(cube, np.ones_like(cube), config=_config())

    recovered = np.count_nonzero(result.mask & affected)
    assert recovered / np.count_nonzero(affected) > 0.9
    assert result.diagnostics["tracks_accepted"] >= 1
    assert result.diagnostics["neighbor_track_comparisons"] > 0


def test_cr_crossing_bright_target_does_not_create_broad_trough() -> None:
    rng = np.random.default_rng(19)
    cube = rng.normal(0.0, 0.7, size=(100, 36, 3)).astype(np.float32)
    y = np.arange(cube.shape[1], dtype=float)
    for wavelength_pixel in range(cube.shape[0]):
        target_center = 15.0 + 0.015 * (wavelength_pixel - 50)
        cube[wavelength_pixel, :, 1] += 28.0 * np.exp(
            -0.5 * ((y - target_center) / 1.1) ** 2
        )
    uncontaminated = cube.copy()

    affected = np.zeros(cube.shape, dtype=bool)
    for wavelength_pixel in range(42, 59):
        center = 10 + int(round(0.65 * (wavelength_pixel - 42)))
        cube[wavelength_pixel, center - 1, 1] += 3.0
        cube[wavelength_pixel, center, 1] += 18.0
        cube[wavelength_pixel, center + 1, 1] += 14.0
        cube[wavelength_pixel, center + 2, 1] += 3.0
        affected[wavelength_pixel, center - 1 : center + 3, 1] = True

    config = CosmicRayRejectionConfig(
        sigma=6.0,
        motion_sigma=3.5,
        grow_sigma=1.5,
        max_spatial_footprint=8,
        min_track_length=4,
        max_track_span=32,
        slice_shift_pixels=2,
        workers=1,
    )
    result = reject_cosmic_rays(cube, np.ones_like(cube), config=config)

    recovered = np.count_nonzero(result.mask & affected)
    assert recovered / np.count_nonzero(affected) > 0.75
    assert not np.any(result.mask[:39])
    assert not np.any(result.mask[62:])
    clean_target = np.sum(result.cleaned_cube[:, 10:22, 1], axis=1)
    expected_target = np.sum(uncontaminated[:, 10:22, 1], axis=1)
    assert np.allclose(clean_target[:39], expected_target[:39])
    assert np.allclose(clean_target[62:], expected_target[62:])
    assert result.diagnostics["max_accepted_track_span"] <= 32


def test_track_span_limit_rejects_unusually_long_candidate() -> None:
    rng = np.random.default_rng(31)
    cube = rng.normal(0.0, 0.7, size=(90, 40, 3)).astype(np.float32)
    for wavelength_pixel in range(5, 81):
        center = 7 + int(round(0.2 * (wavelength_pixel - 5)))
        cube[wavelength_pixel, center, 1] += 16.0
        cube[wavelength_pixel, center + 1, 1] += 13.0

    config = CosmicRayRejectionConfig(
        sigma=6.0,
        motion_sigma=3.5,
        grow_sigma=2.0,
        max_spatial_footprint=6,
        max_track_span=32,
        workers=1,
    )
    result = reject_cosmic_rays(cube, np.ones_like(cube), config=config)

    assert not np.any(result.mask)
    assert result.diagnostics["tracks_rejected_max_span"] >= 1


def test_default_thresholds_recover_lower_amplitude_moving_track() -> None:
    rng = np.random.default_rng(113)
    cube = rng.normal(0.0, 0.35, size=(42, 30, 3)).astype(np.float32)
    affected = np.zeros(cube.shape, dtype=bool)
    for wavelength_pixel in range(8, 29):
        center = 6 + int(round(0.45 * (wavelength_pixel - 8)))
        cube[wavelength_pixel, center, 1] += 3.9
        cube[wavelength_pixel, center + 1, 1] += 3.1
        affected[wavelength_pixel, center : center + 2, 1] = True

    result = reject_cosmic_rays(
        cube,
        np.ones_like(cube),
        config=CosmicRayRejectionConfig(workers=1),
    )

    recovered = np.count_nonzero(result.mask & affected)
    assert recovered / np.count_nonzero(affected) > 0.9
    assert result.diagnostics["tracks_accepted"] == 1
    assert result.diagnostics["slices_with_tracks"] == 1


def test_cleaned_fits_records_track_method_and_runtime(tmp_path) -> None:
    cube, _ = _moving_spike_cube(repeat_across_slices=False)
    result = reject_cosmic_rays(cube, np.ones_like(cube), config=_config())
    source_path = tmp_path / "source.fits"
    output_path = tmp_path / "cleaned.fits"
    fits.PrimaryHDU(cube).writeto(source_path)

    write_cr_cleaned_fits(source_path, output_path, result)

    assert result.cleaned_uncert is not None
    assert np.all(result.cleaned_uncert[result.mask] >= 1.0)
    with fits.open(output_path) as hdul:
        assert hdul[0].header["CRMETH"] == "KCWI_DUAL"
        assert hdul[0].header["CRUUPD"]
        assert hdul[0].header["CRNTRK"] == 1
        assert hdul[0].header["CRWORK"] == 1
        assert hdul[0].header["CRTIME"] > 0
        assert hdul["CR_MASK"].data.shape == cube.shape
        assert hdul["UNCERT"].data.shape == cube.shape
        assert np.array_equal(
            hdul["UNCERT"].data,
            result.cleaned_uncert,
            equal_nan=True,
        )


def test_auto_worker_resolution_is_bounded_by_machine_and_slices() -> None:
    assert resolve_cr_workers(0, 28, cpu_count=10) == 8
    assert resolve_cr_workers(0, 28, cpu_count=4) == 2
    assert resolve_cr_workers(0, 28, cpu_count=2) == 1
    assert resolve_cr_workers(0, 3, cpu_count=10) == 3
    assert resolve_cr_workers(16, 5, cpu_count=2) == 5


def test_parallel_and_serial_rejection_are_identical() -> None:
    cube, _ = _moving_spike_cube(repeat_across_slices=False)
    serial_config = _config()
    parallel_config = replace(serial_config, workers=2)

    serial = reject_cosmic_rays(cube, np.ones_like(cube), config=serial_config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel = reject_cosmic_rays(cube, np.ones_like(cube), config=parallel_config)

    assert np.array_equal(parallel.mask, serial.mask)
    assert np.array_equal(parallel.cleaned_cube, serial.cleaned_cube, equal_nan=True)
    assert np.array_equal(parallel.cleaned_uncert, serial.cleaned_uncert, equal_nan=True)
    assert parallel.n_flagged == serial.n_flagged
    assert parallel.fraction_flagged == serial.fraction_flagged
    assert parallel.diagnostics["workers_used"] in {1, 2}
    fell_back = parallel.diagnostics["workers_used"] == 1
    assert parallel.diagnostics["parallel_fallbacks"] == int(fell_back)
    assert bool(caught) == fell_back
    for key in (
        "support_spikes",
        "seed_spikes",
        "tracks_fitted",
        "validated_support_spikes",
        "tracks_spectral_validated",
        "tracks_accepted",
        "final_mask_voxels",
    ):
        assert parallel.diagnostics[key] == serial.diagnostics[key]
