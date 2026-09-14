from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from kcwi_pipeline.object_workflow import (
    CALIBRATION_SCHEMA_VERSION,
    ICUBED_SPECTRUM_UNITS,
    _calibration_is_compatible,
    _exposure_time_from_header,
    _load_cube_product,
    _normalize_extracted_spectrum_for_product,
    _side_files,
)
from kcwi_pipeline.project import (
    cube_product_type,
    discover_cube_products,
    organize_project,
)


def _write_cube(path: Path, *, object_name: str, camera: str, in_extension: bool = False) -> None:
    header = fits.Header()
    header["OBJECT"] = object_name
    header["CAMERA"] = camera
    header["ELAPTIME"] = 30.0
    cube = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    if in_extension:
        hdul = fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(cube, name="SCI")])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(cube, header=header)])
    hdul.writeto(path)


def test_discovery_and_organization_support_both_cube_products(tmp_path) -> None:
    input_dir = tmp_path / "input"
    project_dir = tmp_path / "project"
    input_dir.mkdir()
    level1 = input_dir / "KB.target_icubed.fits"
    level2 = input_dir / "KB.target_icubes.fits"
    _write_cube(level1, object_name="target", camera="BLUE")
    _write_cube(level2, object_name="target", camera="BLUE")

    assert len(discover_cube_products(input_dir)) == 2
    assert cube_product_type(level1) == "icubed"
    assert cube_product_type(level2) == "icubes"

    with pytest.raises(ValueError, match="mixed cube products"):
        organize_project(input_dir, project_dir)


def test_cube_loader_accepts_cube_in_extension(tmp_path) -> None:
    path = tmp_path / "KR.target_icubed.fits"
    _write_cube(path, object_name="target", camera="RED", in_extension=True)

    cube, header, uncert, flags = _load_cube_product(path)

    assert cube.shape == (3, 4, 2)
    assert header["OBJECT"] == "target"
    assert header["ELAPTIME"] == 30.0
    assert uncert is None
    assert flags is None


def test_side_file_selection_is_explicit(tmp_path) -> None:
    blue_dir = tmp_path / "BLUE"
    blue_dir.mkdir()
    level1 = blue_dir / "KB.target_icubed.fits"
    level2 = blue_dir / "KB.target_icubes.fits"
    _write_cube(level1, object_name="target", camera="BLUE")
    _write_cube(level2, object_name="target", camera="BLUE")

    with pytest.raises(ValueError, match=r"both \*_icubed.fits and \*_icubes.fits"):
        _side_files(tmp_path, "BLUE")


def test_icubed_extraction_is_normalized_by_exposure_time() -> None:
    standard_header = fits.Header({"XPOSURE": 10.0})
    science_header = fits.Header({"XPOSURE": 1000.0})

    standard_rate, standard_sigma, standard_time, keyword, units = (
        _normalize_extracted_spectrum_for_product(
            "icubed",
            standard_header,
            np.array([100.0]),
            np.array([20.0]),
        )
    )
    science_rate, science_sigma, science_time, _, _ = (
        _normalize_extracted_spectrum_for_product(
            "icubed",
            science_header,
            np.array([10000.0]),
            np.array([2000.0]),
        )
    )

    assert standard_time == 10.0
    assert science_time == 1000.0
    assert keyword == "XPOSURE"
    assert units == ICUBED_SPECTRUM_UNITS
    assert standard_rate == pytest.approx([10.0])
    assert science_rate == pytest.approx([10.0])
    assert standard_sigma == pytest.approx([2.0])
    assert science_sigma == pytest.approx([2.0])

    reference_flux = 2.0
    sensitivity = reference_flux / standard_rate
    assert sensitivity * science_rate == pytest.approx([reference_flux])


def test_icubes_extraction_is_not_exposure_normalized() -> None:
    values = np.array([3.0, 4.0])
    sigma = np.array([0.3, 0.4])

    values_out, sigma_out, exposure_time, keyword, units = (
        _normalize_extracted_spectrum_for_product(
            "icubes",
            fits.Header({"XPOSURE": 1000.0}),
            values,
            sigma,
        )
    )

    assert np.array_equal(values_out, values)
    assert np.array_equal(sigma_out, sigma)
    assert exposure_time is None
    assert keyword is None
    assert units == "native_icubes_flux"


def test_icubed_exposure_time_fallback_and_validation() -> None:
    exposure_time, keyword = _exposure_time_from_header(
        fits.Header({"XPOSURE": 0.0, "ELAPTIME": 30.0})
    )
    assert exposure_time == 30.0
    assert keyword == "ELAPTIME"

    with pytest.raises(ValueError, match="no finite, positive exposure time"):
        _normalize_extracted_spectrum_for_product(
            "icubed",
            fits.Header(),
            np.array([1.0]),
            None,
            label="missing-time cube",
        )


def test_legacy_icubed_calibration_is_incompatible() -> None:
    legacy = {"product_type": "icubed"}
    current = {
        "product_type": "icubed",
        "exposure_normalized": True,
        "input_spectrum_units": ICUBED_SPECTRUM_UNITS,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
    }

    assert not _calibration_is_compatible(legacy, "icubed")
    assert _calibration_is_compatible(current, "icubed")
