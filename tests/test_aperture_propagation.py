from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from kcwi_pipeline.config import ApertureShape, TargetBackgroundApertures
from kcwi_pipeline.cosmic_rays import CosmicRayRejectionConfig
from kcwi_pipeline.object_workflow import (
    _ApertureTemplate,
    _SideExtractionResult,
    _transform_apertures_between_headers,
    extract_object,
)


def _celestial_header(
    *,
    crpix: tuple[float, float] = (20.0, 30.0),
    scale_arcsec: tuple[float, float] = (-1.0, 2.0),
) -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.cdelt = np.asarray(scale_arcsec) / 3600.0
    return wcs.to_header()


def _apertures() -> TargetBackgroundApertures:
    return TargetBackgroundApertures(
        target=ApertureShape("circle", (15.0, 25.0, 3.0)),
        background=ApertureShape(
            "ellipse_annulus",
            (15.0, 25.0, 4.0, 2.0, 7.0, 5.0, 0.3),
        ),
    )


def test_wcs_aperture_transfer_preserves_geometry_on_matching_scales() -> None:
    source = _celestial_header()
    destination = _celestial_header(crpix=(22.0, 29.0))

    transformed = _transform_apertures_between_headers(
        _apertures(),
        source,
        destination,
        (80, 80),
    )

    assert transformed.target.shape == "circle"
    assert np.allclose(transformed.target.params, (17.0, 24.0, 3.0), atol=1e-6)
    assert transformed.background.shape == "ellipse_annulus"
    assert np.allclose(transformed.background.params[:2], (17.0, 24.0), atol=1e-6)
    assert np.allclose(transformed.background.params[2:6], (4.0, 2.0, 7.0, 5.0), atol=1e-6)


def test_wcs_aperture_transfer_rescales_for_destination_pixels() -> None:
    transformed = _transform_apertures_between_headers(
        _apertures(),
        _celestial_header(scale_arcsec=(-1.0, 1.0)),
        _celestial_header(scale_arcsec=(-0.5, 0.5)),
        (100, 100),
    )

    assert transformed.target.shape == "circle"
    assert np.isclose(transformed.target.params[2], 6.0, rtol=1e-5)


def test_wcs_aperture_transfer_converts_circle_for_anisotropic_scale() -> None:
    transformed = _transform_apertures_between_headers(
        _apertures(),
        _celestial_header(scale_arcsec=(-1.0, 1.0)),
        _celestial_header(scale_arcsec=(-0.5, 1.0)),
        (100, 100),
    )

    assert transformed.target.shape == "ellipse"
    assert np.allclose(transformed.target.params[2:4], (6.0, 3.0), rtol=1e-5)


def test_wcs_aperture_transfer_handles_rectangle_and_circle_annulus() -> None:
    apertures = TargetBackgroundApertures(
        target=ApertureShape("rect", (15.0, 25.0, 8.0, 4.0, 0.2)),
        background=ApertureShape("circle_annulus", (15.0, 25.0, 5.0, 9.0)),
    )
    transformed = _transform_apertures_between_headers(
        apertures,
        _celestial_header(),
        _celestial_header(crpix=(22.0, 29.0)),
        (80, 80),
    )

    assert transformed.target.shape == "rect"
    assert np.allclose(transformed.target.params[:4], (17.0, 24.0, 8.0, 4.0), atol=1e-6)
    assert transformed.background.shape == "circle_annulus"
    assert np.allclose(
        transformed.background.params,
        (17.0, 24.0, 5.0, 9.0),
        atol=1e-6,
    )


def test_wcs_aperture_transfer_rejects_off_field_proposal() -> None:
    with pytest.raises(ValueError, match="outside the destination image"):
        _transform_apertures_between_headers(
            _apertures(),
            _celestial_header(),
            _celestial_header(crpix=(220.0, 230.0)),
            (80, 80),
        )


def test_extract_both_passes_first_side_aperture_to_second(monkeypatch, tmp_path) -> None:
    for side in ("BLUE", "RED"):
        side_dir = tmp_path / side
        side_dir.mkdir()
        (side_dir / f"{side.lower()}_icubes.fits").touch()
    calls: list[tuple[str, _ApertureTemplate | None, bool]] = []

    def fake_extract_side(object_dir: Path, side: str, **kwargs) -> _SideExtractionResult:
        initial = kwargs.get("initial_aperture_template")
        prefer_initial = bool(kwargs.get("prefer_initial_aperture_template"))
        calls.append((side, initial, prefer_initial))
        template = _ApertureTemplate(
            apertures=_apertures(),
            header=_celestial_header(),
            side=side,
            exposure_path=object_dir / side / f"{side}_icubes.fits",
        )
        return _SideExtractionResult(
            coadd_path=object_dir / f"{side}_coadd.flm",
            aperture_template=template,
        )

    monkeypatch.setattr("kcwi_pipeline.object_workflow._extract_side", fake_extract_side)
    monkeypatch.setattr(
        "kcwi_pipeline.object_workflow._apply_science_calibrations",
        lambda *args, **kwargs: None,
    )

    extract_object(
        tmp_path,
        standard=False,
        side="both",
        cr_config=CosmicRayRejectionConfig(workers=1),
        spectral_cr_review=False,
    )

    assert calls[0] == ("BLUE", None, False)
    assert calls[1][0] == "RED"
    assert calls[1][1] is not None
    assert calls[1][1].side == "BLUE"
    assert calls[1][2] is True
