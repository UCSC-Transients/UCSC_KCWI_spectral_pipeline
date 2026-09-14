from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import FITSFixedWarning, WCS
from scipy.interpolate import UnivariateSpline

from .apertures import (
    WhiteLightRangeController,
    aperture_weight_mask,
    interactive_define_apertures,
    plot_apertures,
    review_apertures,
)
from .calibration import (
    apply_standard_telluric_correction,
    apply_sensitivity_with_uncertainty,
    build_standard_telluric_template,
    estimate_telluric_shift,
    plot_calibration_diagnostics,
    plot_o2_before_after,
    plot_o2_correction_diagnostic,
    plot_o2_template_diagnostic,
    scaled_o2_transmission,
    shifted_transmission,
)
from .config import ApertureShape, TargetBackgroundApertures
from .cosmic_rays import (
    CosmicRayRejectionConfig,
    config_to_dict,
    plot_cr_diagnostic,
    reject_cosmic_rays,
    resolve_cr_workers,
    write_cr_cleaned_fits,
    write_cr_mask_fits,
)
from .io import get_airmass_from_header, get_lambda_axis
from .join import concat_join, interactive_rescale_and_approve_flux, plot_join_diagnostic
from .project import find_project_root
from .spectral_cr import (
    ResolvingPowerEstimate,
    SpectralCRConfig,
    SpectralCRDetection,
    detect_cr_like_narrow_features,
    interpolate_rejected_candidate,
    resolving_power_from_header,
)
from .standard_flux import STANDARD_NAMES, list_standard_stars, reference_flux
from .utils import prompt, safe_filename
from .uncertainty import linear_resample_with_uncertainty


DEFAULT_SIDE_RANGES = {
    "BLUE": (3550.0, 5550.0),
    "RED": (5650.0, 8800.0),
}

FLUX_UNIT_LABEL = "1e-15 erg/s/cm^2/A"
TELLURIC_WINDOWS = [
    (5890.0, 5896.0),
    (6270.0, 6330.0),
    (6860.0, 6935.0),
    (7160.0, 7340.0),
    (7590.0, 7700.0),
    (8120.0, 8350.0),
]
TELLURIC_ALIGNMENT_WINDOWS = [(6860.0, 6935.0), (7590.0, 7700.0)]
O2_WINDOWS = TELLURIC_WINDOWS
TELLURIC_MIN_T = 0.02
TELLURIC_TEMPLATE_SMOOTH_S = 0.001
TELLURIC_AIRMASS_EXPONENT = 0.55
TELLURIC_MAX_SHIFT_A = 5.0
TELLURIC_SHIFT_STEP_A = 0.1
BACKGROUND_CLIP_SIGMA = 2.5
BACKGROUND_CLIP_MAXITERS = 5
COADD_CLIP_SIGMA = 2.0
COADD_CLIP_MAXITERS = 5
EXPOSURE_TIME_KEYS = ("XPOSURE", "ELAPTIME", "EXPTIME", "TELAPSE", "TTIME")
ICUBED_SPECTRUM_UNITS = "electron/s"
ICUBES_SPECTRUM_UNITS = "native_icubes_flux"
CALIBRATION_SCHEMA_VERSION = 2


@dataclass
class ExposureSpectrum:
    path: str
    side: str
    lam_path: str
    spectrum_path: str
    aperture_path: str
    airmass: Optional[float]
    exposure_time_seconds: Optional[float]
    exposure_time_keyword: Optional[str]
    spectrum_units: str
    cr_status: str
    cr_cleaned_path: Optional[str]
    cr_mask_path: Optional[str]
    cr_nvoxels: Optional[int]
    cr_fraction: Optional[float]
    cr_diagnostics: Dict[str, int]
    cr_runtime_seconds: Optional[float]


@dataclass(frozen=True)
class _ApertureTemplate:
    apertures: TargetBackgroundApertures
    header: fits.Header
    side: str
    exposure_path: Path


@dataclass(frozen=True)
class _SideExtractionResult:
    coadd_path: Path
    aperture_template: _ApertureTemplate


def _exposure_time_from_header(
    header: fits.Header,
    *,
    label: str = "KCWI exposure",
) -> Tuple[float, str]:
    """Return a positive exposure time and the FITS keyword that supplied it."""
    invalid = []
    for key in EXPOSURE_TIME_KEYS:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            invalid.append(f"{key}={header[key]!r}")
            continue
        if np.isfinite(value) and value > 0:
            return value, key
        invalid.append(f"{key}={header[key]!r}")

    detail = f" Invalid values: {', '.join(invalid)}." if invalid else ""
    keys = ", ".join(EXPOSURE_TIME_KEYS)
    raise ValueError(
        f"{label} is an *_icubed.fits product but has no finite, positive "
        f"exposure time in {keys}.{detail}"
    )


def _normalize_extracted_spectrum_for_product(
    product_type: str,
    header: fits.Header,
    values: np.ndarray,
    sigma: Optional[np.ndarray],
    *,
    label: str = "KCWI exposure",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[str], str]:
    """Convert icubed electrons to rates; leave DRP-calibrated icubes unchanged."""
    values_out = np.asarray(values, dtype=float)
    sigma_out = None if sigma is None else np.asarray(sigma, dtype=float)
    if product_type == "icubes":
        return values_out, sigma_out, None, None, ICUBES_SPECTRUM_UNITS
    if product_type != "icubed":
        raise ValueError(f"Unknown KCWI cube product type: {product_type}")

    exposure_time, keyword = _exposure_time_from_header(header, label=label)
    values_out = values_out / exposure_time
    if sigma_out is not None:
        sigma_out = sigma_out / exposure_time
    return values_out, sigma_out, exposure_time, keyword, ICUBED_SPECTRUM_UNITS


def _side_limits(side: str) -> Tuple[float, float]:
    return DEFAULT_SIDE_RANGES[side.upper()]


def _trim_side_arrays(side: str, lam: np.ndarray, *arrays: Optional[np.ndarray]):
    lo, hi = _side_limits(side)
    mask = np.isfinite(lam) & (lam >= lo) & (lam <= hi)
    if not np.any(mask):
        raise ValueError(f"No wavelengths for {side} in default range {lo:.0f}-{hi:.0f} A")
    out = [np.asarray(lam)[mask]]
    for arr in arrays:
        out.append(None if arr is None else np.asarray(arr)[mask])
    return tuple(out)


def _load_cube_product(path: Path) -> Tuple[np.ndarray, fits.Header, Optional[np.ndarray], Optional[np.ndarray]]:
    with fits.open(path, memmap=False) as hdul:
        science_hdu = next(
            (hdu for hdu in hdul if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 3),
            None,
        )
        if science_hdu is None:
            raise ValueError(f"No 3D science cube found in {path}")
        science = np.array(science_hdu.data, dtype=np.float32)
        header = science_hdu.header.copy()
        if science_hdu is not hdul[0]:
            for key in (
                "OBJECT", "TARGNAME", "CAMERA", "AIRMASS", "IMTYPE",
                "DATE-OBS", "DATE-BEG", "DATE-END", "EXPTIME", "ELAPTIME",
                "XPOSURE", "TELAPSE", "TTIME",
            ):
                if key in hdul[0].header and key not in header:
                    header[key] = hdul[0].header[key]
            for key, value in hdul[0].header.items():
                if key not in header and (
                    key.startswith(("WCSAXES", "CTYPE", "CRVAL", "CRPIX", "CDELT", "CD", "PC"))
                ):
                    header[key] = value
        uncert = None
        flags = None
        for hdu in hdul[1:]:
            name = str(hdu.header.get("EXTNAME", "")).upper()
            if name == "UNCERT" and hdu.data is not None:
                uncert = np.array(hdu.data, dtype=np.float32)
            elif name in {"MASK", "FLAGS"} and hdu.data is not None:
                arr = np.array(hdu.data)
                flags = arr if flags is None else np.bitwise_or(flags, arr)
    return science, header, uncert, flags


def _aperture_to_json(path: Path, aps: TargetBackgroundApertures) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aps.to_dict(), f, indent=2)


def _aperture_from_json(path: Path) -> TargetBackgroundApertures:
    with open(path, "r", encoding="utf-8") as f:
        return TargetBackgroundApertures.from_dict(json.load(f))


def _normalize_aperture_angle(theta: float) -> float:
    return float((theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


def _celestial_wcs(header: fits.Header, label: str) -> WCS:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            full_wcs = WCS(header)
    except Exception as exc:
        raise ValueError(f"{label} WCS could not be parsed: {exc}") from exc
    if not full_wcs.has_celestial:
        raise ValueError(f"{label} cube has no celestial WCS")
    celestial = full_wcs.celestial
    if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
        raise ValueError(f"{label} celestial WCS is not two-dimensional")
    return celestial


def _local_pixel_transform(
    source_wcs: WCS,
    destination_wcs: WCS,
    x: float,
    y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    step = 0.5
    source_pixels = np.array(
        [
            [x, y],
            [x + step, y],
            [x - step, y],
            [x, y + step],
            [x, y - step],
        ],
        dtype=float,
    )
    try:
        world = source_wcs.all_pix2world(source_pixels, 0)
        destination_pixels = np.asarray(
            destination_wcs.all_world2pix(world, 0),
            dtype=float,
        )
    except Exception as exc:
        raise ValueError(f"WCS coordinate transformation failed: {exc}") from exc
    if destination_pixels.shape != (5, 2) or not np.all(np.isfinite(destination_pixels)):
        raise ValueError("WCS produced non-finite destination pixel coordinates")

    center = destination_pixels[0]
    jacobian = np.column_stack(
        (
            (destination_pixels[1] - destination_pixels[2]) / (2.0 * step),
            (destination_pixels[3] - destination_pixels[4]) / (2.0 * step),
        )
    )
    if not np.all(np.isfinite(jacobian)) or abs(float(np.linalg.det(jacobian))) < 1e-8:
        raise ValueError("WCS pixel transformation is singular")
    return center, jacobian


def _ellipse_from_matrix(matrix: np.ndarray) -> Tuple[float, float, float]:
    axes, lengths, _ = np.linalg.svd(matrix)
    if not np.all(np.isfinite(lengths)) or float(lengths[-1]) <= 0:
        raise ValueError("WCS produced invalid aperture dimensions")
    theta = _normalize_aperture_angle(float(np.arctan2(axes[1, 0], axes[0, 0])))
    return float(lengths[0]), float(lengths[1]), theta


def _shape_center_and_jacobian(
    shape: ApertureShape,
    source_wcs: WCS,
    destination_wcs: WCS,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(shape.params) < 2:
        raise ValueError(f"Aperture shape {shape.shape!r} has no center")
    x, y = float(shape.params[0]), float(shape.params[1])
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("Aperture center is not finite")
    return _local_pixel_transform(source_wcs, destination_wcs, x, y)


def _transform_aperture_shape(
    shape: ApertureShape,
    source_wcs: WCS,
    destination_wcs: WCS,
) -> ApertureShape:
    center, jacobian = _shape_center_and_jacobian(shape, source_wcs, destination_wcs)
    x, y = (float(center[0]), float(center[1]))
    kind = shape.shape
    params = tuple(float(value) for value in shape.params)

    if kind == "circle":
        _, _, radius = params
        a, b, theta = _ellipse_from_matrix(jacobian * radius)
        if a / b <= 1.01:
            return ApertureShape("circle", (x, y, 0.5 * (a + b)))
        return ApertureShape("ellipse", (x, y, a, b, theta))

    if kind == "ellipse":
        _, _, a, b, theta = params
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        out_a, out_b, out_theta = _ellipse_from_matrix(
            jacobian @ rotation @ np.diag([a, b])
        )
        return ApertureShape("ellipse", (x, y, out_a, out_b, out_theta))

    if kind == "rect":
        _, _, width, height, theta = params
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        width_vector = jacobian @ (rotation[:, 0] * width)
        height_vector = jacobian @ (rotation[:, 1] * height)
        out_width = float(np.linalg.norm(width_vector))
        out_height = float(np.linalg.norm(height_vector))
        if not np.isfinite(out_width + out_height) or min(out_width, out_height) <= 0:
            raise ValueError("WCS produced invalid rectangular-aperture dimensions")
        out_theta = _normalize_aperture_angle(
            float(np.arctan2(width_vector[1], width_vector[0]))
        )
        return ApertureShape("rect", (x, y, out_width, out_height, out_theta))

    if kind == "circle_annulus":
        _, _, radius_in, radius_out = params
        axes, scales, _ = np.linalg.svd(jacobian)
        if not np.all(np.isfinite(scales)) or float(scales[-1]) <= 0:
            raise ValueError("WCS produced invalid annulus dimensions")
        if float(scales[0] / scales[1]) <= 1.01:
            scale = 0.5 * float(scales[0] + scales[1])
            return ApertureShape(
                "circle_annulus",
                (x, y, radius_in * scale, radius_out * scale),
            )
        theta = _normalize_aperture_angle(float(np.arctan2(axes[1, 0], axes[0, 0])))
        return ApertureShape(
            "ellipse_annulus",
            (
                x,
                y,
                radius_in * float(scales[0]),
                radius_in * float(scales[1]),
                radius_out * float(scales[0]),
                radius_out * float(scales[1]),
                theta,
            ),
        )

    if kind == "ellipse_annulus":
        _, _, a_in, b_in, a_out, b_out, theta = params
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        outer_matrix = jacobian @ rotation @ np.diag([a_out, b_out])
        out_a, out_b, out_theta = _ellipse_from_matrix(outer_matrix)
        inner_matrix = jacobian @ rotation @ np.diag([a_in, b_in])
        inner_covariance = inner_matrix @ inner_matrix.T
        major_direction = np.array([np.cos(out_theta), np.sin(out_theta)])
        minor_direction = np.array([-np.sin(out_theta), np.cos(out_theta)])
        out_a_in = float(np.sqrt(major_direction @ inner_covariance @ major_direction))
        out_b_in = float(np.sqrt(minor_direction @ inner_covariance @ minor_direction))
        out_a_in = min(out_a_in, out_a * (1.0 - 1e-6))
        out_b_in = min(out_b_in, out_b * (1.0 - 1e-6))
        if not np.isfinite(out_a_in + out_b_in) or min(out_a_in, out_b_in) <= 0:
            raise ValueError("WCS produced invalid annulus dimensions")
        return ApertureShape(
            "ellipse_annulus",
            (x, y, out_a_in, out_b_in, out_a, out_b, out_theta),
        )

    raise ValueError(f"Unsupported aperture shape for WCS transfer: {kind!r}")


def _transform_apertures_between_headers(
    apertures: TargetBackgroundApertures,
    source_header: fits.Header,
    destination_header: fits.Header,
    destination_shape: Tuple[int, int],
) -> TargetBackgroundApertures:
    source_wcs = _celestial_wcs(source_header, "source")
    destination_wcs = _celestial_wcs(destination_header, "destination")
    transformed = TargetBackgroundApertures(
        target=_transform_aperture_shape(apertures.target, source_wcs, destination_wcs),
        background=_transform_aperture_shape(
            apertures.background,
            source_wcs,
            destination_wcs,
        ),
    )

    ny, nx = (int(destination_shape[0]), int(destination_shape[1]))
    for label, shape in (
        ("target", transformed.target),
        ("background", transformed.background),
    ):
        x, y = float(shape.params[0]), float(shape.params[1])
        if not (0.0 <= x < nx and 0.0 <= y < ny):
            raise ValueError(
                f"transformed {label} center ({x:.2f}, {y:.2f}) is outside "
                f"the destination image ({nx} x {ny})"
            )
        try:
            area = float(np.sum(aperture_weight_mask(ny, nx, shape)))
        except Exception as exc:
            raise ValueError(f"transformed {label} aperture is invalid: {exc}") from exc
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"transformed {label} aperture does not overlap the destination image")
    return transformed


def _extract_counts_with_uncert(
    cube: np.ndarray,
    uncert: Optional[np.ndarray],
    flags: Optional[np.ndarray],
    aps: TargetBackgroundApertures,
    *,
    label: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    nz, ny, nx = cube.shape
    if uncert is not None and uncert.shape != cube.shape:
        raise ValueError(
            f"Uncertainty cube shape {uncert.shape} does not match science cube shape {cube.shape}"
        )
    if flags is not None and flags.shape != cube.shape:
        raise ValueError(
            f"Flag cube shape {flags.shape} does not match science cube shape {cube.shape}"
        )
    w_tgt_base = aperture_weight_mask(ny, nx, aps.target)
    w_bkg_base = aperture_weight_mask(ny, nx, aps.background)
    tgt_area = float(np.sum(w_tgt_base))
    bkg_area = float(np.sum(w_bkg_base))
    n_tgt_nonzero = int(np.count_nonzero(w_tgt_base > 0))
    n_bkg_nonzero = int(np.count_nonzero(w_bkg_base > 0))

    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Aperture effective area: target={tgt_area:.2f} spaxels "
        f"({n_tgt_nonzero} touched), background={bkg_area:.2f} spaxels "
        f"({n_bkg_nonzero} touched); exact fractional-pixel masks"
    )
    print(
        f"{prefix}Background estimator: sigma-clipped weighted mean per wavelength slice "
        f"(sigma={BACKGROUND_CLIP_SIGMA:g}, maxiters={BACKGROUND_CLIP_MAXITERS})."
    )
    if tgt_area < 3:
        print(f"{prefix}WARNING: target aperture effective area is only {tgt_area:.2f} spaxels; extraction may be unstable.")
    elif tgt_area < 10:
        print(f"{prefix}WARNING: target aperture effective area is only {tgt_area:.2f} spaxels; check aperture placement/size.")
    if bkg_area <= 0:
        print(f"{prefix}WARNING: background aperture has zero effective area; no background will be subtracted.")
    elif bkg_area < 10:
        print(f"{prefix}WARNING: background effective area is only {bkg_area:.2f} spaxels; background may be noisy.")
    elif bkg_area < 30:
        print(f"{prefix}NOTE: background effective area is {bkg_area:.2f} spaxels; consider a larger background region if feasible.")

    counts = np.zeros(nz, dtype=float)
    sigma = np.full(nz, np.nan, dtype=float) if uncert is not None else None
    background_inflated = 0
    background_inflation_factors: List[float] = []

    for k in range(nz):
        img = cube[k, :, :]
        bad = np.zeros((ny, nx), dtype=bool)
        if flags is not None:
            bad |= flags[k, :, :] != 0

        w_tgt = np.where(bad, 0.0, w_tgt_base)
        w_bkg = np.where(bad, 0.0, w_bkg_base)
        finite_tgt = np.isfinite(img) & (w_tgt > 0)
        finite_bkg = np.isfinite(img) & (w_bkg > 0)

        tgt_sum = np.nansum(w_tgt[finite_tgt] * img[finite_tgt])
        effective_tgt_area = float(np.sum(w_tgt[finite_tgt]))
        bkg = 0.0
        bkg_var_mean = 0.0
        if np.any(finite_bkg):
            bkg_vals = img[finite_bkg].astype(float)
            bkg_weights = w_bkg[finite_bkg].astype(float)
            clipped = sigma_clip(
                bkg_vals,
                sigma=BACKGROUND_CLIP_SIGMA,
                maxiters=BACKGROUND_CLIP_MAXITERS,
                masked=True,
            )
            keep = ~np.ma.getmaskarray(clipped)
            if np.any(keep):
                kept_values = bkg_vals[keep]
                kept_weights = bkg_weights[keep]
                sum_weights = float(np.sum(kept_weights))
                sum_weights_squared = float(np.sum(kept_weights ** 2))
                bkg = float(np.average(kept_values, weights=kept_weights))

                empirical_var_mean = 0.0
                if sum_weights > 0 and sum_weights_squared > 0:
                    effective_n = sum_weights ** 2 / sum_weights_squared
                    sample_denom = sum_weights - sum_weights_squared / sum_weights
                    if effective_n > 1 and sample_denom > 0:
                        sample_variance = float(
                            np.sum(kept_weights * (kept_values - bkg) ** 2)
                            / sample_denom
                        )
                        if np.isfinite(sample_variance) and sample_variance >= 0:
                            empirical_var_mean = sample_variance / effective_n

                if uncert is not None:
                    kept_sigma = np.abs(
                        uncert[k, :, :].astype(float)[finite_bkg][keep]
                    )
                    valid_sigma = np.isfinite(kept_sigma)
                    formal_var_mean = np.nan
                    if sum_weights > 0 and np.all(valid_sigma):
                        formal_var_mean = float(
                            np.sum((kept_weights ** 2) * kept_sigma ** 2)
                            / sum_weights ** 2
                        )
                    if np.isfinite(formal_var_mean):
                        bkg_var_mean = max(formal_var_mean, empirical_var_mean)
                        if empirical_var_mean > formal_var_mean and formal_var_mean > 0:
                            background_inflated += 1
                            background_inflation_factors.append(
                                float(np.sqrt(empirical_var_mean / formal_var_mean))
                            )
                    else:
                        bkg_var_mean = empirical_var_mean
        counts[k] = tgt_sum - bkg * effective_tgt_area

        if uncert is not None:
            sigma_image = np.abs(uncert[k, :, :].astype(float))
            target_sigma = sigma_image[finite_tgt]
            if np.all(np.isfinite(target_sigma)):
                var_tgt = float(
                    np.sum((w_tgt[finite_tgt] ** 2) * target_sigma ** 2)
                )
                sigma[k] = np.sqrt(
                    var_tgt + (effective_tgt_area ** 2) * bkg_var_mean
                )

    if background_inflated:
        median_factor = float(np.median(background_inflation_factors))
        print(
            f"{prefix}Empirical background scatter exceeded the formal background "
            f"uncertainty in {background_inflated}/{nz} wavelength slices "
            f"(median sigma inflation {median_factor:.2f}x in those slices)."
        )

    return counts, sigma


def _save_spectrum(path: Path, lam: np.ndarray, y: np.ndarray, sigma: Optional[np.ndarray], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unit_suffix = f"_{FLUX_UNIT_LABEL.replace(' ', '_').replace('/', '_per_').replace('^', '')}" if header == "flux" else ""
    if sigma is None:
        arr = np.c_[lam, y]
        hdr = f"lambda_A  {header}{unit_suffix}"
    else:
        arr = np.c_[lam, y, sigma]
        hdr = f"lambda_A  {header}{unit_suffix}  sigma_{header}{unit_suffix}"
    np.savetxt(path, arr, header=hdr)


def _plot_spectrum_png(
    path: Path,
    title: str,
    lam: np.ndarray,
    flux: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    *,
    show: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(lam, flux, lw=1.0, label="Flux")
    if sigma is not None:
        lo = flux - sigma
        hi = flux + sigma
        ax.fill_between(
            lam,
            lo,
            hi,
            color="0.65",
            alpha=0.32,
            linewidth=0,
            label="1 sigma uncertainty",
        )
    finite = np.isfinite(lam) & np.isfinite(flux)
    if np.any(finite):
        y0, y1 = np.nanpercentile(flux[finite], [1, 99])
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            pad = 0.1 * (y1 - y0)
            ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel(f"Flux ({FLUX_UNIT_LABEL})")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_joined_spectrum_png(
    path: Path,
    title: str,
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    sigma_blue: Optional[np.ndarray],
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    sigma_red: Optional[np.ndarray],
    *,
    show: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(lam_blue, flux_blue, lw=1.0, color="tab:blue", label="BLUE")
    ax.plot(lam_red, flux_red, lw=1.0, color="tab:red", label="RED")
    if sigma_blue is not None:
        ax.fill_between(
            lam_blue,
            flux_blue - sigma_blue,
            flux_blue + sigma_blue,
            color="0.65",
            alpha=0.30,
            linewidth=0,
            label="1 sigma uncertainty",
        )
    if sigma_red is not None:
        ax.fill_between(
            lam_red,
            flux_red - sigma_red,
            flux_red + sigma_red,
            color="0.65",
            alpha=0.30,
            linewidth=0,
            label="_nolegend_",
        )
    both_flux = np.concatenate([np.asarray(flux_blue, dtype=float), np.asarray(flux_red, dtype=float)])
    finite = np.isfinite(both_flux)
    if np.any(finite):
        y0, y1 = np.nanpercentile(both_flux[finite], [1, 99])
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            pad = 0.1 * (y1 - y0)
            ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel(f"Flux ({FLUX_UNIT_LABEL})")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _coadd_1d_spectra(
    spectra: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    *,
    sigma_clip_value: float = COADD_CLIP_SIGMA,
    maxiters: int = COADD_CLIP_MAXITERS,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if not spectra:
        raise ValueError("No spectra to coadd")

    lam_ref = spectra[0][0]
    values = []
    sigmas = []
    have_sigma = all(s[2] is not None for s in spectra)

    for lam, y, sig in spectra:
        if lam.shape == lam_ref.shape and np.allclose(lam, lam_ref, rtol=0.0, atol=1e-7):
            y_i = y
            sig_i = sig
        else:
            y_i, sig_i = linear_resample_with_uncertainty(
                lam_ref,
                lam,
                y,
                sig,
            )
        values.append(y_i)
        if have_sigma:
            sigmas.append(sig_i)

    stack = np.asarray(values, dtype=float)
    good = np.isfinite(stack)
    if have_sigma:
        sigma_stack = np.asarray(sigmas, dtype=float)
        good &= np.isfinite(sigma_stack) & (sigma_stack > 0)
    else:
        sigma_stack = None

    clipped = sigma_clip(np.ma.array(stack, mask=~good), sigma=sigma_clip_value, maxiters=maxiters, axis=0)
    good = ~np.ma.getmaskarray(clipped)
    n_good = np.sum(good, axis=0).astype(np.int16)

    out = np.full(lam_ref.shape, np.nan, dtype=float)
    out_sigma = np.full(lam_ref.shape, np.nan, dtype=float) if have_sigma else None

    if have_sigma and sigma_stack is not None:
        var = sigma_stack ** 2
        weights = np.zeros_like(var)
        weights[good] = 1.0 / var[good]
        sumw = np.sum(weights, axis=0)
        valid = sumw > 0
        out[valid] = np.sum(np.where(good, stack, 0.0) * weights, axis=0)[valid] / sumw[valid]
        formal_sigma = np.sqrt(1.0 / sumw[valid])
        residual = stack[:, valid] - out[valid][None, :]
        chi_square = np.sum(
            np.where(good[:, valid], residual ** 2 * weights[:, valid], 0.0),
            axis=0,
        )
        degrees_of_freedom = n_good[valid].astype(float) - 1.0
        reduced_chi_square = np.ones(formal_sigma.shape, dtype=float)
        can_inflate = degrees_of_freedom > 0
        reduced_chi_square[can_inflate] = (
            chi_square[can_inflate] / degrees_of_freedom[can_inflate]
        )
        inflation = np.sqrt(np.maximum(1.0, reduced_chi_square))
        out_sigma[valid] = formal_sigma * inflation
    else:
        valid = n_good > 0
        out[valid] = np.sum(np.where(good, stack, 0.0), axis=0)[valid] / n_good[valid]

    return lam_ref, out, out_sigma, n_good


def _plot_1d_coadd_diagnostic(
    path: Path,
    title: str,
    spectra: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    lam_coadd: np.ndarray,
    flux_coadd: np.ndarray,
    sigma_coadd: Optional[np.ndarray],
    n_good: np.ndarray,
    *,
    show: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_spec, ax_n) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    _draw_1d_coadd_diagnostic_axes(ax_spec, ax_n, title, spectra, lam_coadd, flux_coadd, sigma_coadd, n_good)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _draw_1d_coadd_diagnostic_axes(
    ax_spec,
    ax_n,
    title: str,
    spectra: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    lam_coadd: np.ndarray,
    flux_coadd: np.ndarray,
    sigma_coadd: Optional[np.ndarray],
    n_good: np.ndarray,
) -> None:
    ax_spec.clear()
    ax_n.clear()

    finite_coadd = np.isfinite(flux_coadd)
    if np.any(finite_coadd):
        ylo, yhi = np.nanpercentile(flux_coadd[finite_coadd], [5, 95])
        offset_step = yhi - ylo
        if not np.isfinite(offset_step) or offset_step <= 0:
            offset_step = np.nanstd(flux_coadd[finite_coadd])
        if not np.isfinite(offset_step) or offset_step <= 0:
            offset_step = 1.0
    else:
        offset_step = 1.0
    offset_step *= 1.25

    for i, (lam, y, _sig) in enumerate(spectra):
        if lam.shape == lam_coadd.shape and np.allclose(lam, lam_coadd, rtol=0.0, atol=1e-7):
            y_plot = y
        else:
            y_plot = np.interp(lam_coadd, lam, y, left=np.nan, right=np.nan)
        offset = i * offset_step
        ax_spec.plot(lam_coadd, y_plot + offset, lw=0.75, alpha=0.75, label=f"Exposure {i + 1}")

    coadd_offset = len(spectra) * offset_step
    ax_spec.plot(lam_coadd, flux_coadd + coadd_offset, lw=1.5, color="black", label="Sigma-clipped coadd")
    if sigma_coadd is not None:
        ax_spec.fill_between(
            lam_coadd,
            flux_coadd + coadd_offset - sigma_coadd,
            flux_coadd + coadd_offset + sigma_coadd,
            color="black",
            alpha=0.14,
            linewidth=0,
            label="Coadd 1 sigma",
        )

    ax_spec.set_ylabel("Counts + vertical offset")
    ax_spec.set_title(title)
    ax_spec.grid(alpha=0.2)
    if len(spectra) <= 8:
        ax_spec.legend(fontsize=8, ncol=2)
    else:
        ax_spec.text(
            0.01,
            0.98,
            f"{len(spectra)} exposures + coadd",
            transform=ax_spec.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )

    ax_n.step(lam_coadd, n_good, where="mid", color="tab:blue", lw=1.0)
    ax_n.set_ylim(0, max(len(spectra), int(np.nanmax(n_good)) if n_good.size else 1) + 0.5)
    ax_n.set_ylabel("N used")
    ax_n.set_xlabel("Wavelength (A)")
    ax_n.grid(alpha=0.2)


def _review_1d_coadd_sigma_clip(
    path: Path,
    title: str,
    spectra: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    *,
    initial_sigma: float = COADD_CLIP_SIGMA,
    maxiters: int = COADD_CLIP_MAXITERS,
    show: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    lam, flux, sigma, n_good = _coadd_1d_spectra(
        spectra,
        sigma_clip_value=initial_sigma,
        maxiters=maxiters,
    )
    if not show:
        _plot_1d_coadd_diagnostic(path, title, spectra, lam, flux, sigma, n_good, show=False)
        return lam, flux, sigma, n_good, float(initial_sigma)

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "approved": False,
        "sigma_clip": float(initial_sigma),
        "lam": lam,
        "flux": flux,
        "sigma": sigma,
        "n_good": n_good,
    }
    fig, (ax_spec, ax_n) = plt.subplots(2, 1, figsize=(11, 7.1), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.88, hspace=0.08)

    ax_slider = fig.add_axes([0.16, 0.085, 0.55, 0.028])
    ax_reset = fig.add_axes([0.74, 0.065, 0.09, 0.06])
    ax_approve = fig.add_axes([0.85, 0.065, 0.10, 0.06])
    slider = Slider(ax_slider, "Clip sigma", 0.5, 6.0, valinit=float(initial_sigma), valstep=0.1)
    reset_button = Button(ax_reset, "Reset")
    approve_button = Button(ax_approve, "Approve")

    def recompute(sigma_clip_value: float) -> None:
        lam_i, flux_i, sigma_i, n_good_i = _coadd_1d_spectra(
            spectra,
            sigma_clip_value=float(sigma_clip_value),
            maxiters=maxiters,
        )
        state["sigma_clip"] = float(sigma_clip_value)
        state["lam"] = lam_i
        state["flux"] = flux_i
        state["sigma"] = sigma_i
        state["n_good"] = n_good_i
        full_title = (
            f"{title}\nclip sigma={float(sigma_clip_value):.1f}, maxiters={maxiters}; "
            "a/Enter=approve, q=abort"
        )
        _draw_1d_coadd_diagnostic_axes(ax_spec, ax_n, full_title, spectra, lam_i, flux_i, sigma_i, n_good_i)
        fig.canvas.draw_idle()

    def approve(_event=None) -> None:
        state["approved"] = True
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def reset(_event=None) -> None:
        slider.set_val(float(initial_sigma))

    def on_key(event) -> None:
        if event.key in ("a", "enter", "return"):
            approve()
        elif event.key == "q":
            plt.close(fig)

    slider.on_changed(recompute)
    reset_button.on_clicked(reset)
    approve_button.on_clicked(approve)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    fig._kcwi_coadd_widgets = (slider, reset_button, approve_button)

    recompute(float(initial_sigma))
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if not state["approved"]:
        raise RuntimeError("Sigma-clipped coadd was not approved")

    return (
        state["lam"],
        state["flux"],
        state["sigma"],
        state["n_good"],
        float(state["sigma_clip"]),
    )


def _candidate_plot_edges(
    wavelength: np.ndarray,
    start: int,
    stop: int,
) -> Tuple[float, float]:
    left = 0.5 * (wavelength[start - 1] + wavelength[start])
    right = 0.5 * (wavelength[stop - 1] + wavelength[stop])
    return float(min(left, right)), float(max(left, right))


def _review_spectral_cr_candidates(
    title: str,
    wavelength: np.ndarray,
    flux: np.ndarray,
    sigma: Optional[np.ndarray],
    detection: SpectralCRDetection,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, object]]]:
    candidates = detection.candidates
    if not candidates:
        return flux.copy(), None if sigma is None else sigma.copy(), []

    state: Dict[str, object] = {
        "index": 0,
        "phase": "candidates",
        "completed": False,
        "flux": flux.copy(),
        "sigma": None if sigma is None else sigma.copy(),
        "decisions": [],
    }
    fig, (ax_flux, ax_overview) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.0),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.84, hspace=0.08)
    accept_ax = fig.add_axes([0.63, 0.055, 0.14, 0.065])
    remove_ax = fig.add_axes([0.79, 0.055, 0.17, 0.065])
    accept_button = Button(accept_ax, "Accept line")
    remove_button = Button(remove_ax, "Remove as CR")

    def draw_decision_spans(ax) -> None:
        labels_used = set()
        decisions = state["decisions"]
        assert isinstance(decisions, list)
        for decision in decisions:
            start = int(decision["start_index"])
            stop = int(decision["stop_index"])
            left, right = _candidate_plot_edges(wavelength, start, stop)
            removed = decision.get("decision") == "removed"
            label = "Removed candidate" if removed else "Accepted line"
            ax.axvspan(
                left,
                right,
                color="tab:red" if removed else "tab:green",
                alpha=0.18 if removed else 0.11,
                label=label if label not in labels_used else None,
            )
            labels_used.add(label)

    def draw_full_spectrum_overview(
        current_flux: np.ndarray,
        *,
        zoom_limits: Optional[Tuple[float, float]],
        candidate_wavelength: Optional[float] = None,
    ) -> None:
        ax_overview.clear()
        ax_overview.plot(
            wavelength,
            current_flux,
            color="black",
            lw=0.75,
            label="Full spectrum",
        )
        draw_decision_spans(ax_overview)
        if zoom_limits is not None:
            left, right = zoom_limits
            ax_overview.axvspan(
                left,
                right,
                color="tab:blue",
                alpha=0.18,
                label="Upper-panel wavelength range",
            )
        if candidate_wavelength is not None:
            ax_overview.axvline(
                candidate_wavelength,
                color="tab:red",
                lw=0.9,
                alpha=0.85,
                label="Current candidate",
            )
        finite_wavelength = wavelength[np.isfinite(wavelength)]
        if finite_wavelength.size:
            ax_overview.set_xlim(
                float(np.min(finite_wavelength)),
                float(np.max(finite_wavelength)),
            )
        ax_overview.set_ylabel("Counts")
        ax_overview.set_xlabel("Wavelength (A)")
        ax_overview.grid(alpha=0.2)
        ax_overview.legend(fontsize=7, ncol=3, loc="best")

    def redraw_candidate() -> None:
        index = int(state["index"])
        candidate = candidates[index]
        current_flux = np.asarray(state["flux"], dtype=float)
        current_sigma = state["sigma"]
        radius = max(int(np.ceil(4.0 * candidate.expected_fwhm_pixels)), 10)
        lo = max(candidate.peak_index - radius, 0)
        hi = min(candidate.peak_index + radius + 1, wavelength.size)
        region = slice(lo, hi)
        mask_lo, mask_hi = _candidate_plot_edges(
            wavelength,
            candidate.start_index,
            candidate.stop_index,
        )
        lsf_half_width = 0.5 * candidate.expected_fwhm_angstrom

        ax_flux.clear()
        ax_flux.plot(
            wavelength[region],
            current_flux[region],
            color="black",
            lw=1.0,
            marker=".",
            ms=4,
            label="Coadded spectrum",
        )
        ax_flux.plot(
            wavelength[region],
            detection.continuum[region],
            color="0.45",
            lw=1.0,
            linestyle="--",
            label="Local continuum",
        )
        uncertainty = detection.noise if current_sigma is None else np.asarray(current_sigma)
        ax_flux.fill_between(
            wavelength[region],
            current_flux[region] - uncertainty[region],
            current_flux[region] + uncertainty[region],
            color="0.5",
            alpha=0.16,
            linewidth=0,
            label="1 sigma",
        )
        ax_flux.axvspan(
            candidate.wavelength - lsf_half_width,
            candidate.wavelength + lsf_half_width,
            color="tab:blue",
            alpha=0.10,
            label="Expected LSF FWHM",
        )
        ax_flux.axvspan(
            mask_lo,
            mask_hi,
            color="tab:red",
            alpha=0.18,
            label="Proposed removal",
        )
        ax_flux.axvline(candidate.wavelength, color="tab:red", lw=0.9, alpha=0.8)
        ax_flux.set_ylabel("Counts")
        ax_flux.grid(alpha=0.2)
        ax_flux.legend(fontsize=8, ncol=2, loc="best")
        zoom_limits = (
            float(min(wavelength[lo], wavelength[hi - 1])),
            float(max(wavelength[lo], wavelength[hi - 1])),
        )
        ax_flux.set_xlim(*zoom_limits)
        draw_full_spectrum_overview(
            current_flux,
            zoom_limits=zoom_limits,
            candidate_wavelength=candidate.wavelength,
        )
        fig.suptitle(
            f"{title}: {candidate.polarity} candidate {index + 1}/{len(candidates)} "
            f"at {candidate.wavelength:.2f} A\n"
            f"S/N={candidate.snr:.1f}; measured FWHM={candidate.measured_fwhm_pixels:.2f} px; "
            f"expected={candidate.expected_fwhm_pixels:.2f} px; ratio={candidate.width_ratio:.2f}"
        )
        accept_button.label.set_text("Accept line")
        remove_button.label.set_text("Remove as CR")
        fig.canvas.draw_idle()

    def redraw_result() -> None:
        current_flux = np.asarray(state["flux"], dtype=float)
        current_sigma = state["sigma"]
        ax_flux.clear()
        ax_flux.plot(
            wavelength,
            current_flux,
            color="black",
            lw=0.9,
            label="Resultant spectrum",
        )
        if current_sigma is not None:
            uncertainty = np.asarray(current_sigma, dtype=float)
            ax_flux.fill_between(
                wavelength,
                current_flux - uncertainty,
                current_flux + uncertainty,
                color="0.5",
                alpha=0.15,
                linewidth=0,
                label="Resultant 1 sigma",
            )
        draw_decision_spans(ax_flux)
        ax_flux.set_ylabel("Counts")
        ax_flux.grid(alpha=0.2)
        ax_flux.legend(fontsize=8, ncol=3, loc="best")
        draw_full_spectrum_overview(current_flux, zoom_limits=None)
        decisions = state["decisions"]
        assert isinstance(decisions, list)
        removed_count = sum(item.get("decision") == "removed" for item in decisions)
        fig.suptitle(
            f"{title}: full resultant spectrum\n"
            f"Removed {removed_count}/{len(candidates)} candidates"
        )
        accept_button.label.set_text("Redo review")
        remove_button.label.set_text("Accept result")
        fig.canvas.draw_idle()

    def finish_or_advance() -> None:
        state["index"] = int(state["index"]) + 1
        if int(state["index"]) >= len(candidates):
            state["phase"] = "result"
            redraw_result()
        else:
            redraw_candidate()

    def accept(_event=None) -> None:
        candidate = candidates[int(state["index"])]
        decisions = state["decisions"]
        assert isinstance(decisions, list)
        decisions.append({**candidate.to_dict(), "decision": "accepted"})
        finish_or_advance()

    def remove(_event=None) -> None:
        candidate = candidates[int(state["index"])]
        cleaned_flux, cleaned_sigma, interpolation = interpolate_rejected_candidate(
            wavelength,
            np.asarray(state["flux"], dtype=float),
            state["sigma"],
            candidate,
            continuum=detection.continuum,
            noise=detection.noise,
        )
        state["flux"] = cleaned_flux
        state["sigma"] = cleaned_sigma
        decisions = state["decisions"]
        assert isinstance(decisions, list)
        decisions.append(
            {
                **candidate.to_dict(),
                "decision": "removed",
                "interpolation": interpolation,
            }
        )
        finish_or_advance()

    def redo_review() -> None:
        state["index"] = 0
        state["phase"] = "candidates"
        state["flux"] = flux.copy()
        state["sigma"] = None if sigma is None else sigma.copy()
        state["decisions"] = []
        redraw_candidate()

    def accept_result() -> None:
        state["completed"] = True
        plt.close(fig)

    def left_button_action(_event=None) -> None:
        if state["phase"] == "result":
            redo_review()
        else:
            accept()

    def right_button_action(_event=None) -> None:
        if state["phase"] == "result":
            accept_result()
        else:
            remove()

    def on_key(event) -> None:
        if state["phase"] == "result":
            if event.key in ("a", "enter", "return"):
                accept_result()
            elif event.key == "r":
                redo_review()
        elif event.key in ("a", "enter", "return"):
            accept()
        elif event.key in ("r", "backspace", "delete"):
            remove()

    accept_button.on_clicked(left_button_action)
    remove_button.on_clicked(right_button_action)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    fig._kcwi_spectral_cr_widgets = (accept_button, remove_button)
    fig._kcwi_spectral_cr_axes = (ax_flux, ax_overview)
    redraw_candidate()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if not state["completed"]:
        raise RuntimeError("Spectral CR candidate review was not completed")
    return state["flux"], state["sigma"], state["decisions"]


def _plot_spectral_cr_review_summary(
    path: Path,
    title: str,
    wavelength: np.ndarray,
    original_flux: np.ndarray,
    cleaned_flux: np.ndarray,
    cleaned_sigma: Optional[np.ndarray],
    decisions: List[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(wavelength, original_flux, color="0.65", lw=0.7, label="Before review")
    ax.plot(wavelength, cleaned_flux, color="black", lw=1.0, label="After review")
    if cleaned_sigma is not None:
        ax.fill_between(
            wavelength,
            cleaned_flux - cleaned_sigma,
            cleaned_flux + cleaned_sigma,
            color="0.5",
            alpha=0.15,
            linewidth=0,
            label="After-review 1 sigma",
        )
    labels_used = set()
    for decision in decisions:
        start = int(decision["start_index"])
        stop = int(decision["stop_index"])
        left, right = _candidate_plot_edges(wavelength, start, stop)
        removed = decision.get("decision") == "removed"
        label = "Removed candidate" if removed else "Accepted line"
        ax.axvspan(
            left,
            right,
            color="tab:red" if removed else "tab:green",
            alpha=0.18 if removed else 0.11,
            label=label if label not in labels_used else None,
        )
        labels_used.add(label)
    finite = np.isfinite(cleaned_flux)
    if np.any(finite):
        y0, y1 = np.nanpercentile(cleaned_flux[finite], [1, 99])
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            pad = 0.1 * (y1 - y0)
            ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _side_product_files(object_dir: Path, side: str) -> Tuple[str, List[Path]]:
    icubed = sorted((object_dir / side).glob("*_icubed.fits"))
    icubes = sorted((object_dir / side).glob("*_icubes.fits"))
    if icubed and icubes:
        raise ValueError(
            f"{object_dir / side} contains both *_icubed.fits and *_icubes.fits. "
            "All standards and science targets must use one consistent cube type."
        )
    if icubed:
        return "icubed", icubed
    if icubes:
        return "icubes", icubes
    return "", []


def _side_files(object_dir: Path, side: str, product_type: Optional[str] = None) -> List[Path]:
    detected_type, files = _side_product_files(object_dir, side)
    if product_type is not None and product_type != detected_type:
        return []
    return files


def _saved_aperture_template(
    object_dir: Path,
    side: str,
) -> Optional[_ApertureTemplate]:
    for exposure_path in _side_files(object_dir, side):
        aperture_path = object_dir / "apertures" / side / f"{exposure_path.stem}_aperture.json"
        if not aperture_path.exists():
            continue
        try:
            apertures = _aperture_from_json(aperture_path)
            header = fits.getheader(exposure_path, 0)
        except Exception as exc:
            print(
                f"[{object_dir.name} {side}] WARNING: Could not load saved aperture "
                f"template from {aperture_path}: {exc}"
            )
            continue
        return _ApertureTemplate(
            apertures=apertures,
            header=header,
            side=side,
            exposure_path=exposure_path,
        )
    return None


def _cr_cleaned_path(object_dir: Path, side: str, source_path: Path) -> Path:
    return object_dir / side / f"{source_path.stem}_crclean.fits"


def _cr_mask_path(object_dir: Path, side: str, source_path: Path) -> Path:
    return object_dir / side / f"{source_path.stem}_crmask.fits"


def _format_cr_mask_summary(nvoxels: Optional[int], fraction: Optional[float]) -> str:
    if nvoxels is None and fraction is None:
        return "masking summary unavailable"
    if nvoxels is None:
        return f"masked {100.0 * float(fraction):.4f}% of finite cube"
    if fraction is None:
        return f"masked {int(nvoxels)} voxels"
    return f"masked {int(nvoxels)} voxels ({100.0 * float(fraction):.4f}% of finite cube)"


def _write_reused_cr_mask_if_needed(clean_path: Path, mask_path: Path) -> bool:
    if mask_path.exists():
        return True
    try:
        with fits.open(clean_path, memmap=False) as hdul:
            if "CR_MASK" not in hdul:
                return False
            mask = np.asarray(hdul["CR_MASK"].data, dtype=np.uint8)
            n_flagged = _finite_float(hdul[0].header.get("CRNPIX"))
            fraction_flagged = _finite_float(hdul[0].header.get("CRFRAC"))
        write_cr_mask_fits(
            clean_path,
            mask_path,
            mask,
            n_flagged=int(n_flagged) if n_flagged is not None else None,
            fraction_flagged=fraction_flagged,
        )
        return True
    except Exception as exc:
        print(f"WARNING: Could not write standalone CR mask from {clean_path}: {exc}")
        return False


def _cube_for_extraction(
    object_dir: Path,
    side: str,
    source_path: Path,
    *,
    cr_reject: bool,
    redo_cr_reject: bool,
    cr_config: CosmicRayRejectionConfig,
) -> Tuple[np.ndarray, fits.Header, Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    clean_path = _cr_cleaned_path(object_dir, side, source_path)
    mask_path = _cr_mask_path(object_dir, side, source_path)
    info: Dict[str, object] = {
        "enabled": bool(cr_reject),
        "status": "disabled",
        "cleaned_path": None,
        "mask_path": None,
        "nvoxels": None,
        "fraction": None,
        "diagnostics": {},
        "runtime_seconds": None,
        "config": config_to_dict(cr_config),
    }

    if not cr_reject:
        cube, hdr, uncert, flags = _load_cube_product(source_path)
        print(
            f"[CR] Stage: before white-light image, aperture review, and extraction. "
            f"Disabled; using original cube -> {source_path}"
        )
        return cube, hdr, uncert, flags, info

    print("[CR] Stage: before white-light image, aperture review, and extraction.")
    print(f"[CR] Config: {config_to_dict(cr_config)}")
    if clean_path.exists() and not redo_cr_reject:
        cube, hdr, uncert, flags = _load_cube_product(clean_path)
        method = str(hdr.get("CRMETH", "")).strip().upper()
        uncertainty_is_current = uncert is None or bool(hdr.get("CRUUPD", False))
        if method == "KCWI_DUAL" and uncertainty_is_current:
            cr_nvoxels = _finite_float(hdr.get("CRNPIX"))
            cr_fraction = _finite_float(hdr.get("CRFRAC"))
            cr_runtime = _finite_float(hdr.get("CRTIME"))
            cr_nvoxels_int = int(cr_nvoxels) if cr_nvoxels is not None else None
            mask_available = _write_reused_cr_mask_if_needed(clean_path, mask_path)
            info.update({
                "status": "reused_existing",
                "cleaned_path": str(clean_path),
                "mask_path": str(mask_path) if mask_available else None,
                "nvoxels": cr_nvoxels_int,
                "fraction": cr_fraction,
                "diagnostics": {},
                "runtime_seconds": cr_runtime,
            })
            print(f"[CR] Reusing existing CR-cleaned cube -> {clean_path}")
            print(f"[CR] Result: {_format_cr_mask_summary(cr_nvoxels_int, cr_fraction)}")
            if cr_runtime is not None:
                print(f"[CR] Recorded rejection runtime: {cr_runtime:.2f} s")
            if mask_available:
                print(f"[CR] Standalone mask cube -> {mask_path}")
            return cube, hdr, uncert, flags, info
        if method == "KCWI_DUAL" and not uncertainty_is_current:
            print(
                "[CR] Existing cleaned cube predates CR uncertainty propagation; "
                "rerunning from the original cube."
            )
        else:
            print(
                f"[CR] Existing cleaned cube uses {method or 'an unknown method'}; "
                "rerunning with the spatial+spectral track detector."
            )

    if clean_path.exists() and redo_cr_reject:
        print(f"[CR] --redo-cr-reject set; overwriting derived CR products from original cube.")

    cube, hdr, uncert, flags = _load_cube_product(source_path)

    print(f"[CR] Running rejection on original cube -> {source_path}")
    motion_axis = str(cr_config.slice_motion_axis).lower().strip()
    nslices = cube.shape[1] if motion_axis == "x" else cube.shape[2]
    resolved_workers = resolve_cr_workers(cr_config.workers, nslices)
    if resolved_workers > 1:
        mode = "automatic" if int(cr_config.workers) == 0 else "requested"
        print(
            f"[CR] Parallel track detection: {resolved_workers} workers "
            f"across {nslices} detector slices ({mode})."
        )
    else:
        print(f"[CR] Track detection: serial across {nslices} detector slices.")
    stage_started = perf_counter()
    result = reject_cosmic_rays(cube, uncert, flags, config=cr_config)
    print(f"[CR] Rejection runtime: {result.runtime_seconds:.2f} s")
    write_cr_cleaned_fits(source_path, clean_path, result)
    write_cr_mask_fits(
        source_path,
        mask_path,
        result.mask,
        n_flagged=result.n_flagged,
        fraction_flagged=result.fraction_flagged,
        config=result.config,
    )
    diag_path = object_dir / "diagnostics" / side / f"{source_path.stem}_cr_mask.png"
    plot_cr_diagnostic(
        result.mask,
        diag_path,
        title=f"{object_dir.name} {side} {source_path.stem}: CR mask",
    )
    info.update({
        "status": "created",
        "cleaned_path": str(clean_path),
        "mask_path": str(mask_path),
        "nvoxels": result.n_flagged,
        "fraction": result.fraction_flagged,
        "diagnostics": result.diagnostics,
        "runtime_seconds": result.runtime_seconds,
    })
    print(f"[CR] Diagnostics: {result.diagnostics}")
    print(
        f"[CR] Result: {_format_cr_mask_summary(result.n_flagged, result.fraction_flagged)}"
    )
    print(f"[CR] Saved cleaned cube -> {clean_path}")
    print(f"[CR] Saved standalone mask cube -> {mask_path}")
    if result.fraction_flagged > 0.005:
        print(
            "[CR] WARNING: CR rejection cleaned more than 0.5% of finite cube voxels; "
            "inspect the CR mask diagnostic before trusting this extraction."
        )
    print(f"[CR] Saved diagnostic -> {diag_path}")
    print(f"[CR] Total CR stage runtime including output writes: {perf_counter() - stage_started:.2f} s")
    return result.cleaned_cube, hdr, result.cleaned_uncert, flags, info


def _project_calib_dir(object_dir: Path, calib_dir: Optional[Path]) -> Path:
    if calib_dir is not None:
        return calib_dir.expanduser().resolve()
    root = find_project_root(object_dir)
    if root is not None:
        return root / "calibrations"
    return object_dir / "calibrations"


def _load_registry(calib_dir: Path) -> Dict[str, object]:
    path = calib_dir / "calibration_registry.json"
    if not path.exists():
        return {"standards": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(calib_dir: Path, registry: Dict[str, object]) -> None:
    calib_dir.mkdir(parents=True, exist_ok=True)
    with open(calib_dir / "calibration_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _load_extraction_state(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("top-level JSON value is not an object")
        return state
    except (OSError, ValueError) as exc:
        print(f"WARNING: Ignoring invalid extraction state {state_path}: {exc}")
        return {}


def _write_extraction_state(state_path: Path, state: Dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_name(f".{state_path.name}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(state_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _mean_airmass_for_side(
    object_dir: Path,
    side: str,
) -> Optional[float]:
    state_path = object_dir / "extraction_state.json"
    state = _load_extraction_state(state_path)

    exposures = state.get("sides", {}).get(side, {}).get("exposures", [])
    airmasses = [
        x for x in (_finite_float(item.get("airmass")) for item in exposures)
        if x is not None and x > 0
    ]
    if not airmasses:
        return None
    return float(np.mean(airmasses))


def _standard_airmass_from_calibration(cal: Dict[str, object], side: str) -> Optional[float]:
    x_std = _finite_float(cal.get("x_std"))
    if x_std is not None:
        return x_std
    counts_file = cal.get("counts_file")
    if not counts_file:
        return None
    try:
        object_dir = Path(str(counts_file)).expanduser().resolve().parent.parent
    except (OSError, RuntimeError):
        return None
    return _mean_airmass_for_side(
        object_dir,
        side,
    )


def _registry_product_type(item: Dict[str, object]) -> str:
    """Normalize registry labels from the old quicklook implementation."""
    value = str(item.get("product_type", "icubes"))
    if value in {"level2", "icubes"}:
        return "icubes"
    if value in {"level1", "level1_quicklook", "icubed"}:
        return "icubed"
    return value


def _calibration_is_compatible(item: Dict[str, object], product_type: str) -> bool:
    """Reject pre-normalization icubed sensitivities while preserving icubes entries."""
    if _registry_product_type(item) != product_type:
        return False
    if product_type != "icubed":
        return True
    return (
        item.get("exposure_normalized") is True
        and item.get("input_spectrum_units") == ICUBED_SPECTRUM_UNITS
        and item.get("calibration_schema_version") == CALIBRATION_SCHEMA_VERSION
    )


def _choose_calibration(
    calib_dir: Path,
    side: str,
    product_type: str = "icubes",
) -> Optional[Dict[str, object]]:
    registry = _load_registry(calib_dir)
    product_matches = [
        item for item in registry.get("standards", [])
        if item.get("side") == side
        and _registry_product_type(item) == product_type
    ]
    matches = [
        item for item in product_matches
        if _calibration_is_compatible(item, product_type)
    ]
    if not matches:
        if product_type == "icubed" and product_matches:
            print(
                f"WARNING: Ignoring {len(product_matches)} legacy {side} icubed "
                "calibration(s) built before exposure-time normalization. "
                "Rerun the standard-star extraction to rebuild the sensitivity."
            )
        print(f"No {side} calibration for cube type {product_type}.")
        return None
    print(f"\nAvailable {side} calibrations:")
    for i, item in enumerate(matches):
        print(f"  {i}: {item.get('standard_name')}  {item.get('sensitivity_file')}")
    if len(matches) == 1:
        ans = prompt("Use this calibration? (y/n)", "y").lower()
        return matches[0] if ans.startswith("y") else None
    idx = int(prompt("Calibration index", "0"))
    return matches[idx]


def _choose_standard_star(default_name: str) -> Tuple[int, str]:
    standards = list_standard_stars()
    print("\nAvailable AB standard stars:")
    for idx, name in standards:
        print(f"  {idx:2d}: {name}")
    default_id = None
    clean_default = default_name.replace("_", " ").strip().lower()
    for idx, name in standards:
        if clean_default and clean_default == name.lower():
            default_id = idx
            break
    default_text = str(default_id) if default_id is not None else ""
    raw = prompt("Standard star number", default_text if default_text else None)
    star_id = int(raw)
    if star_id not in STANDARD_NAMES:
        raise ValueError(f"Unknown standard star number: {star_id}")
    return star_id, STANDARD_NAMES[star_id]


def _continuum_from_points(lam: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
    points = sorted(points, key=lambda item: item[0])
    xp = np.asarray([p[0] for p in points], dtype=float)
    yp = np.asarray([p[1] for p in points], dtype=float)
    if len(points) >= 4:
        spline = UnivariateSpline(xp, yp, s=0, k=min(3, len(points) - 1))
        return spline(lam)
    return np.interp(lam, xp, yp, left=np.nan, right=np.nan)


def _nearest_point(points: List[Tuple[float, float]], x: float, y: float) -> Optional[int]:
    if not points:
        return None
    arr = np.asarray(points, dtype=float)
    dx = (arr[:, 0] - x) / max(np.nanmax(arr[:, 0]) - np.nanmin(arr[:, 0]), 1.0)
    dy = (arr[:, 1] - y) / max(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]), 1.0)
    return int(np.argmin(dx * dx + dy * dy))


def _nearest_point_pixels(ax, points: List[Tuple[float, float]], event) -> Tuple[Optional[int], float]:
    if not points:
        return None, np.inf
    pts = ax.transData.transform(np.asarray(points, dtype=float))
    mouse = np.asarray([event.x, event.y], dtype=float)
    dist = np.hypot(pts[:, 0] - mouse[0], pts[:, 1] - mouse[1])
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def _window_mask(lam: np.ndarray, windows: Optional[List[Tuple[float, float]]]) -> np.ndarray:
    mask = np.zeros(np.asarray(lam).shape, dtype=bool)
    if not windows:
        return mask
    for lo, hi in windows:
        mask |= (lam >= lo) & (lam <= hi)
    return mask


def _continuum_from_points_excluding_windows(
    lam: np.ndarray,
    points: List[Tuple[float, float]],
    exclude_windows: Optional[List[Tuple[float, float]]],
) -> np.ndarray:
    if not exclude_windows:
        return _continuum_from_points(lam, points)
    fit_points = [
        (x, y) for x, y in points
        if not any(lo <= x <= hi for lo, hi in exclude_windows)
    ]
    if len(fit_points) < 2:
        fit_points = points
    return _continuum_from_points(lam, fit_points)


def interactive_continuum_spline(
    lam: np.ndarray,
    counts: np.ndarray,
    ref_flux: np.ndarray,
    *,
    title: str,
    show: bool,
    exclude_windows: Optional[List[Tuple[float, float]]] = None,
    initial_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """Pick continuum points on a standard spectrum and return continuum + sensitivity."""
    good = np.isfinite(lam) & np.isfinite(counts) & np.isfinite(ref_flux) & (ref_flux > 0)
    if np.count_nonzero(good) < 8:
        raise ValueError("Not enough finite standard/reference samples to build sensitivity")

    lam_g = lam[good]
    counts_g = counts[good]
    ref_g = ref_flux[good]
    excluded_g = _window_mask(lam_g, exclude_windows)
    continuum_seed = good & ~_window_mask(lam, exclude_windows)
    if np.count_nonzero(continuum_seed) < 8:
        continuum_seed = good

    n_init = min(12, max(6, lam_g.size // 250))
    qs = np.linspace(5, 95, n_init)
    x_seed = lam[continuum_seed]
    y_seed = counts[continuum_seed]
    x_init = np.nanpercentile(x_seed, qs)
    y_init = np.interp(x_init, x_seed, y_seed)
    default_points: List[Tuple[float, float]] = list(zip(x_init, y_init))
    if initial_points is not None and len(initial_points) >= 2:
        points = [(float(x), float(y)) for x, y in initial_points if np.isfinite(x) and np.isfinite(y)]
        if len(points) < 2:
            points = list(default_points)
    else:
        points = list(default_points)
    accepted = {"done": False}

    fig, (ax_obs, ax_sens) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    ax_ref = ax_obs.twinx()
    original_view = {"xlim": None, "obs_ylim": None, "sens_ylim": None}
    zoom = {"active": False, "start": None, "axis": None, "patch": None}

    def redraw() -> None:
        current_view = {
            "xlim": ax_obs.get_xlim() if original_view["xlim"] is not None else None,
            "obs_ylim": ax_obs.get_ylim() if original_view["obs_ylim"] is not None else None,
        }
        ax_obs.clear()
        ax_ref.clear()
        ax_sens.clear()
        cont = (
            _continuum_from_points_excluding_windows(lam_g, points, exclude_windows)
            if len(points) >= 2 else np.full_like(lam_g, np.nan)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            sens = ref_g / cont

        ax_obs.plot(lam_g, counts_g, lw=0.8, color="0.35", label="Extracted standard")
        for lo, hi in exclude_windows or []:
            ax_obs.axvspan(lo, hi, alpha=0.16, color="tab:orange")
        if len(points) >= 2:
            ax_obs.plot(lam_g, cont, lw=1.5, color="tab:red", label="Continuum spline")
        if points:
            xp = [p[0] for p in points]
            yp = [p[1] for p in points]
            ax_obs.scatter(xp, yp, s=35, color="tab:red", zorder=5)
        ax_obs.set_ylabel("Observed counts")
        ax_obs.set_title(
            title
            + "\nleft-click add point, drag marker to move, right-click delete, z=zoom box, o=original zoom, a=accept, r=reset, q=quit"
            + ("\norange telluric windows are excluded from the continuum fit" if exclude_windows else "")
        )
        ax_obs.legend(loc="best")
        ax_obs.grid(alpha=0.2)

        ax_ref.plot(lam_g, ref_g, lw=0.9, color="tab:blue", alpha=0.55, label="AB reference flux")
        ax_ref.set_ylabel(f"Reference flux ({FLUX_UNIT_LABEL})")
        ax_ref.yaxis.set_label_position("right")
        ax_ref.yaxis.tick_right()
        ax_ref.tick_params(axis="y", colors="tab:blue")
        ax_ref.yaxis.label.set_color("tab:blue")
        ax_obs.set_zorder(ax_ref.get_zorder() + 1)
        ax_obs.patch.set_visible(False)

        ax_sens.plot(lam_g, sens, lw=1.0, color="tab:green")
        for lo, hi in exclude_windows or []:
            ax_sens.axvspan(lo, hi, alpha=0.16, color="tab:orange")
        ax_sens.set_ylabel("Sensitivity")
        ax_sens.set_xlabel("Wavelength (A)")
        ax_sens.grid(alpha=0.2)
        sens_good = sens[np.isfinite(sens) & (sens > 0) & ~excluded_g]
        if sens_good.size >= 5:
            ylo, yhi = np.nanpercentile(sens_good, [2, 98])
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                pad = 0.08 * (yhi - ylo)
                ax_sens.set_ylim(ylo - pad, yhi + pad)
        if original_view["xlim"] is None:
            original_view["xlim"] = ax_obs.get_xlim()
            original_view["obs_ylim"] = ax_obs.get_ylim()
            original_view["sens_ylim"] = ax_sens.get_ylim()
        elif current_view["xlim"] is not None:
            ax_obs.set_xlim(current_view["xlim"])
            ax_obs.set_ylim(current_view["obs_ylim"])
            ax_ref.relim()
            ax_ref.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()

    drag = {"idx": None}

    def event_obs_xy(event) -> Optional[Tuple[float, float]]:
        if event.inaxes not in (ax_obs, ax_ref) or event.x is None or event.y is None:
            return None
        x, y = ax_obs.transData.inverted().transform((event.x, event.y))
        xlim = ax_obs.get_xlim()
        ylim = ax_obs.get_ylim()
        if not (min(xlim) <= x <= max(xlim) and min(ylim) <= y <= max(ylim)):
            return None
        return float(x), float(y)

    def event_data_xy(event) -> Optional[Tuple[float, float]]:
        if event.inaxes not in (ax_obs, ax_sens) or event.x is None or event.y is None:
            return None
        x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
        return float(x), float(y)

    def clear_zoom_patch() -> None:
        if zoom["patch"] is not None:
            try:
                zoom["patch"].remove()
            except Exception:
                pass
            zoom["patch"] = None

    def draw_zoom_patch(event) -> None:
        if zoom["start"] is None or zoom["axis"] is None:
            return
        xy = event_data_xy(event)
        if xy is None:
            return
        x0, y0 = zoom["start"]
        x1, y1 = xy
        clear_zoom_patch()
        rect = Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0),
            abs(y1 - y0),
            fill=False,
            edgecolor="tab:purple",
            linewidth=1.5,
            linestyle="--",
        )
        zoom["axis"].add_patch(rect)
        zoom["patch"] = rect
        fig.canvas.draw_idle()

    def on_press(event):
        if zoom["active"]:
            xy = event_data_xy(event)
            if xy is not None:
                zoom["start"] = xy
                zoom["axis"] = event.inaxes
            return

        xy = event_obs_xy(event)
        if xy is None:
            return
        x, y = xy
        if event.button == 1:
            idx, dist_px = _nearest_point_pixels(ax_obs, points, event)
            if idx is not None and dist_px <= 8.0:
                drag["idx"] = idx
                return
            points.append((x, y))
            redraw()
        elif event.button == 3:
            idx = _nearest_point(points, x, y)
            if idx is not None and len(points) > 2:
                points.pop(idx)
                redraw()

    def on_motion(event):
        if zoom["active"]:
            draw_zoom_patch(event)
            return
        xy = event_obs_xy(event)
        if drag["idx"] is None or xy is None:
            return
        points[drag["idx"]] = xy
        redraw()

    def on_release(event):
        if zoom["active"]:
            xy = event_data_xy(event)
            if zoom["start"] is not None and xy is not None and zoom["axis"] is not None:
                x0, y0 = zoom["start"]
                x1, y1 = xy
                if abs(x1 - x0) > 0 and abs(y1 - y0) > 0:
                    ax_obs.set_xlim(min(x0, x1), max(x0, x1))
                    if zoom["axis"] is ax_obs:
                        ax_obs.set_ylim(min(y0, y1), max(y0, y1))
                    elif zoom["axis"] is ax_sens:
                        ax_sens.set_ylim(min(y0, y1), max(y0, y1))
                    fig.canvas.draw_idle()
            clear_zoom_patch()
            zoom["active"] = False
            zoom["start"] = None
            zoom["axis"] = None
            return
        drag["idx"] = None

    def on_key(event):
        if event.key in ("a", "enter", "return"):
            accepted["done"] = True
            plt.close(fig)
        elif event.key == "q":
            plt.close(fig)
        elif event.key == "z":
            zoom["active"] = True
            zoom["start"] = None
            zoom["axis"] = None
            clear_zoom_patch()
            ax_obs.set_title("Zoom mode: drag a box on either panel. Press o for original zoom.")
            fig.canvas.draw_idle()
        elif event.key == "o":
            clear_zoom_patch()
            zoom["active"] = False
            if original_view["xlim"] is not None:
                ax_obs.set_xlim(original_view["xlim"])
                ax_obs.set_ylim(original_view["obs_ylim"])
                ax_sens.set_ylim(original_view["sens_ylim"])
                fig.canvas.draw_idle()
        elif event.key == "r":
            points.clear()
            points.extend(default_points)
            redraw()

    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    redraw()
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)

    if not accepted["done"]:
        raise RuntimeError("Continuum spline was not accepted")
    continuum = _continuum_from_points_excluding_windows(lam, points, exclude_windows)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = ref_flux / continuum
    return continuum, sensitivity, sorted(points, key=lambda item: item[0])


def _normalized_for_telluric_plot(lam: np.ndarray, flux: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = np.isfinite(lam) & np.isfinite(flux) & (lam >= lo) & (lam <= hi)
    if np.count_nonzero(m) < 3:
        return np.full_like(flux, np.nan, dtype=float)
    scale = np.nanpercentile(flux[m], 90)
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanmedian(flux[m])
    if not np.isfinite(scale) or scale == 0:
        return np.full_like(flux, np.nan, dtype=float)
    return flux / scale


def interactive_telluric_shift(
    objname: str,
    lam_flux: np.ndarray,
    flux_before: np.ndarray,
    t_std: np.ndarray,
    telluric_mask: np.ndarray,
    x_std: float,
    x_sci: float,
    initial_shift_A: float,
) -> float:
    """Review and manually refine the telluric wavelength shift using O2 A/B bands."""
    shift = float(initial_shift_A)
    accepted = {"done": False, "shift": shift}

    fig, axes = plt.subplots(
        2,
        len(TELLURIC_ALIGNMENT_WINDOWS),
        figsize=(6.0 * len(TELLURIC_ALIGNMENT_WINDOWS), 7.2),
        squeeze=False,
    )
    fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.25, hspace=0.35)
    fig.suptitle(
        f"{objname} RED telluric alignment\n"
        "O2 B and A bands only. Positive shift moves the template redward."
    )

    line_telluric = []
    line_after = []
    panels = []
    for col, (lo, hi) in enumerate(TELLURIC_ALIGNMENT_WINDOWS):
        pad = 12.0
        m = np.isfinite(lam_flux) & (lam_flux >= lo - pad) & (lam_flux <= hi + pad)
        before_norm = _normalized_for_telluric_plot(lam_flux, flux_before, lo, hi)

        ax = axes[0, col]
        ax.plot(lam_flux[m], before_norm[m], lw=0.9, color="tab:red", label="Science before, normalized")
        tell_line, = ax.plot(lam_flux[m], np.ones(np.count_nonzero(m)), lw=1.0, color="tab:green",
                             label="Shifted/scaled telluric T")
        ax.axvspan(lo, hi, alpha=0.12, color="tab:orange")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Normalized flux / T")
        ax.set_title(f"O2 {'B' if hi < 7000 else 'A'} band")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax2 = axes[1, col]
        ax2.plot(lam_flux[m], before_norm[m], lw=0.8, color="tab:red", alpha=0.7, label="Before")
        after_line, = ax2.plot(lam_flux[m], before_norm[m], lw=0.9, color="k", label="After")
        ax2.axvspan(lo, hi, alpha=0.12, color="tab:orange")
        ax2.set_xlim(lo - pad, hi + pad)
        ax2.set_xlabel("Wavelength (A)")
        ax2.set_ylabel("Normalized flux")
        ax2.set_title("Correction preview")
        ax2.grid(alpha=0.2)
        ax2.legend(fontsize=8)

        panels.append((m, lo, hi))
        line_telluric.append(tell_line)
        line_after.append(after_line)

    slider_ax = fig.add_axes([0.16, 0.075, 0.58, 0.035])
    accept_ax = fig.add_axes([0.79, 0.06, 0.13, 0.06])
    shift_slider = Slider(
        slider_ax,
        "Shift (A)",
        -TELLURIC_MAX_SHIFT_A,
        TELLURIC_MAX_SHIFT_A,
        valinit=shift,
        valstep=TELLURIC_SHIFT_STEP_A,
    )
    accept_button = Button(accept_ax, "Accept")

    def update(new_shift: float) -> None:
        accepted["shift"] = float(new_shift)
        t_shifted = shifted_transmission(lam_flux, t_std, accepted["shift"])
        t_scaled = scaled_o2_transmission(
            t_shifted,
            x_std,
            x_sci,
            telluric_mask,
            min_T=TELLURIC_MIN_T,
            airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
        )
        flux_after = apply_standard_telluric_correction(
            flux_before,
            t_shifted,
            x_std,
            x_sci,
            telluric_mask,
            min_T=TELLURIC_MIN_T,
            airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
        )
        for i, (m, lo, hi) in enumerate(panels):
            line_telluric[i].set_ydata(t_scaled[m])
            line_after[i].set_ydata(_normalized_for_telluric_plot(lam_flux, flux_after, lo, hi)[m])
        fig.suptitle(
            f"{objname} RED telluric alignment: shift={accepted['shift']:.3f} A\n"
            "O2 B and A bands only. Positive shift moves the template redward."
        )
        fig.canvas.draw_idle()

    def accept(_event) -> None:
        accepted["done"] = True
        plt.close(fig)

    shift_slider.on_changed(update)
    accept_button.on_clicked(accept)
    update(shift)
    plt.show()
    plt.close(fig)

    if not accepted["done"]:
        raise RuntimeError("Telluric shift was not accepted")
    return float(accepted["shift"])


def _extract_side(
    object_dir: Path,
    side: str,
    *,
    show_plots: bool,
    redo_apertures: bool,
    cr_reject: bool,
    redo_cr_reject: bool,
    cr_config: CosmicRayRejectionConfig,
    spectral_cr_review: bool,
    spectral_cr_resolving_power: Optional[float],
    spectral_cr_config: SpectralCRConfig,
    product_type: str,
    initial_aperture_template: Optional[_ApertureTemplate] = None,
    prefer_initial_aperture_template: bool = False,
    show_coadd_diagnostic: bool = False,
) -> Optional[_SideExtractionResult]:
    files = _side_files(object_dir, side, product_type)
    if not files:
        return None

    spectra_dir = object_dir / "extracted" / side
    ap_dir = object_dir / "apertures" / side
    diag_dir = object_dir / "diagnostics" / side
    coadd_dir = object_dir / "coadded_spectra"

    extracted: List[ExposureSpectrum] = []
    spectra_for_coadd: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    resolving_power_estimates: List[ResolvingPowerEstimate] = []
    current_aps: Optional[TargetBackgroundApertures] = None
    first_aperture_template: Optional[_ApertureTemplate] = None
    print(
        f"[{object_dir.name} {side}] Aperture propagation: first approved aperture is proposed "
        "for subsequent exposures; later exposure-specific saved apertures are overwritten after approval."
    )

    for i, path in enumerate(files):
        exposure_label = f"{object_dir.name} {side} exposure {i + 1}/{len(files)}"
        cube, hdr, uncert, flags, cr_info = _cube_for_extraction(
            object_dir,
            side,
            path,
            cr_reject=cr_reject,
            redo_cr_reject=redo_cr_reject,
            cr_config=cr_config,
        )
        resolving_power_estimate = resolving_power_from_header(
            hdr,
            side,
            override=spectral_cr_resolving_power,
        )
        if resolving_power_estimate is not None:
            resolving_power_estimates.append(resolving_power_estimate)
        lam = get_lambda_axis(hdr, cube.shape)
        lo, hi = _side_limits(side)
        white_light_controller = WhiteLightRangeController(
            cube,
            lam,
            minimum=lo,
            maximum=hi,
        )
        img = white_light_controller.image()

        ap_path = ap_dir / f"{path.stem}_aperture.json"
        use_initial_template = initial_aperture_template is not None and (
            prefer_initial_aperture_template
            or redo_apertures
            or not ap_path.exists()
        )
        if current_aps is not None:
            plot_apertures(
                img,
                current_aps,
                diag_dir / f"{path.stem}_aperture_reuse_preview.png",
                title=f"{exposure_label}: proposed current aperture",
                show=False,
                wavelength_controller=white_light_controller,
            )
            aps = review_apertures(
                img,
                current_aps,
                side_label=f"{exposure_label}: proposed current aperture",
                show=show_plots,
                wavelength_controller=white_light_controller,
            )
        elif use_initial_template:
            assert initial_aperture_template is not None
            try:
                proposed_aps = _transform_apertures_between_headers(
                    initial_aperture_template.apertures,
                    initial_aperture_template.header,
                    hdr,
                    (cube.shape[1], cube.shape[2]),
                )
            except ValueError as exc:
                print(
                    f"[{object_dir.name} {side}] WARNING: Could not transform approved "
                    f"{initial_aperture_template.side} aperture into this cube: {exc}. "
                    "Define apertures normally."
                )
                aps = interactive_define_apertures(
                    img,
                    exposure_label,
                    show=show_plots,
                    wavelength_controller=white_light_controller,
                )
            else:
                source_name = initial_aperture_template.exposure_path.name
                print(
                    f"[{object_dir.name} {side}] Initial aperture proposal transformed "
                    f"from approved {initial_aperture_template.side} aperture "
                    f"({source_name})."
                )
                plot_apertures(
                    img,
                    proposed_aps,
                    diag_dir / f"{path.stem}_cross_side_aperture_proposal.png",
                    title=(
                        f"{exposure_label}: proposal from "
                        f"{initial_aperture_template.side} aperture"
                    ),
                    show=False,
                    wavelength_controller=white_light_controller,
                )
                aps = review_apertures(
                    img,
                    proposed_aps,
                    side_label=(
                        f"{exposure_label}: transformed "
                        f"{initial_aperture_template.side} aperture"
                    ),
                    show=show_plots,
                    wavelength_controller=white_light_controller,
                )
        elif ap_path.exists() and not redo_apertures:
            aps = _aperture_from_json(ap_path)
            aps = review_apertures(
                img,
                aps,
                side_label=f"{exposure_label}: saved first-exposure aperture",
                show=show_plots,
                wavelength_controller=white_light_controller,
            )
        else:
            aps = interactive_define_apertures(
                img,
                exposure_label,
                show=show_plots,
                wavelength_controller=white_light_controller,
            )

        current_aps = aps
        if first_aperture_template is None:
            first_aperture_template = _ApertureTemplate(
                apertures=aps,
                header=hdr.copy(),
                side=side,
                exposure_path=path,
            )
        _aperture_to_json(ap_path, aps)
        plot_apertures(
            img,
            aps,
            diag_dir / f"{path.stem}_aperture.png",
            title=f"{exposure_label}: aperture",
            show=False,
            wavelength_controller=white_light_controller,
        )

        counts, sigma = _extract_counts_with_uncert(cube, uncert, flags, aps, label=exposure_label)
        counts, sigma, exposure_time, exposure_time_keyword, spectrum_units = (
            _normalize_extracted_spectrum_for_product(
                product_type,
                hdr,
                counts,
                sigma,
                label=exposure_label,
            )
        )
        if exposure_time is not None:
            print(
                f"[{exposure_label}] Normalized icubed spectrum and uncertainty "
                f"by {exposure_time:g} s from {exposure_time_keyword}."
            )
        lam, counts, sigma = _trim_side_arrays(side, lam, counts, sigma)
        spec_path = spectra_dir / f"{path.stem}_counts.flm"
        spectrum_header = "count_rate_e_per_s" if product_type == "icubed" else "native_icubes_flux"
        _save_spectrum(spec_path, lam, counts, sigma, spectrum_header)
        spectra_for_coadd.append((lam, counts, sigma))
        extracted.append(
            ExposureSpectrum(
                path=str(path),
                side=side,
                lam_path=str(spec_path),
                spectrum_path=str(spec_path),
                aperture_path=str(ap_path),
                airmass=get_airmass_from_header(hdr),
                exposure_time_seconds=exposure_time,
                exposure_time_keyword=exposure_time_keyword,
                spectrum_units=spectrum_units,
                cr_status=str(cr_info.get("status")),
                cr_cleaned_path=cr_info.get("cleaned_path") if cr_info.get("cleaned_path") is not None else None,
                cr_mask_path=cr_info.get("mask_path") if cr_info.get("mask_path") is not None else None,
                cr_nvoxels=int(cr_info["nvoxels"]) if cr_info.get("nvoxels") is not None else None,
                cr_fraction=float(cr_info["fraction"]) if cr_info.get("fraction") is not None else None,
                cr_diagnostics=dict(cr_info.get("diagnostics", {})),
                cr_runtime_seconds=(
                    float(cr_info["runtime_seconds"])
                    if cr_info.get("runtime_seconds") is not None
                    else None
                ),
            )
        )
        print(f"Extracted {exposure_label} -> {spec_path}")

    coadd_diag_path = diag_dir / f"{object_dir.name}_{side}_coadd_diagnostic.png"
    lam_c, counts_c, sigma_c, n_good, coadd_clip_sigma = _review_1d_coadd_sigma_clip(
        coadd_diag_path,
        f"{object_dir.name} {side}: extracted spectra and sigma-clipped coadd",
        spectra_for_coadd,
        initial_sigma=COADD_CLIP_SIGMA,
        maxiters=COADD_CLIP_MAXITERS,
        show=show_coadd_diagnostic,
    )
    spectral_cr_state: Dict[str, object] = {
        "enabled": bool(spectral_cr_review),
        "config": asdict(spectral_cr_config),
        "resolving_power": None,
        "candidate_count": 0,
        "removed_count": 0,
    }
    if spectral_cr_review:
        if not resolving_power_estimates:
            print(
                f"[1D CR {side}] WARNING: Could not derive resolving power from "
                "BGRATNAM/RGRATNAM and IFUNAM; skipping narrow-line review. "
                "Use --spectral-cr-resolving-power to supply it explicitly."
            )
            spectral_cr_state["status"] = "skipped_no_resolving_power"
        else:
            estimate_values = np.array(
                [item.value for item in resolving_power_estimates],
                dtype=float,
            )
            resolving_power = float(np.median(estimate_values))
            representative = min(
                resolving_power_estimates,
                key=lambda item: abs(item.value - resolving_power),
            )
            if not np.allclose(estimate_values, resolving_power, rtol=0.01, atol=0.0):
                print(
                    f"[1D CR {side}] WARNING: Exposure resolving-power estimates differ "
                    f"({estimate_values.tolist()}); using median R={resolving_power:g}."
                )
            qualifier = "approximately " if representative.approximate else ""
            print(
                f"[1D CR {side}] Resolving power: {qualifier}R={resolving_power:g} "
                f"from grating={representative.grating}, slicer={representative.slicer} "
                f"({representative.source})."
            )
            detection = detect_cr_like_narrow_features(
                lam_c,
                counts_c,
                sigma_c,
                resolving_power=resolving_power,
                config=spectral_cr_config,
            )
            candidate_count = len(detection.candidates)
            print(
                f"[1D CR {side}] Found {candidate_count} candidate"
                f"{'s' if candidate_count != 1 else ''} with absolute S/N above "
                f"{spectral_cr_config.detection_sigma:g} sigma with FWHM below "
                f"{spectral_cr_config.max_lsf_fraction:g} of the expected LSF."
            )
            review_path = coadd_dir / f"{object_dir.name}_{side}_spectral_cr_review.json"
            pre_review_path: Optional[Path] = None
            summary_path: Optional[Path] = None
            decisions: List[Dict[str, object]] = []
            original_counts = counts_c.copy()
            if candidate_count:
                pre_review_path = (
                    coadd_dir
                    / f"{object_dir.name}_{side}_counts_coadd_before_spectral_cr.flm"
                )
                _save_spectrum(
                    pre_review_path,
                    lam_c,
                    counts_c,
                    sigma_c,
                    "counts_coadd_before_spectral_cr",
                )
                counts_c, sigma_c, decisions = _review_spectral_cr_candidates(
                    f"{object_dir.name} {side} resolving-power CR review",
                    lam_c,
                    counts_c,
                    sigma_c,
                    detection,
                )
                summary_path = (
                    diag_dir / f"{object_dir.name}_{side}_spectral_cr_review.png"
                )
                _plot_spectral_cr_review_summary(
                    summary_path,
                    f"{object_dir.name} {side}: resolving-power CR review",
                    lam_c,
                    original_counts,
                    counts_c,
                    sigma_c,
                    decisions,
                )
            removed_count = sum(
                decision.get("decision") == "removed" for decision in decisions
            )
            review_report = {
                "status": "completed",
                "side": side,
                "resolving_power": resolving_power,
                "resolving_power_estimates": [
                    asdict(item) for item in resolving_power_estimates
                ],
                "config": asdict(spectral_cr_config),
                "candidate_count": candidate_count,
                "removed_count": int(removed_count),
                "pre_review_spectrum": (
                    str(pre_review_path) if pre_review_path is not None else None
                ),
                "summary_diagnostic": (
                    str(summary_path) if summary_path is not None else None
                ),
                "decisions": decisions,
            }
            _write_extraction_state(review_path, review_report)
            spectral_cr_state.update(review_report)
            spectral_cr_state["review_file"] = str(review_path)
            print(
                f"[1D CR {side}] Removed {removed_count}/{candidate_count} reviewed "
                f"candidates; decisions -> {review_path}"
            )
            if summary_path is not None:
                print(f"[1D CR {side}] Saved review diagnostic -> {summary_path}")

    out_path = coadd_dir / f"{object_dir.name}_{side}_counts_coadd.flm"
    coadd_header = "count_rate_e_per_s_coadd" if product_type == "icubed" else "native_icubes_flux_coadd"
    _save_spectrum(out_path, lam_c, counts_c, sigma_c, coadd_header)
    np.savetxt(coadd_dir / f"{object_dir.name}_{side}_nexp.txt", np.c_[lam_c, n_good], header="lambda_A  n_exposures_used")

    state_path = object_dir / "extraction_state.json"
    state = _load_extraction_state(state_path)
    state.setdefault("sides", {})[side] = {
        "coadd_counts": str(out_path),
        "coadd_clip_sigma": coadd_clip_sigma,
        "coadd_clip_maxiters": COADD_CLIP_MAXITERS,
        "cr_reject": bool(cr_reject),
        "redo_cr_reject": bool(redo_cr_reject),
        "cr_config": config_to_dict(cr_config),
        "product_type": product_type,
        "input_spectrum_units": (
            ICUBED_SPECTRUM_UNITS if product_type == "icubed" else ICUBES_SPECTRUM_UNITS
        ),
        "exposure_normalized": product_type == "icubed",
        "spectral_cr_review": spectral_cr_state,
        "exposures": [asdict(item) for item in extracted],
    }
    _write_extraction_state(state_path, state)

    print(f"Coadded {side} {product_type} 1D spectra with sigma={coadd_clip_sigma:g} -> {out_path}")
    print(f"Saved {side} coadd diagnostic -> {coadd_diag_path}")
    if first_aperture_template is None:
        raise RuntimeError(f"No approved {side} aperture was available after extraction")
    return _SideExtractionResult(
        coadd_path=out_path,
        aperture_template=first_aperture_template,
    )


def _load_txt_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    lam = arr[:, 0]
    y = arr[:, 1]
    sigma = arr[:, 2] if arr.shape[1] >= 3 else None
    return lam, y, sigma


def _existing_fluxcal_path(
    object_dir: Path,
    side: str,
) -> Path:
    return object_dir / "fluxcal" / f"{object_dir.name}_{side}_fluxcal.flm"


def _find_existing_fluxcal_path(
    object_dir: Path,
    side: str,
) -> Optional[Path]:
    path = _existing_fluxcal_path(object_dir, side)
    if not path.exists():
        legacy_path = object_dir / "fluxcal" / f"{object_dir.name}_{side}_fluxcal.txt"
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        return None
    return path


def _load_existing_fluxcal_side(
    object_dir: Path,
    side: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    path = _find_existing_fluxcal_path(object_dir, side)
    if path is None:
        return None
    lam, flux, sigma = _load_txt_spectrum(path)
    if path.suffix == ".txt":
        print(f"Converting legacy {side} fluxcal file from 1e-16 to 1e-15 units for join reuse: {path}")
        flux = flux * 0.1
        sigma = sigma * 0.1 if sigma is not None else None
    lam, flux, sigma = _trim_side_arrays(side, lam, flux, sigma)
    return lam, flux, sigma


def _calibration_flux_unit_scale(cal: Dict[str, object]) -> float:
    units = str(cal.get("reference_units", "")).replace(" ", "").lower()
    if "1e-16" in units:
        return 0.1
    return 1.0


def _join_science_flux_sides(
    object_dir: Path,
    flux_paths: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    *,
    show_plots: bool,
) -> bool:
    final_dir = object_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if "BLUE" not in flux_paths or "RED" not in flux_paths:
        return False

    lam_b, flux_b, sig_b = flux_paths["BLUE"]
    lam_r, flux_r, sig_r = flux_paths["RED"]
    blue_scale, red_scale = interactive_rescale_and_approve_flux(
        objname=object_dir.name,
        lam_blue=lam_b,
        flux_blue=flux_b,
        lam_red=lam_r,
        flux_red=flux_r,
        outdir=final_dir,
        show=True,
        interactive=True,
    )
    flux_b_scaled = flux_b * blue_scale
    flux_r_scaled = flux_r * red_scale
    sig_b_scaled = sig_b * abs(blue_scale) if sig_b is not None else None
    sig_r_scaled = sig_r * abs(red_scale) if sig_r is not None else None

    lam_j, flux_j = concat_join(lam_b, flux_b_scaled, lam_r, flux_r_scaled)
    if sig_b is not None and sig_r is not None:
        lam_s, sig_j = concat_join(lam_b, sig_b_scaled, lam_r, sig_r_scaled)
        order = np.argsort(lam_s)
        sig_j = sig_j[order] if not np.allclose(lam_s, lam_j) else sig_j
    else:
        sig_j = None

    out_path = final_dir / f"{object_dir.name}_BLUE+RED_spectrum.flm"
    _save_spectrum(out_path, lam_j, flux_j, sig_j, "flux")
    _plot_joined_spectrum_png(
        final_dir / f"{object_dir.name}_BLUE+RED_spectrum.png",
        f"{object_dir.name}: final BLUE+RED spectrum",
        lam_b,
        flux_b_scaled,
        sig_b_scaled,
        lam_r,
        flux_r_scaled,
        sig_r_scaled,
        show=True,
    )
    plot_join_diagnostic(
        object_dir.name,
        lam_b,
        flux_b_scaled,
        lam_r,
        flux_r_scaled,
        outpng=final_dir / f"{object_dir.name}_joined.png",
        show=show_plots,
    )
    print(f"Saved joined spectrum -> {out_path}")
    return True


def join_existing_science_sides(
    object_dir: Path,
    *,
    show_plots: bool = False,
) -> None:
    object_dir = object_dir.expanduser().resolve()
    flux_paths: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    for side in ("BLUE", "RED"):
        existing = _load_existing_fluxcal_side(object_dir, side)
        if existing is None:
            print(
                f"No existing {side} flux-calibrated spectrum found at "
                f"{_existing_fluxcal_path(object_dir, side)}"
            )
        else:
            flux_paths[side] = existing
            print(
                f"Loaded existing flux-calibrated {side} spectrum -> "
                f"{_find_existing_fluxcal_path(object_dir, side)}"
            )

    if not _join_science_flux_sides(
        object_dir,
        flux_paths,
        show_plots=show_plots,
    ):
        raise FileNotFoundError(
            f"Join-only requires both BLUE and RED fluxcal spectra under {object_dir / 'fluxcal'}"
        )


def _load_spline_points(paths: List[Path]) -> Optional[List[Tuple[float, float]]]:
    for path in paths:
        if not path.exists():
            continue
        try:
            arr = np.loadtxt(path, comments="#")
        except Exception as exc:
            print(f"WARNING: could not load saved spline points from {path}: {exc}")
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 2 or arr.shape[0] < 2:
            print(f"WARNING: saved spline point file has too few points: {path}")
            continue
        points = [(float(row[0]), float(row[1])) for row in arr if np.isfinite(row[0]) and np.isfinite(row[1])]
        if len(points) >= 2:
            print(f"Loaded {len(points)} saved continuum spline points from {path}")
            return points
    return None


def _save_spline_points(paths: List[Path], points: List[Tuple[float, float]]) -> None:
    arr = np.asarray(sorted(points, key=lambda item: item[0]), dtype=float)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, arr, header="lambda_A  observed_continuum_counts_point")


def _build_standard_calibrations(
    object_dir: Path,
    coadd_paths: Dict[str, Path],
    calib_dir: Path,
    *,
    show_plots: bool,
    product_type: str,
) -> None:
    standard_name = object_dir.name
    registry = _load_registry(calib_dir)
    registry.setdefault("standards", [])

    for side, counts_path in coadd_paths.items():
        lam_std, counts, sigma_counts = _load_txt_spectrum(counts_path)
        lam_std, counts, sigma_counts = _trim_side_arrays(side, lam_std, counts, sigma_counts)
        star_id, star_name = _choose_standard_star(standard_name)
        flux_ref = reference_flux(star_id, lam_std, scale_1e15=True)

        outdir = calib_dir / standard_name / side
        object_diag_dir = object_dir / "diagnostics" / side / "standard_calibration"
        object_flux_dir = object_dir / "fluxcal"
        object_final_dir = object_dir / "final"
        outdir.mkdir(parents=True, exist_ok=True)
        object_diag_dir.mkdir(parents=True, exist_ok=True)
        object_flux_dir.mkdir(parents=True, exist_ok=True)
        object_final_dir.mkdir(parents=True, exist_ok=True)
        # Use a distinct filename so legacy control points fitted to integrated
        # icubed electrons cannot be reused against the new electron/s spectra.
        spline_suffix = "_electron_per_s" if product_type == "icubed" else ""
        spline_name = f"continuum_spline_points_{side}{spline_suffix}.txt"
        spline_points_path = outdir / spline_name
        object_spline_points_path = object_diag_dir / spline_name
        initial_points = _load_spline_points([object_spline_points_path, spline_points_path])

        continuum, sens, spline_points = interactive_continuum_spline(
            lam_std,
            counts,
            flux_ref,
            title=f"{standard_name} {side}: continuum fit for {star_name}",
            show=show_plots,
            exclude_windows=O2_WINDOWS if side == "RED" else None,
            initial_points=initial_points,
        )
        _save_spline_points([spline_points_path, object_spline_points_path], spline_points)
        lam_cal_std, flux_cal_std, sigma_flux_std = apply_sensitivity_with_uncertainty(
            lam_std,
            sens,
            lam_std,
            counts,
            sigma_counts,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux_ref / continuum

        sens_path = outdir / f"sensitivity_{side}.txt"
        continuum_path = outdir / f"observed_continuum_{side}.txt"
        ref_path = outdir / f"ab_reference_flux_{side}.txt"
        np.savetxt(sens_path, np.c_[lam_std, sens], header="lambda_A  S_lambda")
        np.savetxt(continuum_path, np.c_[lam_std, continuum], header="lambda_A  observed_continuum_counts")
        np.savetxt(ref_path, np.c_[lam_std, flux_ref], header="lambda_A  reference_flux_1e-15_erg_s_cm2_A")

        item: Dict[str, object] = {
            "standard_name": standard_name,
            "ab_standard_id": star_id,
            "ab_standard_name": star_name,
            "side": side,
            "wavelength_range_A": list(_side_limits(side)),
            "counts_file": str(counts_path),
            "reference_flux_file": str(ref_path),
            "observed_continuum_file": str(continuum_path),
            "continuum_spline_points_file": str(spline_points_path),
            "sensitivity_file": str(sens_path),
            "reference_units": FLUX_UNIT_LABEL,
            "product_type": product_type,
            "input_spectrum_units": (
                ICUBED_SPECTRUM_UNITS if product_type == "icubed" else ICUBES_SPECTRUM_UNITS
            ),
            "exposure_normalized": product_type == "icubed",
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        }

        tell_path = None
        mask_path = None
        tell_before = None
        tell_after = None
        final_standard_flux = flux_cal_std
        final_standard_sigma = sigma_flux_std
        if side == "RED":
            t_o2, o2_mask = build_standard_telluric_template(
                lam_std=lam_std,
                C_std=counts,
                continuum_std=continuum,
                telluric_windows=O2_WINDOWS,
                min_T=TELLURIC_MIN_T,
                smooth_s=TELLURIC_TEMPLATE_SMOOTH_S,
            )
            tell_path = outdir / "telluric_standard_template_RED.txt"
            np.savetxt(tell_path, np.c_[lam_std, t_o2, o2_mask.astype(int)],
                       header="lambda_A  T_telluric_std  in_telluric_mask")
            mask_path = str(tell_path)
            tell_before = flux_cal_std
            tell_after = apply_standard_telluric_correction(
                flux_cal_std,
                t_o2,
                1.0,
                1.0,
                o2_mask,
                min_T=TELLURIC_MIN_T,
                airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
            )
            if sigma_flux_std is not None:
                final_standard_sigma = apply_standard_telluric_correction(
                    sigma_flux_std,
                    t_o2,
                    1.0,
                    1.0,
                    o2_mask,
                    min_T=TELLURIC_MIN_T,
                    airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
                )
            final_standard_flux = tell_after
            t_scaled_std = scaled_o2_transmission(
                t_o2,
                1.0,
                1.0,
                o2_mask,
                min_T=TELLURIC_MIN_T,
                airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
            )
            std_tell_path = outdir / "standard_fluxcal_RED_tellcorr.flm"
            np.savetxt(
                std_tell_path,
                np.c_[lam_std, tell_before, tell_after],
                header="lambda_A  standard_flux_before_telluric  standard_flux_after_telluric",
            )
            plot_o2_template_diagnostic(lam_std, t_o2, O2_WINDOWS,
                                        outdir / "telluric_template_RED.png", show=True)
            plot_o2_template_diagnostic(lam_std, t_o2, O2_WINDOWS,
                                        object_diag_dir / "telluric_template_RED.png", show=False)
            item["telluric_file"] = str(tell_path)
            item["telluric_source"] = "same_spectrophotometric_standard"
            item["telluric_airmass_exponent"] = TELLURIC_AIRMASS_EXPONENT
            item["x_std"] = _mean_airmass_for_side(object_dir, side)
            if item["x_std"] is None:
                print("WARNING: RED standard airmass not found; science telluric correction will be skipped unless registry x_std is set.")
            else:
                print(f"RED standard mean airmass: X_std={item['x_std']:.4f}")

        plot_calibration_diagnostics(
            side=side,
            std_name=standard_name,
            lam_std=lam_std,
            C_std=counts,
            lam_ref=lam_std,
            F_ref=flux_ref,
            ratio=ratio,
            S=sens,
            F_std_cal=flux_cal_std,
            outdir=outdir / "diagnostics",
            show=show_plots,
            telluric_windows=O2_WINDOWS if side == "RED" else None,
            red_tell_before=tell_before,
            red_tell_after=tell_after,
        )
        plot_calibration_diagnostics(
            side=side,
            std_name=standard_name,
            lam_std=lam_std,
            C_std=counts,
            lam_ref=lam_std,
            F_ref=flux_ref,
            ratio=ratio,
            S=sens,
            F_std_cal=final_standard_flux,
            outdir=object_diag_dir,
            show=False,
            telluric_windows=O2_WINDOWS if side == "RED" else None,
            red_tell_before=tell_before,
            red_tell_after=tell_after,
        )
        if side == "RED" and tell_before is not None and tell_after is not None:
            plot_o2_correction_diagnostic(
                standard_name,
                lam_std,
                tell_before,
                tell_after,
                t_o2,
                t_scaled_std,
                o2_mask,
                O2_WINDOWS,
                outdir / "diagnostics" / f"{safe_filename(standard_name)}_RED_telluric_detail.png",
                show=True,
            )
            plot_o2_correction_diagnostic(
                standard_name,
                lam_std,
                tell_before,
                tell_after,
                t_o2,
                t_scaled_std,
                o2_mask,
                O2_WINDOWS,
                object_diag_dir / f"{safe_filename(standard_name)}_RED_telluric_detail.png",
                show=False,
            )

        final_std_path = object_final_dir / f"{standard_name}_{side}_standard_processed.flm"
        _save_spectrum(final_std_path, lam_std, final_standard_flux, final_standard_sigma, "flux")
        _plot_spectrum_png(
            object_final_dir / f"{standard_name}_{side}_standard_processed.png",
            f"{standard_name} {side}: processed standard spectrum",
            lam_std,
            final_standard_flux,
            final_standard_sigma,
            show=True,
        )

        registry["standards"] = [
            old for old in registry["standards"]
            if not (
                old.get("standard_name") == standard_name
                and old.get("side") == side
                and _registry_product_type(old) == product_type
            )
        ]
        registry["standards"].append(item)
        print(f"Saved {side} calibration -> {sens_path}")
        if mask_path:
            print(f"Saved RED telluric template -> {tell_path}")

    _save_registry(calib_dir, registry)
    print(f"Updated calibration registry -> {calib_dir / 'calibration_registry.json'}")


def _apply_science_calibrations(
    object_dir: Path,
    coadd_paths: Dict[str, Path],
    calib_dir: Path,
    *,
    show_plots: bool,
    product_type: str,
) -> None:
    flux_paths: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    flux_dir = object_dir / "fluxcal"
    final_dir = object_dir / "final"
    flux_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    for side, counts_path in coadd_paths.items():
        cal = _choose_calibration(calib_dir, side, product_type)
        if cal is None:
            print(f"No {side} calibration selected; leaving counts-only product.")
            continue

        lam_counts, counts, sigma_counts = _load_txt_spectrum(counts_path)
        lam_counts, counts, sigma_counts = _trim_side_arrays(side, lam_counts, counts, sigma_counts)
        sens_arr = np.loadtxt(cal["sensitivity_file"], comments="#")
        lam_sens = sens_arr[:, 0]
        sens = sens_arr[:, 1]
        lam_sens, sens = _trim_side_arrays(side, lam_sens, sens)
        unit_scale = _calibration_flux_unit_scale(cal)
        if unit_scale != 1.0:
            print(
                f"Converting selected {side} calibration from {cal.get('reference_units')} "
                f"to {FLUX_UNIT_LABEL}."
            )
            sens = sens * unit_scale
        lam_flux, flux, sigma_flux = apply_sensitivity_with_uncertainty(
            lam_sens,
            sens,
            lam_counts,
            counts,
            sigma_counts,
        )

        if side == "RED" and cal.get("telluric_file"):
            tell = np.loadtxt(cal["telluric_file"], comments="#")
            lam_tell = tell[:, 0]
            t_std = np.interp(lam_flux, lam_tell, tell[:, 1], left=1.0, right=1.0)
            o2_mask = np.interp(lam_flux, lam_tell, tell[:, 2].astype(float), left=0.0, right=0.0) > 0.5
            x_std = _standard_airmass_from_calibration(cal, side)
            x_sci = _mean_airmass_for_side(object_dir, side)
            if x_std is None or x_sci is None:
                print(
                    f"WARNING: Skipping RED telluric correction for {object_dir.name}; "
                    f"standard/science airmass unavailable (X_std={x_std}, X_sci={x_sci})."
                )
            else:
                flux_before_telluric = flux.copy()
                tell_shift_A = estimate_telluric_shift(
                    lam_flux,
                    flux,
                    t_std,
                    TELLURIC_ALIGNMENT_WINDOWS,
                    max_shift_A=TELLURIC_MAX_SHIFT_A,
                    step_A=TELLURIC_SHIFT_STEP_A,
                )
                print(f"Initial RED telluric shift estimate from O2 A/B bands: {tell_shift_A:.3f} A")
                tell_shift_A = interactive_telluric_shift(
                    object_dir.name,
                    lam_flux,
                    flux_before_telluric,
                    t_std,
                    o2_mask,
                    x_std,
                    x_sci,
                    tell_shift_A,
                )
                t_shifted = shifted_transmission(lam_flux, t_std, tell_shift_A)
                t_scaled = scaled_o2_transmission(
                    t_shifted,
                    x_std,
                    x_sci,
                    o2_mask,
                    min_T=TELLURIC_MIN_T,
                    airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
                )
                flux = apply_standard_telluric_correction(
                    flux,
                    t_shifted,
                    x_std,
                    x_sci,
                    o2_mask,
                    min_T=TELLURIC_MIN_T,
                    airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
                )
                if sigma_flux is not None:
                    sigma_flux = apply_standard_telluric_correction(
                        sigma_flux,
                        t_shifted,
                        x_std,
                        x_sci,
                        o2_mask,
                        min_T=TELLURIC_MIN_T,
                        airmass_exponent=TELLURIC_AIRMASS_EXPONENT,
                    )
                plot_o2_before_after(
                    object_dir.name,
                    lam_flux,
                    flux_before_telluric,
                    flux,
                    O2_WINDOWS,
                    flux_dir / f"{object_dir.name}_RED_telluric_correction.png",
                    show=False,
                )
                plot_o2_correction_diagnostic(
                    object_dir.name,
                    lam_flux,
                    flux_before_telluric,
                    flux,
                    t_shifted,
                    t_scaled,
                    o2_mask,
                    O2_WINDOWS,
                    flux_dir / f"{object_dir.name}_RED_telluric_detail.png",
                    show=True,
                )
                np.savetxt(
                    flux_dir / f"{object_dir.name}_RED_fluxcal_before_telluric.flm",
                    np.c_[lam_flux, flux_before_telluric],
                    header="lambda_A  flux_before_telluric",
                )
                np.savetxt(
                    flux_dir / f"{object_dir.name}_RED_telluric_correction_arrays.txt",
                    np.c_[lam_flux, flux_before_telluric, t_std, t_shifted, t_scaled, flux],
                    header=(
                        "lambda_A  flux_before_telluric  T_std_unshifted  "
                        f"T_std_shifted_shift_A_{tell_shift_A:.3f}  "
                        "T_airmass_scaled  flux_after_telluric"
                    ),
                )
                print(
                    f"Applied RED telluric correction with X_std={x_std:.4f}, "
                    f"X_sci={x_sci:.4f}, shift={tell_shift_A:.3f} A."
                )

        out_path = flux_dir / f"{object_dir.name}_{side}_fluxcal.flm"
        _save_spectrum(out_path, lam_flux, flux, sigma_flux, "flux")
        _plot_spectrum_png(
            flux_dir / f"{object_dir.name}_{side}_fluxcal.png",
            f"{object_dir.name} {side}: flux-calibrated spectrum",
            lam_flux,
            flux,
            sigma_flux,
            show=False,
        )
        flux_paths[side] = (lam_flux, flux, sigma_flux)
        print(f"Saved flux-calibrated {side} spectrum -> {out_path}")

    for side in ("BLUE", "RED"):
        if side in flux_paths:
            continue
        existing = _load_existing_fluxcal_side(object_dir, side)
        if existing is not None:
            flux_paths[side] = existing
            print(
                f"Reusing existing flux-calibrated {side} spectrum for join -> "
                f"{_find_existing_fluxcal_path(object_dir, side)}"
            )

    if _join_science_flux_sides(
        object_dir,
        flux_paths,
        show_plots=show_plots,
    ):
        return

    if len(flux_paths) == 1:
        side, (lam, flux, sigma) = next(iter(flux_paths.items()))
        out_path = final_dir / f"{object_dir.name}_{side}_spectrum.flm"
        _save_spectrum(out_path, lam, flux, sigma, "flux")
        _plot_spectrum_png(
            final_dir / f"{object_dir.name}_{side}_spectrum.png",
            f"{object_dir.name}: final {side} spectrum",
            lam,
            flux,
            sigma,
            show=True,
        )
        print(f"Saved final {side} spectrum -> {out_path}")


def extract_object(
    object_dir: Path,
    *,
    calib_dir: Optional[Path] = None,
    standard: Optional[bool] = None,
    side: str = "both",
    show_plots: bool = False,
    redo_apertures: bool = False,
    cr_reject: bool = True,
    redo_cr_reject: bool = False,
    cr_config: Optional[CosmicRayRejectionConfig] = None,
    spectral_cr_review: bool = True,
    spectral_cr_resolving_power: Optional[float] = None,
    spectral_cr_config: Optional[SpectralCRConfig] = None,
    join_only: bool = False,
) -> None:
    object_dir = object_dir.expanduser().resolve()
    if not object_dir.exists():
        raise FileNotFoundError(object_dir)

    if join_only:
        if standard is True:
            raise ValueError("--join-only is only valid for science reductions")
        join_existing_science_sides(
            object_dir,
            show_plots=show_plots,
        )
        return

    if standard is None:
        standard = prompt("Is this a standard star? (y/n)", "n").lower().startswith("y")

    side = side.lower().strip()
    if side == "both":
        sides = ("BLUE", "RED")
    elif side == "blue":
        sides = ("BLUE",)
    elif side == "red":
        sides = ("RED",)
    else:
        raise ValueError("side must be one of: blue, red, both")

    if cr_config is None:
        cr_config = CosmicRayRejectionConfig()
    if spectral_cr_config is None:
        spectral_cr_config = SpectralCRConfig()

    product_types = set()
    for side_name in sides:
        detected_type, files = _side_product_files(object_dir, side_name)
        if files:
            product_types.add(detected_type)
    if not product_types:
        raise FileNotFoundError(f"No requested-side *_icubed.fits or *_icubes.fits files found under {object_dir}")
    if len(product_types) != 1:
        raise ValueError(
            f"Requested sides contain mixed cube products: {sorted(product_types)}. "
            "All standards and science targets must use the same cube type."
        )
    product_type = next(iter(product_types))
    print(f"[{object_dir.name}] Input cube product: *_{product_type}.fits")

    coadd_paths: Dict[str, Path] = {}
    aperture_template: Optional[_ApertureTemplate] = None
    aperture_template_from_current_run = False
    if len(sides) == 1 and not redo_apertures:
        opposite_side = "RED" if sides[0] == "BLUE" else "BLUE"
        aperture_template = _saved_aperture_template(object_dir, opposite_side)
    for side_name in sides:
        result = _extract_side(
            object_dir,
            side_name,
            show_plots=show_plots,
            redo_apertures=redo_apertures,
            cr_reject=cr_reject,
            redo_cr_reject=redo_cr_reject,
            cr_config=cr_config,
            product_type=product_type,
            spectral_cr_review=(
                bool(spectral_cr_review) and ((not standard) or show_plots)
            ),
            spectral_cr_resolving_power=spectral_cr_resolving_power,
            spectral_cr_config=spectral_cr_config,
            initial_aperture_template=aperture_template,
            prefer_initial_aperture_template=aperture_template_from_current_run,
            show_coadd_diagnostic=(not standard) or show_plots,
        )
        if result is not None:
            coadd_paths[side_name] = result.coadd_path
            aperture_template = result.aperture_template
            aperture_template_from_current_run = True

    if not coadd_paths:
        raise FileNotFoundError(f"No requested-side cube files found under {object_dir}")

    resolved_calib_dir = _project_calib_dir(object_dir, calib_dir)
    if standard:
        _build_standard_calibrations(
            object_dir,
            coadd_paths,
            resolved_calib_dir,
            show_plots=show_plots,
            product_type=product_type,
        )
    else:
        _apply_science_calibrations(
            object_dir,
            coadd_paths,
            resolved_calib_dir,
            show_plots=show_plots,
            product_type=product_type,
        )
