from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths


# Central resolving powers from the WMKO KCWI configuration table. Red-side
# values are published lower limits, so those estimates are marked approximate.
_GRATING_BASE_RESOLVING_POWER = {
    "BL": 900.0,
    "BM": 2000.0,
    "BH1": 4500.0,
    "BH2": 4500.0,
    "BH3": 4500.0,
    "RL": 500.0,
    "RM1": 1400.0,
    "RM2": 1400.0,
    "RH1": 3250.0,
    "RH2": 3250.0,
    "RH3": 3250.0,
    "RH4": 3250.0,
}
_SLICER_RESOLUTION_FACTOR = {
    "LARGE": 1.0,
    "MEDIUM": 2.0,
    "SMALL": 4.0,
}


@dataclass(frozen=True)
class ResolvingPowerEstimate:
    value: float
    grating: str
    slicer: str
    source: str
    approximate: bool


@dataclass(frozen=True)
class SpectralCRConfig:
    detection_sigma: float = 5.0
    max_lsf_fraction: float = 0.65
    mask_support_sigma: float = 1.5
    continuum_lsf_widths: float = 6.0
    min_continuum_window: int = 31


@dataclass(frozen=True)
class SpectralCRCandidate:
    peak_index: int
    start_index: int
    stop_index: int
    wavelength: float
    snr: float
    polarity: str
    prominence_sigma: float
    measured_fwhm_pixels: float
    expected_fwhm_pixels: float
    measured_fwhm_angstrom: float
    expected_fwhm_angstrom: float
    width_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralCRDetection:
    candidates: Tuple[SpectralCRCandidate, ...]
    continuum: np.ndarray
    noise: np.ndarray
    significance: np.ndarray
    expected_fwhm_pixels: np.ndarray
    resolving_power: float
    config: SpectralCRConfig


def _normalized_header_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def resolving_power_from_header(
    header: Mapping[str, object],
    side: str,
    *,
    override: Optional[float] = None,
) -> Optional[ResolvingPowerEstimate]:
    """Infer central resolving power from KCWI grating and slicer keywords."""
    if override is not None:
        value = float(override)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Spectral CR resolving-power override must be positive")
        return ResolvingPowerEstimate(
            value=value,
            grating="OVERRIDE",
            slicer="OVERRIDE",
            source="command-line override",
            approximate=False,
        )

    side_name = str(side).upper().strip()
    if side_name not in {"BLUE", "RED"}:
        raise ValueError("KCWI side must be BLUE or RED")
    grating_keys = ("BGRATNAM", "GRATNAME", "GRATING") if side_name == "BLUE" else (
        "RGRATNAM",
        "GRATNAME",
        "GRATING",
    )
    grating = ""
    for key in grating_keys:
        if header.get(key) not in (None, ""):
            grating = _normalized_header_name(header[key])
            break

    slicer_value = header.get("IFUNAM", header.get("SLICER", ""))
    normalized_slicer = _normalized_header_name(slicer_value)
    slicer = next(
        (name for name in _SLICER_RESOLUTION_FACTOR if name in normalized_slicer),
        "",
    )
    if grating not in _GRATING_BASE_RESOLVING_POWER or not slicer:
        return None

    return ResolvingPowerEstimate(
        value=(
            _GRATING_BASE_RESOLVING_POWER[grating]
            * _SLICER_RESOLUTION_FACTOR[slicer]
        ),
        grating=grating,
        slicer=slicer.title(),
        source="WMKO KCWI grating/slicer configuration table",
        approximate=grating.startswith("R"),
    )


def _bounded_odd_window(value: int, size: int, *, minimum: int = 3) -> int:
    if size < minimum:
        return max(size, 1)
    value = max(int(value), minimum)
    if value % 2 == 0:
        value += 1
    largest = size if size % 2 else size - 1
    return min(value, largest)


def _filled_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=float)
    indices = np.arange(values.size, dtype=float)
    return np.interp(indices, indices[finite], values[finite])


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(finite))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def _local_noise(residual: np.ndarray, window: int) -> np.ndarray:
    local_center = median_filter(residual, size=window, mode="nearest")
    local_mad = median_filter(
        np.abs(residual - local_center),
        size=window,
        mode="nearest",
    )
    noise = 1.4826 * local_mad
    global_sigma = _robust_sigma(residual)
    floor = max(global_sigma * 0.25, np.finfo(float).eps)
    return np.maximum(noise, floor)


def _candidate_mask_bounds(
    significance: np.ndarray,
    peak: int,
    measured_fwhm_pixels: float,
    support_sigma: float,
) -> Tuple[int, int]:
    max_radius = max(int(np.ceil(measured_fwhm_pixels)), 1)
    start = peak
    while (
        start > max(peak - max_radius, 0)
        and significance[start - 1] >= support_sigma
    ):
        start -= 1
    stop = peak + 1
    while (
        stop < min(peak + max_radius + 1, significance.size)
        and significance[stop] >= support_sigma
    ):
        stop += 1
    return start, stop


def detect_cr_like_narrow_features(
    wavelength: np.ndarray,
    flux: np.ndarray,
    sigma: Optional[np.ndarray],
    *,
    resolving_power: float,
    config: SpectralCRConfig = SpectralCRConfig(),
) -> SpectralCRDetection:
    """Flag significant positive or negative features narrower than the KCWI LSF."""
    lam = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    if lam.ndim != 1 or values.ndim != 1 or lam.shape != values.shape:
        raise ValueError("Wavelength and flux must be matching one-dimensional arrays")
    if lam.size < 7:
        raise ValueError("At least seven spectral samples are required")
    if not np.isfinite(resolving_power) or resolving_power <= 0:
        raise ValueError("Resolving power must be positive")
    if config.detection_sigma <= 0:
        raise ValueError("Spectral CR detection sigma must be positive")
    if not 0 < config.max_lsf_fraction < 1:
        raise ValueError("Spectral CR maximum LSF fraction must be between 0 and 1")
    if not 0 < config.mask_support_sigma <= config.detection_sigma:
        raise ValueError("Spectral CR support sigma must be positive and no larger than detection sigma")

    lam_filled = _filled_1d(lam)
    flux_filled = _filled_1d(values)
    dispersion = np.abs(np.gradient(lam_filled))
    valid_dispersion = np.isfinite(dispersion) & (dispersion > 0)
    if not np.any(valid_dispersion):
        raise ValueError("Wavelength array has no usable dispersion")
    dispersion_floor = float(np.median(dispersion[valid_dispersion]))
    dispersion = np.where(valid_dispersion, dispersion, dispersion_floor)
    expected_fwhm_angstrom = np.abs(lam_filled) / float(resolving_power)
    expected_fwhm_pixels = expected_fwhm_angstrom / dispersion
    median_lsf_pixels = float(
        np.median(expected_fwhm_pixels[np.isfinite(expected_fwhm_pixels)])
    )

    continuum_window = _bounded_odd_window(
        max(
            int(config.min_continuum_window),
            int(np.ceil(config.continuum_lsf_widths * median_lsf_pixels)),
        ),
        lam.size,
    )
    continuum = median_filter(flux_filled, size=continuum_window, mode="nearest")
    residual = flux_filled - continuum
    noise = _local_noise(residual, continuum_window)
    if sigma is not None:
        supplied_sigma = np.asarray(sigma, dtype=float)
        if supplied_sigma.shape != values.shape:
            raise ValueError("Spectrum uncertainty must match the flux shape")
        valid_sigma = np.isfinite(supplied_sigma) & (supplied_sigma > 0)
        noise = np.where(valid_sigma, np.maximum(noise, np.abs(supplied_sigma)), noise)

    valid = np.isfinite(lam) & np.isfinite(values) & np.isfinite(noise) & (noise > 0)
    significance = np.full(values.shape, -np.inf, dtype=float)
    significance[valid] = residual[valid] / noise[valid]
    peak_distance = max(int(np.floor(0.5 * median_lsf_pixels)), 1)
    raw_candidates = []
    for sign, polarity in ((1.0, "positive"), (-1.0, "negative")):
        signed_significance = sign * significance
        profile = np.where(
            np.isfinite(signed_significance),
            np.maximum(signed_significance, 0.0),
            0.0,
        )
        peaks, properties = find_peaks(
            profile,
            height=float(config.detection_sigma),
            prominence=max(0.5 * float(config.detection_sigma), 1.0),
            distance=peak_distance,
        )
        if peaks.size:
            widths, _height, _left, _right = peak_widths(
                profile,
                peaks,
                rel_height=0.5,
            )
            prominences = properties.get(
                "prominences",
                np.zeros(peaks.size, dtype=float),
            )
            for peak, width, prominence in zip(peaks, widths, prominences):
                expected_pixels = float(expected_fwhm_pixels[peak])
                if (
                    not np.isfinite(width)
                    or not np.isfinite(expected_pixels)
                    or expected_pixels <= 0
                ):
                    continue
                width_ratio = float(width / expected_pixels)
                if width_ratio >= float(config.max_lsf_fraction):
                    continue
                start, stop = _candidate_mask_bounds(
                    signed_significance,
                    int(peak),
                    float(width),
                    float(config.mask_support_sigma),
                )
                if start == 0 or stop >= values.size:
                    continue
                raw_candidates.append(
                    SpectralCRCandidate(
                        peak_index=int(peak),
                        start_index=int(start),
                        stop_index=int(stop),
                        wavelength=float(lam[peak]),
                        snr=float(significance[peak]),
                        polarity=polarity,
                        prominence_sigma=float(prominence),
                        measured_fwhm_pixels=float(width),
                        expected_fwhm_pixels=expected_pixels,
                        measured_fwhm_angstrom=float(width * dispersion[peak]),
                        expected_fwhm_angstrom=float(expected_fwhm_angstrom[peak]),
                        width_ratio=width_ratio,
                    )
                )

    selected = []
    for candidate in sorted(raw_candidates, key=lambda item: abs(item.snr), reverse=True):
        overlaps = any(
            candidate.start_index < other.stop_index
            and other.start_index < candidate.stop_index
            for other in selected
        )
        if not overlaps:
            selected.append(candidate)
    selected.sort(key=lambda item: item.peak_index)

    return SpectralCRDetection(
        candidates=tuple(selected),
        continuum=continuum,
        noise=noise,
        significance=significance,
        expected_fwhm_pixels=expected_fwhm_pixels,
        resolving_power=float(resolving_power),
        config=config,
    )


def interpolate_rejected_candidate(
    wavelength: np.ndarray,
    flux: np.ndarray,
    sigma: Optional[np.ndarray],
    candidate: SpectralCRCandidate,
    *,
    continuum: np.ndarray,
    noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Replace one reviewed feature and include interpolation-model uncertainty."""
    lam = np.asarray(wavelength, dtype=float)
    values = np.asarray(flux, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if not (lam.shape == values.shape == continuum.shape == noise.shape):
        raise ValueError("Interpolation arrays must have matching shapes")

    start = int(candidate.start_index)
    stop = int(candidate.stop_index)
    left = start - 1
    right = stop
    if left < 0 or right >= values.size:
        raise ValueError("Candidate does not have interpolation samples on both sides")
    if not np.all(np.isfinite([lam[left], lam[right], values[left], values[right]])):
        raise ValueError("Candidate interpolation boundary is not finite")
    denominator = lam[right] - lam[left]
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("Candidate interpolation wavelengths are degenerate")

    cleaned_flux = values.copy()
    fractions = (lam[start:stop] - lam[left]) / denominator
    cleaned_flux[start:stop] = (
        (1.0 - fractions) * values[left] + fractions * values[right]
    )

    if sigma is None:
        cleaned_sigma = noise.copy()
        uncertainty_source = "local robust noise estimate"
    else:
        supplied_sigma = np.asarray(sigma, dtype=float)
        if supplied_sigma.shape != values.shape:
            raise ValueError("Spectrum uncertainty must match the flux shape")
        cleaned_sigma = supplied_sigma.copy()
        uncertainty_source = "input uncertainty plus local interpolation scatter"

    left_sigma = cleaned_sigma[left]
    right_sigma = cleaned_sigma[right]
    if not np.isfinite(left_sigma) or left_sigma <= 0:
        left_sigma = noise[left]
    if not np.isfinite(right_sigma) or right_sigma <= 0:
        right_sigma = noise[right]

    radius = max(int(np.ceil(3.0 * candidate.expected_fwhm_pixels)), 5)
    side_start = max(start - radius, 0)
    side_stop = min(stop + radius, values.size)
    side_mask = np.ones(side_stop - side_start, dtype=bool)
    side_mask[max(start - side_start, 0) : max(stop - side_start, 0)] = False
    side_residual = (values - continuum)[side_start:side_stop][side_mask]
    local_scatter = _robust_sigma(side_residual)
    local_noise = noise[side_start:side_stop][side_mask]
    finite_local_noise = local_noise[np.isfinite(local_noise) & (local_noise > 0)]
    if finite_local_noise.size:
        local_scatter = max(local_scatter, float(np.median(finite_local_noise)))

    interpolation_sigma = np.sqrt(
        ((1.0 - fractions) * left_sigma) ** 2
        + (fractions * right_sigma) ** 2
        + local_scatter**2
    )
    cleaned_sigma[start:stop] = interpolation_sigma
    details = {
        "left_index": int(left),
        "right_index": int(right),
        "left_wavelength": float(lam[left]),
        "right_wavelength": float(lam[right]),
        "local_interpolation_scatter": float(local_scatter),
        "uncertainty_source": uncertainty_source,
        "interpolated_sigma_min": float(np.min(interpolation_sigma)),
        "interpolated_sigma_max": float(np.max(interpolation_sigma)),
    }
    return cleaned_flux, cleaned_sigma, details


def config_to_dict(config: SpectralCRConfig) -> dict[str, object]:
    return asdict(config)


def candidates_to_dict(
    candidates: Sequence[SpectralCRCandidate],
) -> list[dict[str, object]]:
    return [candidate.to_dict() for candidate in candidates]
