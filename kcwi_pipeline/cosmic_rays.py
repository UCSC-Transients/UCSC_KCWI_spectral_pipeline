from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from scipy.ndimage import label, median_filter


@dataclass(frozen=True)
class CosmicRayRejectionConfig:
    sigma: float = 5.5
    motion_sigma: float = 3.25
    grow_sigma: float = 1.75
    spectral_window: int = 7
    spatial_window: int = 21
    max_spatial_footprint: int = 10
    max_spectral_neighbors: int = 1
    require_slice_motion: bool = True
    min_slice_shift_pixels: int = 1
    slice_shift_pixels: int = 3
    slice_motion_axis: str = "y"
    min_track_length: int = 4
    max_track_span: int = 64
    max_track_gap: int = 1
    track_fit_tolerance: float = 1.5
    mask_margin: int = 1
    neighbor_veto_fraction: float = 0.6
    workers: int = 0


@dataclass
class CosmicRayRejectionResult:
    cleaned_cube: np.ndarray
    mask: np.ndarray
    n_flagged: int
    fraction_flagged: float
    config: CosmicRayRejectionConfig
    diagnostics: Dict[str, int]
    runtime_seconds: float
    cleaned_uncert: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _Spike:
    row: int
    start: int
    stop: int
    centroid: float
    peak_sigma: float
    integrated_sigma: float

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class _Track:
    spikes: Tuple[_Spike, ...]
    slope: float
    intercept: float
    rms: float


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1


def _filled_cube(cube: np.ndarray) -> np.ndarray:
    finite = np.isfinite(cube)
    if np.any(finite):
        fill = float(np.nanmedian(cube[finite]))
    else:
        fill = 0.0
    return np.where(finite, cube, fill).astype(np.float32, copy=False)


def _plane_sigma(residual: np.ndarray) -> np.ndarray:
    finite = np.isfinite(residual)
    work = np.where(finite, residual, np.nan)
    med = np.nanmedian(work, axis=(1, 2))
    mad = np.nanmedian(np.abs(work - med[:, None, None]), axis=(1, 2))
    sigma = 1.4826 * mad
    finite_sigma = np.isfinite(sigma) & (sigma > 0)
    if np.any(finite_sigma):
        floor = float(np.nanmedian(sigma[finite_sigma]))
    else:
        floor = 1.0
    sigma = np.where(finite_sigma, sigma, floor)
    sigma = np.maximum(sigma, 1e-6)
    return sigma.astype(np.float32)


def _line_sigma(residual: np.ndarray, axis: str) -> np.ndarray:
    axis = axis.lower().strip()
    finite = np.isfinite(residual)
    work = np.where(finite, residual, np.nan)

    if axis == "x":
        med = np.nanmedian(work, axis=2)
        mad = np.nanmedian(np.abs(work - med[:, :, None]), axis=2)
        sigma = 1.4826 * mad
    else:
        med = np.nanmedian(work, axis=1)
        mad = np.nanmedian(np.abs(work - med[:, None, :]), axis=1)
        sigma = 1.4826 * mad

    finite_sigma = np.isfinite(sigma) & (sigma > 0)
    if np.any(finite_sigma):
        floor = float(np.nanmedian(sigma[finite_sigma]))
    else:
        floor = 1.0
    sigma = np.where(finite_sigma, sigma, floor)
    sigma = np.maximum(sigma, 1e-6).astype(np.float32)
    if axis == "x":
        return sigma[:, :, None]
    return sigma[:, None, :]


def _neighbor_count_along_wavelength(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dz in range(-radius, radius + 1):
        if dz == 0:
            continue
        if dz < 0:
            count[:dz] += mask[-dz:]
        else:
            count[dz:] += mask[:-dz]
    return count


def _validate_config(config: CosmicRayRejectionConfig) -> str:
    axis = config.slice_motion_axis.lower().strip()
    if axis not in {"x", "y"}:
        raise ValueError("slice_motion_axis must be 'x' or 'y'")
    if not (config.sigma >= config.motion_sigma >= config.grow_sigma > 0):
        raise ValueError("CR thresholds must satisfy sigma >= motion_sigma >= grow_sigma > 0")
    if int(config.max_spatial_footprint) < 1:
        raise ValueError("max_spatial_footprint must be at least 1")
    if int(config.min_track_length) < 2:
        raise ValueError("min_track_length must be at least 2")
    if int(config.max_track_span) < 0:
        raise ValueError("max_track_span cannot be negative")
    if int(config.max_track_gap) < 0:
        raise ValueError("max_track_gap cannot be negative")
    if int(config.slice_shift_pixels) < 1:
        raise ValueError("slice_shift_pixels must be at least 1")
    if float(config.track_fit_tolerance) <= 0:
        raise ValueError("track_fit_tolerance must be positive")
    if not 0.0 <= float(config.neighbor_veto_fraction) <= 1.0:
        raise ValueError("neighbor_veto_fraction must be between 0 and 1")
    if int(config.workers) < 0:
        raise ValueError("workers cannot be negative")
    return axis


def resolve_cr_workers(
    requested: int,
    nslices: int,
    *,
    cpu_count: Optional[int] = None,
) -> int:
    """Resolve 0=auto CR workers while limiting process and memory pressure."""
    requested = int(requested)
    nslices = int(nslices)
    if requested < 0:
        raise ValueError("workers cannot be negative")
    if nslices < 1:
        return 1
    if requested > 0:
        return min(requested, nslices)

    detected = os.cpu_count() if cpu_count is None else cpu_count
    available = max(int(detected), 1) if detected is not None else 1
    automatic = min(8, max(available - 2, 1))
    return min(automatic, nslices)


def _panel(array: np.ndarray, axis: str, slice_index: int) -> np.ndarray:
    if axis == "x":
        return array[:, slice_index, :]
    return array[:, :, slice_index]


def _set_panel(array: np.ndarray, axis: str, slice_index: int, panel: np.ndarray) -> None:
    if axis == "x":
        array[:, slice_index, :] = panel
    else:
        array[:, :, slice_index] = panel


def _true_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, breaks)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _find_spikes(
    significance: np.ndarray,
    good: np.ndarray,
    *,
    support_sigma: float,
    max_width: int,
) -> List[_Spike]:
    spikes: List[_Spike] = []
    support = good & (significance >= float(support_sigma))
    for row in range(support.shape[0]):
        for start, stop in _true_runs(support[row]):
            width = stop - start
            if width > max_width:
                continue
            weights = np.maximum(significance[row, start:stop], 0.0)
            total = float(np.sum(weights))
            if total > 0:
                coords = np.arange(start, stop, dtype=float)
                centroid = float(np.sum(coords * weights) / total)
            else:
                centroid = 0.5 * float(start + stop - 1)
            spikes.append(
                _Spike(
                    row=row,
                    start=start,
                    stop=stop,
                    centroid=centroid,
                    peak_sigma=float(np.max(significance[row, start:stop])),
                    integrated_sigma=total / np.sqrt(float(width)),
                )
            )
    return spikes


def _is_seed(spike: _Spike, seed_sigma: float) -> bool:
    return bool(max(spike.peak_sigma, spike.integrated_sigma) >= float(seed_sigma))


def _track_components(
    spikes: Sequence[_Spike],
    *,
    max_gap: int,
    max_step: int,
) -> List[List[_Spike]]:
    if not spikes:
        return []

    ordered = sorted(spikes, key=lambda spike: (spike.row, spike.centroid))
    by_row: Dict[int, List[int]] = {}
    for index, spike in enumerate(ordered):
        by_row.setdefault(spike.row, []).append(index)

    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    max_row_delta = max_gap + 1
    for left_index, left in enumerate(ordered):
        for delta in range(1, max_row_delta + 1):
            for right_index in by_row.get(left.row + delta, []):
                right = ordered[right_index]
                width_allowance = 0.5 * float(max(left.width, right.width))
                if abs(right.centroid - left.centroid) <= max_step * delta + width_allowance:
                    union(left_index, right_index)

    components: Dict[int, List[_Spike]] = {}
    for index, spike in enumerate(ordered):
        components.setdefault(find(index), []).append(spike)
    return list(components.values())


def _select_near_line(
    spikes: Sequence[_Spike],
    *,
    seed: _Spike,
    slope: float,
    intercept: float,
    tolerance: float,
) -> List[_Spike]:
    by_row: Dict[int, List[_Spike]] = {}
    for spike in spikes:
        by_row.setdefault(spike.row, []).append(spike)

    selected: List[_Spike] = []
    for row, options in by_row.items():
        if row == seed.row:
            selected.append(seed)
            continue
        predicted = intercept + slope * row
        valid = [
            spike
            for spike in options
            if abs(spike.centroid - predicted) <= tolerance + 0.5 * spike.width
        ]
        if valid:
            selected.append(
                min(
                    valid,
                    key=lambda spike: (
                        abs(spike.centroid - predicted),
                        -spike.integrated_sigma,
                    ),
                )
            )
    return sorted(selected, key=lambda spike: spike.row)


def _segment_containing_seed(
    spikes: Sequence[_Spike],
    *,
    seed: _Spike,
    max_gap: int,
) -> List[_Spike]:
    if not spikes:
        return []
    segments: List[List[_Spike]] = [[spikes[0]]]
    for spike in spikes[1:]:
        if spike.row - segments[-1][-1].row <= max_gap + 1:
            segments[-1].append(spike)
        else:
            segments.append([spike])
    for segment in segments:
        if seed in segment:
            return segment
    return []


def _fit_seeded_track(
    component: Sequence[_Spike],
    *,
    seed: _Spike,
    config: CosmicRayRejectionConfig,
) -> Optional[_Track]:
    slope_candidates = {0.0}
    max_step = float(config.slice_shift_pixels)
    for spike in component:
        delta = spike.row - seed.row
        if delta == 0:
            continue
        slope = (spike.centroid - seed.centroid) / float(delta)
        if abs(slope) <= max_step:
            slope_candidates.add(round(slope * 4.0) / 4.0)

    best: Optional[Tuple[float, _Track]] = None
    tolerance = float(config.track_fit_tolerance)
    for initial_slope in slope_candidates:
        slope = initial_slope
        intercept = seed.centroid - slope * seed.row
        selected: List[_Spike] = []
        for _ in range(3):
            selected = _select_near_line(
                component,
                seed=seed,
                slope=slope,
                intercept=intercept,
                tolerance=tolerance,
            )
            selected = _segment_containing_seed(
                selected,
                seed=seed,
                max_gap=int(config.max_track_gap),
            )
            if len(selected) < 2:
                break
            rows = np.array([spike.row for spike in selected], dtype=float)
            centers = np.array([spike.centroid for spike in selected], dtype=float)
            weights = np.sqrt(
                np.array([max(spike.integrated_sigma, 1.0) for spike in selected], dtype=float)
            )
            slope, intercept = np.polyfit(rows, centers, 1, w=weights)
            if abs(slope) > max_step:
                selected = []
                break

        if len(selected) < int(config.min_track_length):
            continue
        if not any(_is_seed(spike, config.sigma) for spike in selected):
            continue

        rows = np.array([spike.row for spike in selected], dtype=float)
        centers = np.array([spike.centroid for spike in selected], dtype=float)
        residuals = centers - (intercept + slope * rows)
        rms = float(np.sqrt(np.mean(residuals**2)))
        if rms > tolerance:
            continue
        displacement = abs(slope) * float(rows[-1] - rows[0])
        if config.require_slice_motion and displacement < float(config.min_slice_shift_pixels):
            continue

        track = _Track(tuple(selected), float(slope), float(intercept), rms)
        score = (
            8.0 * len(selected)
            + sum(min(spike.integrated_sigma, 25.0) for spike in selected)
            - 4.0 * rms
        )
        if best is None or score > best[0]:
            best = (score, track)
    return None if best is None else best[1]


def _extract_tracks(
    spikes: Sequence[_Spike],
    *,
    config: CosmicRayRejectionConfig,
) -> List[_Track]:
    tracks: List[_Track] = []
    for component in _track_components(
        spikes,
        max_gap=int(config.max_track_gap),
        max_step=int(config.slice_shift_pixels),
    ):
        remaining = list(component)
        while remaining:
            seeds = [spike for spike in remaining if _is_seed(spike, config.sigma)]
            if not seeds:
                break
            # A bright continuum trace can contribute thousands of seeds to one
            # component. A small set spanning the strongest peaks is sufficient
            # to initialize the same line fits without quadratic runtime.
            seeds = sorted(
                seeds,
                key=lambda spike: max(spike.peak_sigma, spike.integrated_sigma),
                reverse=True,
            )[:12]
            candidates = [
                track
                for seed in seeds
                if (track := _fit_seeded_track(remaining, seed=seed, config=config)) is not None
            ]
            if not candidates:
                break
            track = max(
                candidates,
                key=lambda item: (
                    len(item.spikes),
                    sum(spike.integrated_sigma for spike in item.spikes),
                    -item.rms,
                ),
            )
            tracks.append(track)
            used = set(track.spikes)
            remaining = [spike for spike in remaining if spike not in used]
    return tracks


def _detect_slice_tracks(
    task: Tuple[np.ndarray, np.ndarray, CosmicRayRejectionConfig],
) -> Tuple[List[_Spike], List[_Track]]:
    """Find spikes and tracks in one detector-slice panel."""
    significance, good, config = task
    spikes = _find_spikes(
        significance,
        good,
        support_sigma=float(config.motion_sigma),
        max_width=int(config.max_spatial_footprint),
    )
    return spikes, _extract_tracks(spikes, config=config)


def _slice_detection_tasks(
    significance: np.ndarray,
    good: np.ndarray,
    axis: str,
    nslices: int,
    config: CosmicRayRejectionConfig,
    *,
    copy_panels: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray, CosmicRayRejectionConfig]]:
    for slice_index in range(nslices):
        significance_panel = _panel(significance, axis, slice_index)
        good_panel = _panel(good, axis, slice_index)
        if copy_panels:
            significance_panel = np.ascontiguousarray(significance_panel)
            good_panel = np.ascontiguousarray(good_panel)
        yield significance_panel, good_panel, config


def _run_slice_detection(
    significance: np.ndarray,
    good: np.ndarray,
    axis: str,
    nslices: int,
    config: CosmicRayRejectionConfig,
    executor: Optional[ProcessPoolExecutor],
) -> List[Tuple[List[_Spike], List[_Track]]]:
    tasks = _slice_detection_tasks(
        significance,
        good,
        axis,
        nslices,
        config,
        copy_panels=executor is not None,
    )
    if executor is None:
        return [_detect_slice_tracks(task) for task in tasks]
    return list(executor.map(_detect_slice_tracks, tasks, chunksize=1))


def _track_match_fraction(track: _Track, neighbor: _Track, *, tolerance: float) -> float:
    """Return the fraction of one track matched by fitted spikes in another."""
    neighbor_by_row = {spike.row: spike for spike in neighbor.spikes}
    matched = 0
    for spike in track.spikes:
        other = neighbor_by_row.get(spike.row)
        if other is None:
            continue
        radius = tolerance + 0.5 * max(spike.width, other.width) + 1.0
        if abs(spike.centroid - other.centroid) <= radius:
            matched += 1
    return matched / float(len(track.spikes))


def _neighbor_track_coherence(
    tracks_by_slice: Sequence[Sequence[_Track]],
    *,
    slice_index: int,
    track: _Track,
    tolerance: float,
) -> Tuple[float, int]:
    neighbors = [
        index
        for index in (slice_index - 1, slice_index + 1)
        if 0 <= index < len(tracks_by_slice)
    ]
    matches: List[float] = []
    comparisons = 0
    for neighbor_index in neighbors:
        for neighbor_track in tracks_by_slice[neighbor_index]:
            comparisons += 1
            matches.append(
                _track_match_fraction(track, neighbor_track, tolerance=tolerance)
            )
    return (max(matches, default=0.0), comparisons)


def _nearest_growth_run(
    row: np.ndarray,
    *,
    predicted: float,
    max_distance: float,
) -> Optional[Tuple[int, int]]:
    runs = _true_runs(row)
    if not runs:
        return None

    def distance(run: Tuple[int, int]) -> float:
        start, stop = run
        if start <= predicted <= stop - 1:
            return 0.0
        return min(abs(predicted - start), abs(predicted - (stop - 1)))

    best = min(runs, key=distance)
    return best if distance(best) <= max_distance else None


def _grow_track_mask(
    significance: np.ndarray,
    track: _Track,
    *,
    config: CosmicRayRejectionConfig,
) -> np.ndarray:
    mask = np.zeros(significance.shape, dtype=bool)
    first_row = track.spikes[0].row
    last_row = track.spikes[-1].row
    observed = {spike.row: spike for spike in track.spikes}
    median_width = max(int(round(np.median([spike.width for spike in track.spikes]))), 1)
    max_width = int(config.max_spatial_footprint)
    margin = max(int(config.mask_margin), 0)
    growth = significance >= float(config.grow_sigma)

    for row in range(first_row, last_row + 1):
        predicted = track.intercept + track.slope * row
        spike = observed.get(row)
        run = _nearest_growth_run(
            growth[row],
            predicted=predicted,
            max_distance=float(config.track_fit_tolerance) + 0.5 * median_width,
        )
        if run is None:
            half_width = 0.5 * median_width
            start = int(np.floor(predicted - half_width))
            stop = int(np.ceil(predicted + half_width)) + 1
        else:
            start, stop = run
        if spike is not None:
            start = min(start, spike.start)
            stop = max(stop, spike.stop)

        allowed_width = max_width
        if stop - start > allowed_width:
            start = int(np.floor(predicted - 0.5 * allowed_width))
            stop = start + allowed_width
        start = max(start - margin, 0)
        stop = min(stop + margin, significance.shape[1])
        mask[row, start:stop] = True

    for direction in (-1, 1):
        row = first_row - 1 if direction < 0 else last_row + 1
        if not 0 <= row < significance.shape[0]:
            continue
        predicted = track.intercept + track.slope * row
        run = _nearest_growth_run(
            growth[row],
            predicted=predicted,
            max_distance=float(config.track_fit_tolerance) + 0.5 * median_width,
        )
        if run is not None and run[1] - run[0] <= max_width:
            start = max(run[0] - margin, 0)
            stop = min(run[1] + margin, significance.shape[1])
            mask[row, start:stop] = True
    return mask


def _spectral_interpolation_model(
    cube: np.ndarray,
    mask: np.ndarray,
    fallback: np.ndarray,
    *,
    sideband: int,
) -> np.ndarray:
    """Interpolate masked wavelength runs independently at each spaxel."""
    model = fallback.astype(np.float32, copy=True)
    sideband = max(int(sideband), 1)
    spatial_positions = np.argwhere(np.any(mask, axis=0))
    for y, x in spatial_positions:
        spectrum = cube[:, y, x]
        spectrum_mask = mask[:, y, x]
        usable = np.isfinite(spectrum) & ~spectrum_mask
        for start, stop in _true_runs(spectrum_mask):
            left = np.flatnonzero(usable[:start])[-sideband:]
            right = np.flatnonzero(usable[stop:])[:sideband] + stop
            rows = np.arange(start, stop, dtype=float)
            if left.size and right.size:
                left_row = float(np.median(left))
                right_row = float(np.median(right))
                left_value = float(np.median(spectrum[left]))
                right_value = float(np.median(spectrum[right]))
                model[start:stop, y, x] = np.interp(
                    rows,
                    [left_row, right_row],
                    [left_value, right_value],
                )
            elif left.size:
                model[start:stop, y, x] = float(np.median(spectrum[left]))
            elif right.size:
                model[start:stop, y, x] = float(np.median(spectrum[right]))
    return model


def _replace_masked_spectrally(
    cube: np.ndarray,
    mask: np.ndarray,
    fallback: np.ndarray,
    *,
    sideband: int,
) -> np.ndarray:
    replacement = _spectral_interpolation_model(
        cube,
        mask,
        fallback,
        sideband=sideband,
    )
    cleaned = cube.astype(np.float32, copy=True)
    cleaned[mask] = replacement[mask]
    return cleaned


def _uncertainty_of_sideband_median(
    sigma_values: np.ndarray,
    fallback_sigma: float,
) -> float:
    """Approximate the standard error of a median from independent samples."""
    sigma_values = np.abs(np.asarray(sigma_values, dtype=float))
    good = np.isfinite(sigma_values) & (sigma_values > 0)
    if np.any(good):
        # For Gaussian samples, the standard error of a median is about
        # 1.253 times that of the corresponding mean.
        return float(1.2533 * np.sqrt(np.sum(sigma_values[good] ** 2)) / np.count_nonzero(good))
    return float(fallback_sigma)


def _propagate_replacement_uncertainty(
    cube: np.ndarray,
    uncert: Optional[np.ndarray],
    noise: np.ndarray,
    mask: np.ndarray,
    fallback: np.ndarray,
    *,
    sideband: int,
) -> Optional[np.ndarray]:
    """Propagate endpoint and local-model uncertainty into CR replacements."""
    if uncert is None or uncert.shape != cube.shape:
        return None

    cleaned_uncert = np.abs(uncert.astype(np.float32, copy=True))
    sideband = max(int(sideband), 1)
    spatial_positions = np.argwhere(np.any(mask, axis=0))
    for y, x in spatial_positions:
        spectrum = cube[:, y, x].astype(float, copy=False)
        spectrum_mask = mask[:, y, x]
        usable = np.isfinite(spectrum) & ~spectrum_mask
        source_sigma = cleaned_uncert[:, y, x].astype(float, copy=False)
        fallback_sigma = np.abs(noise[:, y, x].astype(float, copy=False))
        model = fallback[:, y, x].astype(float, copy=False)

        for start, stop in _true_runs(spectrum_mask):
            left = np.flatnonzero(usable[:start])[-sideband:]
            right = np.flatnonzero(usable[stop:])[:sideband] + stop
            context = np.concatenate([left, right])
            context_residual = spectrum[context] - model[context]
            finite_residual = context_residual[np.isfinite(context_residual)]
            local_scatter = 0.0
            if finite_residual.size:
                center = float(np.median(finite_residual))
                local_scatter = float(
                    1.4826 * np.median(np.abs(finite_residual - center))
                )
                if not np.isfinite(local_scatter) or local_scatter <= 0:
                    local_scatter = float(np.std(finite_residual))
            if not np.isfinite(local_scatter) or local_scatter < 0:
                local_scatter = 0.0

            rows = np.arange(start, stop, dtype=float)
            local_floor = fallback_sigma[start:stop]
            if left.size and right.size:
                left_row = float(np.median(left))
                right_row = float(np.median(right))
                left_sigma = _uncertainty_of_sideband_median(
                    source_sigma[left],
                    float(np.nanmedian(fallback_sigma[left])),
                )
                right_sigma = _uncertainty_of_sideband_median(
                    source_sigma[right],
                    float(np.nanmedian(fallback_sigma[right])),
                )
                fraction = (rows - left_row) / (right_row - left_row)
                replacement_sigma = np.sqrt(
                    ((1.0 - fraction) * left_sigma) ** 2
                    + (fraction * right_sigma) ** 2
                    + local_scatter ** 2
                )
            elif left.size or right.size:
                side = left if left.size else right
                side_sigma = _uncertainty_of_sideband_median(
                    source_sigma[side],
                    float(np.nanmedian(fallback_sigma[side])),
                )
                replacement_sigma = np.full(
                    rows.shape,
                    np.sqrt(side_sigma ** 2 + local_scatter ** 2),
                    dtype=float,
                )
            else:
                replacement_sigma = local_floor.copy()

            # A modeled CR replacement should never be assigned less uncertainty
            # than either the original voxel or the local noise estimate.
            local_floor = np.where(np.isfinite(local_floor), local_floor, 0.0)
            replacement_sigma = np.maximum(replacement_sigma, local_floor)
            original_floor = np.where(
                np.isfinite(source_sigma[start:stop]),
                source_sigma[start:stop],
                0.0,
            )
            replacement_sigma = np.maximum(
                replacement_sigma,
                original_floor,
            )
            cleaned_uncert[start:stop, y, x] = replacement_sigma.astype(np.float32)

    return cleaned_uncert


def _keep_compact_spatial_components(mask: np.ndarray, max_footprint: int) -> np.ndarray:
    max_footprint = max(int(max_footprint), 1)
    compact = np.zeros(mask.shape, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    for k in range(mask.shape[0]):
        plane = mask[k]
        if not np.any(plane):
            continue
        labels, nlab = label(plane, structure=structure)
        if nlab == 0:
            continue
        counts = np.bincount(labels.ravel())
        keep_labels = np.flatnonzero((counts > 0) & (counts <= max_footprint))
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size:
            compact[k] = np.isin(labels, keep_labels)
    return compact


def reject_cosmic_rays(
    cube: np.ndarray,
    uncert: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    *,
    config: CosmicRayRejectionConfig = CosmicRayRejectionConfig(),
) -> CosmicRayRejectionResult:
    """Reject narrow CR tracks moving within individual KCWI detector slices.

    For y-axis motion, every fixed-x slice is analyzed as an independent
    wavelength-by-y image (and vice versa for x-axis motion). Positive spatial
    spikes are linked into provisional tracks and then validated against an
    interpolated same-spaxel spectral baseline. Only tracks significant relative
    to both models are masked. A candidate is rejected as likely astronomical
    emission when an adjacent detector slice contains a separately fitted track
    with matching wavelength rows and centroids.
    """
    if cube.ndim != 3:
        raise ValueError(f"Cosmic-ray rejection expects a 3D cube, got shape {cube.shape}")

    started = perf_counter()
    axis = _validate_config(config)
    spectral_window = _odd_at_least(config.spectral_window, 3)
    minimum_line_window = 2 * int(config.max_spatial_footprint) + 1
    line_window = _odd_at_least(config.spatial_window, minimum_line_window)
    work = _filled_cube(cube)
    finite = np.isfinite(cube)

    median_spectral_model = median_filter(work, size=(spectral_window, 1, 1), mode="reflect")
    if axis == "x":
        line_model = median_filter(work, size=(1, 1, line_window), mode="reflect")
    else:
        line_model = median_filter(work, size=(1, line_window, 1), mode="reflect")
    spatial_residual = cube.astype(np.float32, copy=False) - line_model

    plane_sig = _plane_sigma(spatial_residual)[:, None, None]
    line_sig = _line_sigma(spatial_residual, axis)
    residual_sig = np.maximum(plane_sig, line_sig)
    if uncert is not None and uncert.shape == cube.shape:
        noise = np.maximum(np.abs(uncert.astype(np.float32, copy=False)), residual_sig)
    else:
        noise = residual_sig

    good_input = finite.copy()
    if flags is not None and flags.shape == cube.shape:
        good_input &= flags == 0

    spatial_significance = np.full(cube.shape, -np.inf, dtype=np.float32)
    np.divide(
        spatial_residual,
        noise,
        out=spatial_significance,
        where=good_input & (noise > 0),
    )
    core: Optional[np.ndarray] = None
    loose: Optional[np.ndarray] = None
    if not config.require_slice_motion:
        global_spectral_residual = cube.astype(np.float32, copy=False) - median_spectral_model
        global_spectral_significance = np.full(cube.shape, -np.inf, dtype=np.float32)
        np.divide(
            global_spectral_residual,
            noise,
            out=global_spectral_significance,
            where=good_input & (noise > 0),
        )
        static_significance = np.minimum(spatial_significance, global_spectral_significance)
        core = good_input & (static_significance >= float(config.sigma))
        loose = good_input & (static_significance >= float(config.motion_sigma))

    nslices = cube.shape[1] if axis == "x" else cube.shape[2]
    support_spikes = 0
    seed_spikes = 0
    tracks_fitted = 0
    validated_support_spikes = 0
    tracks_spectral_validated = 0
    tracks_rejected_neighbor = 0
    tracks_rejected_span = 0
    neighbor_track_comparisons = 0
    tracks_accepted = 0
    slices_with_tracks = 0
    resolved_workers = resolve_cr_workers(config.workers, nslices)
    workers_used = resolved_workers
    parallel_fallbacks = 0
    executor: Optional[ProcessPoolExecutor] = None
    if resolved_workers > 1:
        try:
            executor = ProcessPoolExecutor(max_workers=resolved_workers)
        except OSError as exc:
            warnings.warn(
                f"Could not start CR worker pool ({exc}); using serial track detection.",
                RuntimeWarning,
                stacklevel=2,
            )
            workers_used = 1
            parallel_fallbacks += 1

    try:
        try:
            provisional_results = _run_slice_detection(
                spatial_significance,
                good_input,
                axis,
                nslices,
                config,
                executor,
            )
        except BrokenProcessPool as exc:
            warnings.warn(
                f"CR worker pool failed ({exc}); retrying track detection serially.",
                RuntimeWarning,
                stacklevel=2,
            )
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
            workers_used = 1
            parallel_fallbacks += 1
            provisional_results = _run_slice_detection(
                spatial_significance,
                good_input,
                axis,
                nslices,
                config,
                None,
            )

        provisional_tracks_by_slice: List[List[_Track]] = []
        for spikes, tracks in provisional_results:
            support_spikes += len(spikes)
            seed_spikes += sum(_is_seed(spike, config.sigma) for spike in spikes)
            tracks_fitted += len(tracks)
            provisional_tracks_by_slice.append(tracks)

        provisional_mask = np.zeros(cube.shape, dtype=bool)
        for slice_index, tracks in enumerate(provisional_tracks_by_slice):
            significance_panel = _panel(spatial_significance, axis, slice_index)
            panel_mask = np.zeros(significance_panel.shape, dtype=bool)
            for track in tracks:
                panel_mask |= _grow_track_mask(significance_panel, track, config=config)
            if np.any(panel_mask):
                _set_panel(provisional_mask, axis, slice_index, panel_mask)

        validation_model = _spectral_interpolation_model(
            cube,
            provisional_mask,
            median_spectral_model,
            sideband=max(spectral_window // 2, 1),
        )
        spectral_residual = cube.astype(np.float32, copy=False) - validation_model
        spectral_significance = np.full(cube.shape, -np.inf, dtype=np.float32)
        np.divide(
            spectral_residual,
            noise,
            out=spectral_significance,
            where=provisional_mask & good_input & (noise > 0),
        )
        validated_significance = np.minimum(spatial_significance, spectral_significance)
        validated_good = good_input & provisional_mask

        try:
            validated_results = _run_slice_detection(
                validated_significance,
                validated_good,
                axis,
                nslices,
                config,
                executor,
            )
        except BrokenProcessPool as exc:
            warnings.warn(
                f"CR worker pool failed ({exc}); retrying validation serially.",
                RuntimeWarning,
                stacklevel=2,
            )
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
            parallel_fallbacks += 1
            validated_results = _run_slice_detection(
                validated_significance,
                validated_good,
                axis,
                nslices,
                config,
                None,
            )

        tracks_by_slice: List[List[_Track]] = []
        for spikes, tracks in validated_results:
            validated_support_spikes += len(spikes)
            tracks_spectral_validated += len(tracks)
            tracks_by_slice.append(tracks)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    track_mask = np.zeros(cube.shape, dtype=bool)
    accepted_track_spans: List[int] = []
    for slice_index, tracks in enumerate(tracks_by_slice):
        significance_panel = _panel(validated_significance, axis, slice_index)
        panel_mask = np.zeros(significance_panel.shape, dtype=bool)
        accepted_here = 0
        for track in tracks:
            track_span = track.spikes[-1].row - track.spikes[0].row + 1
            if int(config.max_track_span) > 0 and track_span > int(config.max_track_span):
                tracks_rejected_span += 1
                continue
            coherence, comparisons = _neighbor_track_coherence(
                tracks_by_slice,
                slice_index=slice_index,
                track=track,
                tolerance=float(config.track_fit_tolerance),
            )
            neighbor_track_comparisons += comparisons
            if coherence >= float(config.neighbor_veto_fraction):
                tracks_rejected_neighbor += 1
                continue
            panel_mask |= _grow_track_mask(significance_panel, track, config=config)
            tracks_accepted += 1
            accepted_here += 1
            accepted_track_spans.append(track_span)
        if accepted_here:
            slices_with_tracks += 1
            _set_panel(track_mask, axis, slice_index, panel_mask)

    candidates = track_mask
    before_persistence = int(np.count_nonzero(candidates))
    if not config.require_slice_motion:
        assert core is not None and loose is not None
        persistent = _neighbor_count_along_wavelength(loose, radius=2)
        static_candidates = core & (persistent <= int(config.max_spectral_neighbors))
        static_candidates = _keep_compact_spatial_components(
            static_candidates,
            config.max_spatial_footprint,
        )
        candidates |= static_candidates
    after_persistence = int(np.count_nonzero(candidates))
    after_compactness = after_persistence

    cleaned = _replace_masked_spectrally(
        cube,
        candidates,
        median_spectral_model,
        sideband=max(spectral_window // 2, 1),
    )
    cleaned_uncert = _propagate_replacement_uncertainty(
        cube,
        uncert,
        noise,
        candidates,
        median_spectral_model,
        sideband=max(spectral_window // 2, 1),
    )
    n_flagged = int(np.count_nonzero(candidates))
    denom = int(np.count_nonzero(finite))
    fraction = float(n_flagged / denom) if denom else 0.0
    return CosmicRayRejectionResult(
        cleaned_cube=cleaned,
        cleaned_uncert=cleaned_uncert,
        mask=candidates,
        n_flagged=n_flagged,
        fraction_flagged=fraction,
        config=config,
        diagnostics={
            "workers_requested": int(config.workers),
            "workers_resolved": int(resolved_workers),
            "workers_used": int(workers_used),
            "parallel_fallbacks": int(parallel_fallbacks),
            "support_spikes": int(support_spikes),
            "seed_spikes": int(seed_spikes),
            "tracks_fitted": int(tracks_fitted),
            "provisional_mask_voxels": int(np.count_nonzero(provisional_mask)),
            "validated_support_spikes": int(validated_support_spikes),
            "tracks_spectral_validated": int(tracks_spectral_validated),
            "neighbor_track_comparisons": int(neighbor_track_comparisons),
            "tracks_rejected_neighbor_coherence": int(tracks_rejected_neighbor),
            "tracks_rejected_max_span": int(tracks_rejected_span),
            "tracks_accepted": int(tracks_accepted),
            "slices_with_tracks": int(slices_with_tracks),
            "median_accepted_track_span": (
                int(np.median(accepted_track_spans)) if accepted_track_spans else 0
            ),
            "max_accepted_track_span": max(accepted_track_spans, default=0),
            "track_mask_voxels": int(before_persistence),
            "final_mask_voxels": int(after_compactness),
            "uncertainty_voxels_updated": (
                int(n_flagged) if cleaned_uncert is not None else 0
            ),
            "spatial_background_window": int(line_window),
        },
        runtime_seconds=perf_counter() - started,
    )


def config_to_dict(config: CosmicRayRejectionConfig) -> Dict[str, object]:
    return asdict(config)


def write_cr_cleaned_fits(
    source_path: Path,
    out_path: Path,
    result: CosmicRayRejectionResult,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(source_path, memmap=False) as hdul:
        new_hdus = fits.HDUList([hdu.copy() for hdu in hdul])
        replaced = False
        science_header = None
        for hdu in new_hdus:
            data = getattr(hdu, "data", None)
            if data is not None and getattr(data, "shape", None) == result.cleaned_cube.shape:
                hdu.data = result.cleaned_cube.astype(np.float32, copy=False)
                hdr = hdu.header
                hdr["CRREJ"] = (True, "Cosmic-ray rejection applied")
                hdr["CRNPIX"] = (result.n_flagged, "Number of CR-cleaned voxels")
                hdr["CRFRAC"] = (result.fraction_flagged, "Fraction of finite voxels cleaned")
                hdr["CRMETH"] = ("KCWI_DUAL", "Spatial+spectral CR track rejection")
                hdr["CRSIG"] = (result.config.sigma, "CR track seed sigma threshold")
                hdr["CRMSIG"] = (result.config.motion_sigma, "CR track support sigma threshold")
                hdr["CRGSIG"] = (result.config.grow_sigma, "CR track mask-growth sigma")
                hdr["CRREQM"] = (result.config.require_slice_motion, "Require net motion along track")
                hdr["CRMINSH"] = (result.config.min_slice_shift_pixels, "Minimum fitted track displacement")
                hdr["CRMAXSH"] = (result.config.slice_shift_pixels, "Maximum track step per wave pixel")
                hdr["CRAXIS"] = (result.config.slice_motion_axis, "Within-slice motion axis")
                hdr["CRMINLEN"] = (result.config.min_track_length, "Minimum CR track length")
                hdr["CRMAXLEN"] = (result.config.max_track_span, "Maximum CR track span")
                hdr["CRGAP"] = (result.config.max_track_gap, "Maximum gap within CR track")
                hdr["CRNTRK"] = (result.diagnostics.get("tracks_accepted", 0), "Accepted CR tracks")
                hdr["CRWORK"] = (result.diagnostics.get("workers_used", 1), "CR worker processes used")
                hdr["CRTIME"] = (result.runtime_seconds, "CR algorithm runtime in seconds")
                hdr["CRUUPD"] = (
                    result.cleaned_uncert is not None,
                    "CR replacements propagated into uncertainty",
                )
                hdr.add_history("Spectrally validated CR tracks replaced along wavelength.")
                science_header = hdr
                replaced = True
                break
        if not replaced:
            raise ValueError(f"No HDU in {source_path} matches cleaned cube shape {result.cleaned_cube.shape}")

        if result.cleaned_uncert is not None:
            uncertainty_hdu = next(
                (
                    hdu
                    for hdu in new_hdus
                    if str(hdu.header.get("EXTNAME", "")).upper() == "UNCERT"
                ),
                None,
            )
            if uncertainty_hdu is None:
                uncertainty_hdu = fits.ImageHDU(name="UNCERT")
                new_hdus.append(uncertainty_hdu)
            uncertainty_hdu.data = result.cleaned_uncert.astype(np.float32, copy=False)
            uncertainty_hdu.header["CRUUPD"] = (
                True,
                "CR replacement uncertainty propagated",
            )
            uncertainty_hdu.header.add_history(
                "Uncertainty at CR-replaced voxels includes endpoint and local-model scatter."
            )
            if science_header is not None:
                science_header.add_history(
                    "UNCERT updated at CR-replaced voxels before aperture extraction."
                )

        new_hdus = fits.HDUList([hdu for hdu in new_hdus if str(hdu.name).upper() != "CR_MASK"])
        new_hdus.append(fits.ImageHDU(result.mask.astype(np.uint8), name="CR_MASK"))
        new_hdus.writeto(out_path, overwrite=True, checksum=True)


def write_cr_mask_fits(
    source_path: Path,
    out_path: Path,
    mask: np.ndarray,
    *,
    n_flagged: Optional[int] = None,
    fraction_flagged: Optional[float] = None,
    config: Optional[CosmicRayRejectionConfig] = None,
) -> None:
    """Write a standalone DS9-friendly CR mask cube.

    The output primary image uses 1 for CR-cleaned voxels and 0 elsewhere.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(source_path, memmap=False) as hdul:
        source_hdu = None
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None and getattr(data, "shape", None) == mask.shape:
                source_hdu = hdu
                break
        if source_hdu is None:
            raise ValueError(f"No HDU in {source_path} matches CR mask shape {mask.shape}")

        header = source_hdu.header.copy()

    header["CRMASK"] = (True, "Standalone cosmic-ray mask cube")
    if n_flagged is not None:
        header["CRNPIX"] = (int(n_flagged), "Number of CR-cleaned voxels")
    if fraction_flagged is not None:
        header["CRFRAC"] = (float(fraction_flagged), "Fraction of finite voxels cleaned")
    if config is not None:
        header["CRMETH"] = ("KCWI_DUAL", "Spatial+spectral CR track rejection")
        header["CRSIG"] = (config.sigma, "CR track seed sigma threshold")
        header["CRMSIG"] = (config.motion_sigma, "CR track support sigma threshold")
        header["CRGSIG"] = (config.grow_sigma, "CR track mask-growth sigma")
        header["CRREQM"] = (config.require_slice_motion, "Require net motion along track")
        header["CRMINSH"] = (config.min_slice_shift_pixels, "Minimum fitted track displacement")
        header["CRMAXSH"] = (config.slice_shift_pixels, "Maximum track step per wave pixel")
        header["CRAXIS"] = (config.slice_motion_axis, "Within-slice motion axis")
        header["CRMINLEN"] = (config.min_track_length, "Minimum CR track length")
        header["CRMAXLEN"] = (config.max_track_span, "Maximum CR track span")
        header["CRGAP"] = (config.max_track_gap, "Maximum gap within CR track")
    header["BUNIT"] = ("mask", "1=cosmic-ray voxel, 0=not flagged")
    header.add_history("Standalone CR mask written for DS9 inspection.")

    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8, copy=False), header=header)
    hdu.writeto(out_path, overwrite=True, checksum=True)


def plot_cr_diagnostic(mask: np.ndarray, out_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_wave = np.count_nonzero(mask, axis=(1, 2))
    spatial = np.count_nonzero(mask, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(mask.shape[0]), per_wave, lw=0.9, color="tab:red")
    axes[0].set_xlabel("Wavelength pixel")
    axes[0].set_ylabel("Flagged voxels")
    axes[0].grid(alpha=0.2)
    im = axes[1].imshow(spatial, origin="lower", interpolation="nearest", cmap="magma")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Collapsed CR mask")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
