from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def linear_resample_with_uncertainty(
    wavelength_out: np.ndarray,
    wavelength_in: np.ndarray,
    values_in: np.ndarray,
    sigma_in: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Linearly resample values and propagate independent input variances.

    For an output point between two input samples, ``y = (1-f)y0 + f*y1``
    and ``var(y) = (1-f)^2 var0 + f^2 var1``. Samples outside the input
    wavelength range are returned as NaN rather than extrapolated.
    """
    wavelength_out = np.asarray(wavelength_out, dtype=float).ravel()
    wavelength_in = np.asarray(wavelength_in, dtype=float).ravel()
    values_in = np.asarray(values_in, dtype=float).ravel()
    if wavelength_in.shape != values_in.shape:
        raise ValueError("Input wavelength and value arrays must have matching shapes")

    sigma_array: Optional[np.ndarray]
    if sigma_in is None:
        sigma_array = None
    else:
        sigma_array = np.asarray(sigma_in, dtype=float).ravel()
        if sigma_array.shape != wavelength_in.shape:
            raise ValueError("Input wavelength and uncertainty arrays must have matching shapes")

    valid = np.isfinite(wavelength_in) & np.isfinite(values_in)
    if not np.any(valid):
        empty = np.full(wavelength_out.shape, np.nan, dtype=float)
        return empty, empty.copy() if sigma_array is not None else None

    order = np.argsort(wavelength_in[valid], kind="stable")
    x = wavelength_in[valid][order]
    y = values_in[valid][order]
    sig = sigma_array[valid][order] if sigma_array is not None else None

    # Duplicate wavelength samples do not define a unique interpolation segment.
    # Average their values and, when available, inverse-variance combine them.
    unique_x, starts, counts = np.unique(x, return_index=True, return_counts=True)
    if np.any(counts > 1):
        unique_y = np.full(unique_x.shape, np.nan, dtype=float)
        unique_sig = np.full(unique_x.shape, np.nan, dtype=float) if sig is not None else None
        for i, (start, count) in enumerate(zip(starts, counts)):
            section = slice(start, start + count)
            if sig is not None:
                good_sigma = np.isfinite(sig[section]) & (sig[section] > 0)
                if np.any(good_sigma):
                    weights = 1.0 / sig[section][good_sigma] ** 2
                    unique_y[i] = np.sum(y[section][good_sigma] * weights) / np.sum(weights)
                    unique_sig[i] = np.sqrt(1.0 / np.sum(weights))
                    continue
            unique_y[i] = np.mean(y[section])
        x = unique_x
        y = unique_y
        sig = unique_sig

    values_out = np.full(wavelength_out.shape, np.nan, dtype=float)
    sigma_out = np.full(wavelength_out.shape, np.nan, dtype=float) if sig is not None else None
    finite_out = np.isfinite(wavelength_out)
    if x.size == 1:
        exact = finite_out & (wavelength_out == x[0])
        values_out[exact] = y[0]
        if sigma_out is not None and sig is not None:
            sigma_out[exact] = np.abs(sig[0])
        return values_out, sigma_out

    in_range = finite_out & (wavelength_out >= x[0]) & (wavelength_out <= x[-1])
    if not np.any(in_range):
        return values_out, sigma_out

    output_indices = np.flatnonzero(in_range)
    xout = wavelength_out[output_indices]
    right = np.searchsorted(x, xout, side="left")
    exact_last = right == x.size
    right[exact_last] = x.size - 1
    exact = x[right] == xout

    if np.any(exact):
        out_exact = output_indices[exact]
        source = right[exact]
        values_out[out_exact] = y[source]
        if sigma_out is not None and sig is not None:
            source_sigma = sig[source]
            good = np.isfinite(source_sigma) & (source_sigma >= 0)
            sigma_out[out_exact[good]] = np.abs(source_sigma[good])

    between = ~exact
    if np.any(between):
        out_between = output_indices[between]
        right_between = right[between]
        left_between = right_between - 1
        denominator = x[right_between] - x[left_between]
        fraction = (wavelength_out[out_between] - x[left_between]) / denominator
        values_out[out_between] = (
            (1.0 - fraction) * y[left_between] + fraction * y[right_between]
        )
        if sigma_out is not None and sig is not None:
            sigma_left = np.abs(sig[left_between])
            sigma_right = np.abs(sig[right_between])
            good = (
                np.isfinite(sigma_left)
                & np.isfinite(sigma_right)
                & (sigma_left >= 0)
                & (sigma_right >= 0)
            )
            sigma_out[out_between[good]] = np.sqrt(
                ((1.0 - fraction[good]) * sigma_left[good]) ** 2
                + (fraction[good] * sigma_right[good]) ** 2
            )

    return values_out, sigma_out
