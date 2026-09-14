from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from .uncertainty import linear_resample_with_uncertainty
from .utils import safe_filename, savefig_show

FLUX_UNIT_LABEL = "1e-15 erg/s/cm^2/A"


def truncate_spectrum(
    lam: np.ndarray,
    y: np.ndarray,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate a 1D spectrum to [lam_min, lam_max] (inclusive)."""
    lam = np.asarray(lam).ravel()
    y = np.asarray(y).ravel()
    if lam.shape[0] != y.shape[0]:
        raise ValueError("truncate_spectrum: lam and y must have the same length")

    if lam_min is None:
        lam_min = float(np.nanmin(lam))
    if lam_max is None:
        lam_max = float(np.nanmax(lam))

    m = np.isfinite(lam) & np.isfinite(y) & (lam >= lam_min) & (lam <= lam_max)
    if not np.any(m):
        raise ValueError(f"truncate_spectrum: no finite samples in range [{lam_min}, {lam_max}] Å")
    return lam[m], y[m]


def load_reference_flux(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load reference flux spectrum: columns [wavelength_A, flux]."""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError(f"Reference flux file needs >=2 columns: {path}")
    lam = arr[:, 0].astype(float)
    F = arr[:, 1].astype(float)
    return lam, F


def build_sensitivity(
    lam_ref: np.ndarray,
    F_ref: np.ndarray,
    lam_std: np.ndarray,
    C_std: np.ndarray,
    spline_s: float,
    exclude_windows: Optional[List[Tuple[float, float]]] = None,
    *,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a sensitivity curve S(λ) on a *truncated* reference wavelength grid.

    Returns:
        lam_ref_t, F_ref_t, ratio_raw, S_t

    Notes:
        The reference spectrum often spans 1000–30000+ Å; we truncate it to the arm's
        [lam_min, lam_max] before interpolation and spline fitting. Downstream plots should
        use lam_ref_t for cleaner diagnostics and faster runs.
    """
    lam_ref_t, F_ref_t = truncate_spectrum(lam_ref, F_ref, lam_min=lam_min, lam_max=lam_max)

    # interpolate extracted standard counts onto reference wavelength grid
    C_interp = np.interp(lam_ref_t, lam_std, C_std)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = F_ref_t / C_interp

    mask = np.isfinite(ratio) & (ratio > 0)
    if exclude_windows:
        for lo, hi in exclude_windows:
            mask &= ~((lam_ref_t >= lo) & (lam_ref_t <= hi))

    if np.count_nonzero(mask) < 10:
        raise ValueError("build_sensitivity: too few good points after masking/exclusions to fit spline")

    spline = UnivariateSpline(lam_ref_t[mask], ratio[mask], s=spline_s)
    S_t = spline(lam_ref_t)
    return lam_ref_t, F_ref_t, ratio, S_t


def apply_sensitivity(
    lam_ref: np.ndarray,
    S_sens: np.ndarray,
    lam_obj: np.ndarray,
    C_obj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sensitivity curve (defined on lam_ref grid) to an object counts spectrum."""
    C_int, _ = linear_resample_with_uncertainty(lam_ref, lam_obj, C_obj)
    F_obj = S_sens * C_int
    return lam_ref, F_obj


def apply_sensitivity_with_uncertainty(
    lam_ref: np.ndarray,
    S_sens: np.ndarray,
    lam_obj: np.ndarray,
    C_obj: np.ndarray,
    sigma_obj: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply sensitivity and propagate resampling uncertainty onto its grid."""
    C_int, sigma_int = linear_resample_with_uncertainty(
        lam_ref,
        lam_obj,
        C_obj,
        sigma_obj,
    )
    F_obj = S_sens * C_int
    sigma_flux = np.abs(S_sens) * sigma_int if sigma_int is not None else None
    return np.asarray(lam_ref), F_obj, sigma_flux


def build_o2_transmission_template(
    lam_ref: np.ndarray,
    F_ref: np.ndarray,
    lam_std: np.ndarray,
    C_std: np.ndarray,
    S_sens: np.ndarray,
    o2_windows: List[Tuple[float, float]],
    min_T: float = 0.02,
    smooth_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build O2 transmission template from the standard.

    IMPORTANT: lam_ref/F_ref/S_sens should already be truncated to the RED arm range.
    """
    C_on_ref = np.interp(lam_ref, lam_std, C_std)
    F_std_cal = S_sens * C_on_ref

    with np.errstate(divide="ignore", invalid="ignore"):
        T = F_std_cal / F_ref

    good = np.isfinite(T) & np.isfinite(lam_ref) & (F_ref > 0) & (T > 0)

    o2_mask = np.zeros_like(lam_ref, dtype=bool)
    for lo, hi in o2_windows:
        o2_mask |= (lam_ref >= lo) & (lam_ref <= hi)

    Tin = T.copy()
    Tin[~(good & o2_mask)] = np.nan

    if smooth_s is not None:
        m = np.isfinite(Tin)
        if np.any(m):
            spl = UnivariateSpline(lam_ref[m], Tin[m], s=smooth_s)
            Tin = spl(lam_ref)

    Tin = np.clip(Tin, min_T, 1.0)

    T_out = np.ones_like(lam_ref, dtype=float)
    T_out[o2_mask] = Tin[o2_mask]
    return T_out, o2_mask


def build_standard_telluric_template(
    lam_std: np.ndarray,
    C_std: np.ndarray,
    continuum_std: np.ndarray,
    telluric_windows: List[Tuple[float, float]],
    min_T: float = 0.02,
    smooth_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a normalized telluric transmission template from the flux standard.

    This follows the UCSC/pydux idea: use the spectrophotometric standard as the
    hot-star telluric source, divide by the fitted stellar continuum, and set
    wavelengths outside telluric bands to unity.
    """
    lam_std = np.asarray(lam_std, dtype=float).ravel()
    C_std = np.asarray(C_std, dtype=float).ravel()
    continuum_std = np.asarray(continuum_std, dtype=float).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        T = C_std / continuum_std

    good = np.isfinite(lam_std) & np.isfinite(T) & np.isfinite(continuum_std) & (continuum_std > 0) & (T > 0)

    tell_mask = np.zeros_like(lam_std, dtype=bool)
    for lo, hi in telluric_windows:
        tell_mask |= (lam_std >= lo) & (lam_std <= hi)

    Tin = T.copy()
    Tin[~(good & tell_mask)] = np.nan

    if smooth_s is not None:
        m = np.isfinite(Tin)
        if np.count_nonzero(m) >= 4:
            spl = UnivariateSpline(lam_std[m], Tin[m], s=smooth_s)
            Tin = spl(lam_std)

    Tin = np.clip(Tin, min_T, 1.0)
    T_out = np.ones_like(lam_std, dtype=float)
    T_out[tell_mask] = Tin[tell_mask]
    return T_out, tell_mask


def shifted_transmission(
    lam_ref: np.ndarray,
    T_std: np.ndarray,
    shift_A: float,
) -> np.ndarray:
    """Evaluate a template shifted by ``shift_A`` Angstrom onto its own grid."""
    return np.interp(lam_ref - float(shift_A), lam_ref, T_std, left=1.0, right=1.0)


def estimate_telluric_shift(
    lam_ref: np.ndarray,
    F_fluxcal: np.ndarray,
    T_std: np.ndarray,
    telluric_windows: List[Tuple[float, float]],
    max_shift_A: float = 5.0,
    step_A: float = 0.1,
) -> float:
    """Estimate the wavelength shift that best aligns telluric absorption bands."""
    lam_ref = np.asarray(lam_ref, dtype=float).ravel()
    F_fluxcal = np.asarray(F_fluxcal, dtype=float).ravel()
    T_std = np.asarray(T_std, dtype=float).ravel()

    mask = np.zeros_like(lam_ref, dtype=bool)
    for lo, hi in telluric_windows:
        mask |= (lam_ref >= lo) & (lam_ref <= hi)
    mask &= np.isfinite(lam_ref) & np.isfinite(F_fluxcal) & np.isfinite(T_std) & (T_std < 0.98)

    if np.count_nonzero(mask) < 8:
        return 0.0

    wave = lam_ref[mask]
    flux = F_fluxcal[mask]
    med = np.nanmedian(flux)
    if not np.isfinite(med) or med == 0:
        return 0.0

    flux_norm = flux / med
    flux_depth = 1.0 - flux_norm
    flux_depth -= np.nanmedian(flux_depth)
    flux_std = np.nanstd(flux_depth)
    if not np.isfinite(flux_std) or flux_std <= 0:
        return 0.0

    shifts = np.arange(-abs(max_shift_A), abs(max_shift_A) + 0.5 * step_A, abs(step_A))
    best_shift = 0.0
    best_score = -np.inf
    for shift in shifts:
        T_shift = np.interp(wave - shift, lam_ref, T_std, left=1.0, right=1.0)
        temp_depth = 1.0 - T_shift
        temp_depth -= np.nanmedian(temp_depth)
        temp_std = np.nanstd(temp_depth)
        if not np.isfinite(temp_std) or temp_std <= 0:
            continue
        score = float(np.nansum((flux_depth / flux_std) * (temp_depth / temp_std)))
        if score > best_score:
            best_score = score
            best_shift = float(shift)
    return best_shift


def apply_o2_telluric_correction(
    F_fluxcal: np.ndarray,
    T_std: np.ndarray,
    X_std: Optional[float],
    X_sci: Optional[float],
    o2_mask: np.ndarray,
    min_T: float = 0.02,
) -> np.ndarray:
    """Apply airmass-scaled O2 correction to a flux-calibrated spectrum."""
    if X_std is None or X_sci is None:
        return F_fluxcal

    p = float(X_sci) / float(X_std)
    T_scaled = np.ones_like(T_std, dtype=float)
    T_scaled[o2_mask] = np.clip(T_std[o2_mask], min_T, 1.0) ** p

    F_corr = F_fluxcal.copy()
    F_corr[o2_mask] = F_fluxcal[o2_mask] / np.clip(T_scaled[o2_mask], min_T, 1.0)
    return F_corr


def scaled_o2_transmission(
    T_std: np.ndarray,
    X_std: Optional[float],
    X_sci: Optional[float],
    o2_mask: np.ndarray,
    min_T: float = 0.02,
    airmass_exponent: float = 1.0,
) -> np.ndarray:
    """Return the transmission curve actually used for an airmass-scaled O2 correction."""
    T_scaled = np.ones_like(T_std, dtype=float)
    if X_std is None or X_sci is None:
        return T_scaled
    p = (float(X_sci) / float(X_std)) ** float(airmass_exponent)
    T_scaled[o2_mask] = np.clip(T_std[o2_mask], min_T, 1.0) ** p
    return T_scaled


def apply_standard_telluric_correction(
    F_fluxcal: np.ndarray,
    T_std: np.ndarray,
    X_std: Optional[float],
    X_sci: Optional[float],
    telluric_mask: np.ndarray,
    min_T: float = 0.02,
    airmass_exponent: float = 0.55,
) -> np.ndarray:
    """Apply UCSC-style airmass-scaled telluric correction."""
    if X_std is None or X_sci is None:
        return F_fluxcal

    T_scaled = scaled_o2_transmission(
        T_std,
        X_std,
        X_sci,
        telluric_mask,
        min_T=min_T,
        airmass_exponent=airmass_exponent,
    )
    F_corr = F_fluxcal.copy()
    F_corr[telluric_mask] = F_fluxcal[telluric_mask] / np.clip(T_scaled[telluric_mask], min_T, 1.0)
    return F_corr


# ----------------------------
# Diagnostics
# ----------------------------

def plot_calibration_diagnostics(
    side: str,
    std_name: str,
    lam_std: np.ndarray,
    C_std: np.ndarray,
    lam_ref: np.ndarray,
    F_ref: np.ndarray,
    ratio: np.ndarray,
    S: np.ndarray,
    F_std_cal: np.ndarray,
    outdir: Path,
    show: bool,
    telluric_windows: Optional[List[Tuple[float, float]]] = None,
    red_tell_before: Optional[np.ndarray] = None,
    red_tell_after: Optional[np.ndarray] = None,
    *,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
) -> None:
    """Diagnostic plots restricted to the (possibly truncated) lam_ref grid."""
    side_u = side.upper()
    tag = f"{safe_filename(std_name)}_{side_u}"

    if lam_min is None:
        lam_min = float(np.nanmin(lam_ref))
    if lam_max is None:
        lam_max = float(np.nanmax(lam_ref))

    # 1) counts
    plt.figure(figsize=(10, 4))
    plt.plot(lam_std, C_std, lw=1)
    plt.xlim(lam_min, lam_max)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Counts (tgt - bkg*area)")
    plt.title(f"{std_name} {side_u}: extracted counts")
    savefig_show(outdir / f"{tag}_counts.png", show)

    # 2) ratio + sensitivity
    plt.figure(figsize=(10, 4))
    m = np.isfinite(ratio) & (ratio > 0)
    plt.plot(lam_ref[m], ratio[m], lw=0.8, alpha=0.6, label="Raw ratio F_ref / C_std")
    plt.plot(lam_ref, S, lw=1.5, label="Sensitivity S(λ) (spline)")
    plt.yscale("log")
    plt.xlim(lam_min, lam_max)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Sensitivity")
    plt.title(f"{std_name} {side_u}: sensitivity diagnostic")
    plt.legend()
    savefig_show(outdir / f"{tag}_sensitivity.png", show)

    # 3) calibrated standard vs reference
    plt.figure(figsize=(10, 4))
    plt.plot(lam_ref, F_ref, lw=1.5, label="Reference flux (truncated)")
    plt.plot(lam_ref, F_std_cal, lw=1.0, alpha=0.85, label="Calibrated standard")
    plt.xlim(lam_min, lam_max)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel(f"Flux ({FLUX_UNIT_LABEL})")
    plt.title(f"{std_name} {side_u}: calibrated standard vs reference")
    plt.legend()
    savefig_show(outdir / f"{tag}_stdcheck.png", show)

    # 4) telluric before/after (if provided)
    if side_u == "RED" and red_tell_before is not None and red_tell_after is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(lam_ref, red_tell_before, lw=1.0, alpha=0.7, label="Before telluric corr")
        plt.plot(lam_ref, red_tell_after, lw=1.0, alpha=0.9, label="After telluric corr")
        if telluric_windows:
            for (lo, hi) in telluric_windows:
                lo2 = max(lo, lam_min)
                hi2 = min(hi, lam_max)
                if lo2 < hi2:
                    plt.axvspan(lo2, hi2, alpha=0.2)
        plt.xlim(lam_min, lam_max)
        plt.xlabel("Wavelength (Å)")
        plt.ylabel(f"Flux ({FLUX_UNIT_LABEL})")
        plt.title(f"{std_name} RED: telluric correction")
        plt.legend()
        savefig_show(outdir / f"{tag}_telluric.png", show)


def plot_o2_template_diagnostic(
    lam_ref: np.ndarray,
    T_std: np.ndarray,
    o2_windows: List[Tuple[float, float]],
    outpng: Path,
    show: bool,
) -> None:
    plt.figure(figsize=(10, 3.5))
    plt.plot(lam_ref, T_std, lw=1.0)
    lo_lim = float(np.nanmin(lam_ref))
    hi_lim = float(np.nanmax(lam_ref))
    for lo, hi in o2_windows:
        lo2 = max(lo, lo_lim)
        hi2 = min(hi, hi_lim)
        if lo2 < hi2:
            plt.axvspan(lo2, hi2, alpha=0.2)
    plt.ylim(0, 1.05)
    plt.xlim(lo_lim, hi_lim)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Transmission")
    plt.title("Telluric template from spectrophotometric standard")
    savefig_show(outpng, show)


def plot_o2_before_after(
    objname: str,
    lam_ref: np.ndarray,
    F_before: np.ndarray,
    F_after: np.ndarray,
    o2_windows: List[Tuple[float, float]],
    outpng: Path,
    show: bool,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(lam_ref, F_before, lw=1.0, alpha=0.7, label="Before telluric corr")
    plt.plot(lam_ref, F_after, lw=1.0, alpha=0.9, label="After telluric corr")
    lo_lim = float(np.nanmin(lam_ref))
    hi_lim = float(np.nanmax(lam_ref))
    for lo, hi in o2_windows:
        lo2 = max(lo, lo_lim)
        hi2 = min(hi, hi_lim)
        if lo2 < hi2:
            plt.axvspan(lo2, hi2, alpha=0.2)
    plt.xlim(lo_lim, hi_lim)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel(f"Flux ({FLUX_UNIT_LABEL})")
    plt.title(f"{objname} RED: telluric correction")
    plt.legend()
    savefig_show(outpng, show)


def plot_o2_correction_diagnostic(
    objname: str,
    lam_ref: np.ndarray,
    F_before: np.ndarray,
    F_after: np.ndarray,
    T_std: np.ndarray,
    T_scaled: np.ndarray,
    o2_mask: np.ndarray,
    o2_windows: List[Tuple[float, float]],
    outpng: Path,
    show: bool,
) -> None:
    """Detailed RED telluric diagnostic with full spectrum plus zoomed correction windows."""
    nwin = max(1, len(o2_windows))
    fig, axes = plt.subplots(2, nwin + 1, figsize=(4.2 * (nwin + 1), 7), constrained_layout=True)
    if nwin == 1:
        axes = np.asarray(axes).reshape(2, 2)
    ax_full = axes[0, 0]
    ax_trans = axes[1, 0]
    zoom_axes = axes[:, 1:].ravel()

    ax_full.plot(lam_ref, F_before, lw=0.9, alpha=0.65, label="Before telluric corr")
    ax_full.plot(lam_ref, F_after, lw=0.9, alpha=0.9, label="After telluric corr")
    for lo, hi in o2_windows:
        ax_full.axvspan(lo, hi, alpha=0.15)
    ax_full.set_xlabel("Wavelength (A)")
    ax_full.set_ylabel(f"Flux ({FLUX_UNIT_LABEL})")
    ax_full.set_title(f"{objname} RED: full spectrum")
    ax_full.legend()

    ax_trans.plot(lam_ref, T_std, lw=0.9, label="Standard transmission")
    ax_trans.plot(lam_ref, T_scaled, lw=0.9, label="Airmass-scaled transmission")
    ax_trans.plot(lam_ref, 1.0 / np.clip(T_scaled, 0.02, None), lw=0.9, alpha=0.8, label="Applied correction factor")
    for lo, hi in o2_windows:
        ax_trans.axvspan(lo, hi, alpha=0.15)
    ax_trans.set_xlabel("Wavelength (A)")
    ax_trans.set_ylabel("Transmission / factor")
    ax_trans.set_ylim(0, max(1.2, float(np.nanpercentile(1.0 / np.clip(T_scaled[o2_mask], 0.02, None), 98)) * 1.1) if np.any(o2_mask) else 1.2)
    ax_trans.set_title("Telluric model used")
    ax_trans.legend(fontsize=8)

    for ax, (lo, hi) in zip(zoom_axes, o2_windows):
        m = np.isfinite(lam_ref) & (lam_ref >= lo) & (lam_ref <= hi)
        ax.plot(lam_ref[m], F_before[m], lw=0.9, alpha=0.65, label="Before")
        ax.plot(lam_ref[m], F_after[m], lw=0.9, alpha=0.9, label="After")
        ax_t = ax.twinx()
        ax_t.plot(lam_ref[m], T_scaled[m], lw=0.8, color="tab:green", alpha=0.85, label="T scaled")
        ax.set_xlim(lo, hi)
        ax_t.set_ylim(0, 1.05)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel(f"Flux ({FLUX_UNIT_LABEL})")
        ax_t.set_ylabel("T scaled")
        ax.set_title(f"Telluric window {lo:.0f}-{hi:.0f} A")

    for ax in zoom_axes[len(o2_windows):]:
        ax.axis("off")

    outpng = Path(outpng)
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
