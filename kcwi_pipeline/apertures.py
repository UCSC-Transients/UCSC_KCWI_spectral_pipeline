from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.wcs import WCS
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.widgets import Button, RadioButtons, RangeSlider, Slider, TextBox
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    EllipticalAnnulus,
    EllipticalAperture,
    RectangularAperture,
)

from .config import ApertureShape, TargetBackgroundApertures
from .utils import prompt


class WhiteLightRangeController:
    """Fast wavelength-range sums for interactive white-light displays."""

    def __init__(
        self,
        cube: np.ndarray,
        wavelength: np.ndarray,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> None:
        cube = np.asarray(cube)
        wavelength = np.asarray(wavelength, dtype=float)
        if cube.ndim != 3 or wavelength.ndim != 1 or cube.shape[0] != wavelength.size:
            raise ValueError(
                "White-light wavelength controller requires cube (wavelength, y, x) "
                "and a matching 1D wavelength array"
            )
        finite_wavelength = np.isfinite(wavelength)
        if not np.any(finite_wavelength):
            raise ValueError("White-light wavelength array has no finite values")

        available_min = float(np.min(wavelength[finite_wavelength]))
        available_max = float(np.max(wavelength[finite_wavelength]))
        requested_min = available_min if minimum is None else float(minimum)
        requested_max = available_max if maximum is None else float(maximum)
        allowed = finite_wavelength & (wavelength >= requested_min) & (wavelength <= requested_max)
        if not np.any(allowed):
            allowed = finite_wavelength
        self.slider_values = np.unique(np.sort(wavelength[allowed]))
        self.available_min = float(self.slider_values[0])
        self.available_max = float(self.slider_values[-1])
        self.initial_min = self.available_min
        self.initial_max = self.available_max
        self.minimum = self.initial_min
        self.maximum = self.initial_max
        self._cube = cube
        differences = np.diff(wavelength[finite_wavelength])
        self._monotonic = bool(
            np.all(differences >= 0) or np.all(differences <= 0)
        )
        self._wavelength = wavelength
        self._prefix = np.empty(
            (cube.shape[0] + 1, cube.shape[1], cube.shape[2]),
            dtype=np.float32,
        )
        self._prefix[0] = 0.0
        self._prefix[1:] = cube
        np.nan_to_num(
            self._prefix[1:],
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.cumsum(self._prefix[1:], axis=0, out=self._prefix[1:])

    @property
    def bounds(self) -> Tuple[float, float]:
        return self.minimum, self.maximum

    @property
    def initial_bounds(self) -> Tuple[float, float]:
        return self.initial_min, self.initial_max

    def set_bounds(self, minimum: float, maximum: float) -> None:
        minimum = float(np.clip(minimum, self.available_min, self.available_max))
        maximum = float(np.clip(maximum, self.available_min, self.available_max))
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        self.minimum = minimum
        self.maximum = maximum

    def image(self) -> np.ndarray:
        selected = np.flatnonzero(
            np.isfinite(self._wavelength)
            & (self._wavelength >= self.minimum)
            & (self._wavelength <= self.maximum)
        )
        if selected.size == 0:
            nearest = int(
                np.nanargmin(
                    np.abs(self._wavelength - 0.5 * (self.minimum + self.maximum))
                )
            )
            selected = np.array([nearest], dtype=int)
        if self._monotonic:
            return self._prefix[selected[-1] + 1] - self._prefix[selected[0]]
        finite_cube = np.where(np.isfinite(self._cube[selected]), self._cube[selected], 0.0)
        return np.sum(finite_cube, axis=0, dtype=np.float32)


def _rotated_coords(ny: int, nx: int, x0: float, y0: float, theta: float):
    """Return (xr, yr) = coordinates rotated by theta around (x0, y0)."""
    y, x = np.mgrid[0:ny, 0:nx]
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    return xr, yr


def mask_from_shape(ny: int, nx: int, shape: ApertureShape) -> np.ndarray:
    """Boolean mask for a given ApertureShape."""
    sh = shape.shape
    p = shape.params

    if sh == "ellipse":
        x0, y0, a, b, theta = p
        xr, yr = _rotated_coords(ny, nx, x0, y0, theta)
        return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

    if sh == "circle":
        x0, y0, r = p
        y, x = np.mgrid[0:ny, 0:nx]
        return (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2

    if sh == "rect":
        x0, y0, w, h, theta = p
        xr, yr = _rotated_coords(ny, nx, x0, y0, theta)
        return (np.abs(xr) <= (w / 2.0)) & (np.abs(yr) <= (h / 2.0))

    if sh == "ellipse_annulus":
        x0, y0, a_in, b_in, a_out, b_out, theta = p
        xr, yr = _rotated_coords(ny, nx, x0, y0, theta)
        inner = (xr / a_in) ** 2 + (yr / b_in) ** 2 <= 1.0
        outer = (xr / a_out) ** 2 + (yr / b_out) ** 2 <= 1.0
        return outer & (~inner)

    if sh == "circle_annulus":
        x0, y0, r_in, r_out = p
        y, x = np.mgrid[0:ny, 0:nx]
        rr2 = (x - x0) ** 2 + (y - y0) ** 2
        return (rr2 <= r_out ** 2) & (rr2 > r_in ** 2)

    raise ValueError(f"Unknown shape: {sh}")


def aperture_weight_mask(ny: int, nx: int, shape: ApertureShape) -> np.ndarray:
    """Fractional aperture mask using exact pixel/aperture intersection.

    Returns weights in [0, 1] with shape (ny, nx). A value of 0.25 means
    one quarter of that spaxel is covered by the aperture.
    """
    sh = shape.shape
    p = shape.params

    if sh == "circle":
        x0, y0, r = p
        aper = CircularAperture((x0, y0), r=r)
    elif sh == "circle_annulus":
        x0, y0, r_in, r_out = p
        aper = CircularAnnulus((x0, y0), r_in=r_in, r_out=r_out)
    elif sh == "ellipse":
        x0, y0, a, b, theta = p
        aper = EllipticalAperture((x0, y0), a=a, b=b, theta=theta)
    elif sh == "ellipse_annulus":
        x0, y0, a_in, b_in, a_out, b_out, theta = p
        aper = EllipticalAnnulus((x0, y0), a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out, theta=theta)
    elif sh == "rect":
        x0, y0, w, h, theta = p
        aper = RectangularAperture((x0, y0), w=w, h=h, theta=theta)
    else:
        raise ValueError(f"Unknown shape: {sh}")

    mask = aper.to_mask(method="exact")
    image = mask.to_image((ny, nx))
    if image is None:
        return np.zeros((ny, nx), dtype=float)
    return np.clip(np.asarray(image, dtype=float), 0.0, 1.0)


def _patch_for_shape(shape: ApertureShape, edgecolor: str, linestyle: str = "-", lw: float = 2.0):
    """Matplotlib patch for quick overlays."""
    sh = shape.shape
    p = shape.params

    if sh == "ellipse":
        x0, y0, a, b, theta = p
        return Ellipse((x0, y0), 2 * a, 2 * b, angle=np.degrees(theta), fill=False,
                       linewidth=lw, edgecolor=edgecolor, linestyle=linestyle)

    if sh == "circle":
        x0, y0, r = p
        return Circle((x0, y0), r, fill=False, linewidth=lw, edgecolor=edgecolor, linestyle=linestyle)

    if sh == "rect":
        x0, y0, w, h, theta = p
        # Rectangle expects bottom-left; we give center + angle
        return Rectangle((x0 - w / 2.0, y0 - h / 2.0), w, h, angle=np.degrees(theta),
                         fill=False, linewidth=lw, edgecolor=edgecolor, linestyle=linestyle)

    if sh == "ellipse_annulus":
        x0, y0, a_in, b_in, a_out, b_out, theta = p
        # Return both boundaries
        return (
            Ellipse((x0, y0), 2 * a_in, 2 * b_in, angle=np.degrees(theta), fill=False,
                    linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
            Ellipse((x0, y0), 2 * a_out, 2 * b_out, angle=np.degrees(theta), fill=False,
                    linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
        )

    if sh == "circle_annulus":
        x0, y0, r_in, r_out = p
        return (
            Circle((x0, y0), r_in, fill=False, linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
            Circle((x0, y0), r_out, fill=False, linewidth=1.2, edgecolor=edgecolor, linestyle="--"),
        )

    raise ValueError(f"Unknown shape: {sh}")


def _white_light_limits(img: np.ndarray) -> Tuple[float, float]:
    return tuple(float(x) for x in np.nanpercentile(img, [5, 99]))


def _add_shape_patch(ax, shape: ApertureShape, edgecolor: str, linestyle: str = "-", lw: float = 2.0) -> None:
    patch = _patch_for_shape(shape, edgecolor=edgecolor, linestyle=linestyle, lw=lw)
    if isinstance(patch, tuple):
        for item in patch:
            ax.add_patch(item)
    else:
        ax.add_patch(patch)


def _white_light_two_panel(
    img: np.ndarray,
    title: str,
    *,
    apertures: Optional[TargetBackgroundApertures] = None,
    shapes: Optional[Tuple[ApertureShape, ...]] = None,
    wavelength_controller: Optional[WhiteLightRangeController] = None,
    celestial_wcs: Optional[WCS] = None,
    right_margin: float = 0.90,
):
    """Create a two-panel white-light view.

    Left panel keeps the original aspect ratio and is the aperture-editing view.
    Right panel uses the same data coordinates but compresses the y display scale
    by a factor of 3 to make elongated sources easier to compare to sky charts.
    """
    if wavelength_controller is not None:
        img = wavelength_controller.image()
    v1, v2 = _white_light_limits(img)
    if right_margin <= 0.72:
        figure_width = 14.5
    elif right_margin < 0.85:
        figure_width = 13.5
    else:
        figure_width = 12.0
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(figure_width, 6.6 if wavelength_controller is not None else 5.8),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.subplots_adjust(
        left=0.07,
        right=right_margin,
        bottom=0.31 if wavelength_controller is not None else 0.18,
        top=0.88,
        wspace=0.25,
    )

    im = ax_left.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis", aspect="equal")
    ax_left.set_xlabel("x pixel")
    ax_left.set_ylabel("y pixel")

    im_right = ax_right.imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="viridis", aspect=(1.0 / 3.0))
    ax_right.set_xlabel("x pixel")
    ax_right.set_ylabel("y pixel")

    def update_panel_titles() -> None:
        if wavelength_controller is None:
            range_suffix = ""
        else:
            wave_min, wave_max = wavelength_controller.bounds
            range_suffix = f" ({wave_min:.1f}-{wave_max:.1f} A)"
        ax_left.set_title(f"White light{range_suffix}")
        ax_right.set_title(f"White light, y compressed x3{range_suffix}")

    update_panel_titles()

    if celestial_wcs is not None:
        coordinate_text = fig.text(
            0.025,
            0.52,
            "Cursor WCS\nRA   --\nDec  --",
            ha="left",
            va="center",
            family="monospace",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82},
        )

        def update_cursor_coordinates(event) -> None:
            if (
                event.inaxes not in (ax_left, ax_right)
                or event.xdata is None
                or event.ydata is None
            ):
                return
            try:
                sky = celestial_wcs.pixel_to_world(
                    float(event.xdata),
                    float(event.ydata),
                )
                ra = sky.ra.to_string(
                    unit=u.hourangle,
                    sep=":",
                    precision=2,
                    pad=True,
                )
                dec = sky.dec.to_string(
                    unit=u.deg,
                    sep=":",
                    precision=2,
                    pad=True,
                    alwayssign=True,
                )
                coordinate_text.set_text(f"Cursor WCS\nRA   {ra}\nDec  {dec}")
            except (AttributeError, TypeError, ValueError):
                coordinate_text.set_text("Cursor WCS\nRA   unavailable\nDec  unavailable")
            fig.canvas.draw_idle()

        coordinate_motion_cid = fig.canvas.mpl_connect(
            "motion_notify_event",
            update_cursor_coordinates,
        )
        fig._kcwi_coordinate_text = coordinate_text
        fig._kcwi_coordinate_motion_cid = coordinate_motion_cid

    if apertures is not None:
        _add_shape_patch(ax_left, apertures.target, edgecolor="red", lw=2.0)
        _add_shape_patch(ax_left, apertures.background, edgecolor="orange", lw=1.6)
        _add_shape_patch(ax_right, apertures.target, edgecolor="red", lw=2.0)
        _add_shape_patch(ax_right, apertures.background, edgecolor="orange", lw=1.6)
    if shapes is not None:
        for shape in shapes:
            _add_shape_patch(ax_left, shape, edgecolor="cyan", lw=1.8)
            _add_shape_patch(ax_right, shape, edgecolor="cyan", lw=1.8)

    finite = np.asarray(img)[np.isfinite(img)]
    if finite.size:
        p_min, p_max = 0.0, 100.0
    else:
        finite = np.asarray([0.0, 1.0])
        p_min, p_max = 0.0, 100.0
    display_state = {"finite": finite}

    contrast_y_offset = 0.0 if wavelength_controller is None else 0.01
    slider_left = 0.10
    reset_width = 0.09
    reset_x = right_margin - reset_width - 0.02
    slider_width = reset_x - slider_left - 0.03
    ax_low = fig.add_axes([slider_left, 0.075 + contrast_y_offset, slider_width, 0.025])
    ax_high = fig.add_axes([slider_left, 0.035 + contrast_y_offset, slider_width, 0.025])
    ax_reset = fig.add_axes([reset_x, 0.04 + contrast_y_offset, reset_width, 0.055])
    low_slider = Slider(ax_low, "Low %", p_min, p_max, valinit=5.0, valstep=0.1)
    high_slider = Slider(ax_high, "High %", p_min, p_max, valinit=99.0, valstep=0.1)
    reset_button = Button(ax_reset, "Reset")
    wavelength_slider = None
    if wavelength_controller is not None:
        ax_wavelength = fig.add_axes(
            [slider_left, 0.145, right_margin - slider_left - 0.05, 0.032]
        )
        wavelength_slider = RangeSlider(
            ax_wavelength,
            "Wavelength (A)",
            wavelength_controller.available_min,
            wavelength_controller.available_max,
            valinit=wavelength_controller.bounds,
            valstep=wavelength_controller.slider_values,
        )

    def update_scale(_val=None) -> None:
        lo = float(low_slider.val)
        hi = float(high_slider.val)
        if hi <= lo:
            return
        finite = display_state["finite"]
        new_v1, new_v2 = np.nanpercentile(finite, [lo, hi])
        if not np.isfinite(new_v1) or not np.isfinite(new_v2) or new_v2 <= new_v1:
            return
        im.set_clim(new_v1, new_v2)
        im_right.set_clim(new_v1, new_v2)
        fig.canvas.draw_idle()

    def reset_scale(_event=None) -> None:
        low_slider.set_val(5.0)
        high_slider.set_val(99.0)
        if wavelength_slider is not None and wavelength_controller is not None:
            wavelength_slider.set_val(wavelength_controller.initial_bounds)

    def update_wavelength(bounds) -> None:
        if wavelength_controller is None:
            return
        wavelength_controller.set_bounds(float(bounds[0]), float(bounds[1]))
        updated = wavelength_controller.image()
        im.set_data(updated)
        im_right.set_data(updated)
        updated_finite = updated[np.isfinite(updated)]
        display_state["finite"] = (
            updated_finite if updated_finite.size else np.asarray([0.0, 1.0])
        )
        update_panel_titles()
        update_scale()

    low_slider.on_changed(update_scale)
    high_slider.on_changed(update_scale)
    reset_button.on_clicked(reset_scale)
    if wavelength_slider is not None:
        wavelength_slider.on_changed(update_wavelength)
    fig._kcwi_scale_widgets = (low_slider, high_slider, reset_button)
    fig._kcwi_wavelength_widgets = (wavelength_slider,)

    fig.suptitle(title)
    fig.colorbar(im, ax=[ax_left, ax_right], label="White-light", fraction=0.035, pad=0.03)
    return fig, ax_left, ax_right


def plot_apertures(img: np.ndarray,
                   apertures: TargetBackgroundApertures,
                   outpng: Path,
                   title: str,
                   show: bool = False,
                   wavelength_controller: Optional[WhiteLightRangeController] = None) -> None:
    fig, _, _ = _white_light_two_panel(
        img,
        title,
        apertures=apertures,
        wavelength_controller=wavelength_controller,
    )
    outpng.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpng, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _default_center(img: np.ndarray) -> Tuple[float, float]:
    ny, nx = img.shape
    return (nx - 1) / 2.0, (ny - 1) / 2.0


def prompt_aperture(shape_kind: str,
                    default_center: Tuple[float, float],
                    default_theta: float = 0.0) -> ApertureShape:
    """CLI prompts to define an ApertureShape."""
    shape_kind = shape_kind.lower().strip()

    if shape_kind == "ellipse":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        a = float(prompt("a (semi-major, pix)", "4.0"))
        b = float(prompt("b (semi-minor, pix)", "4.0"))
        theta = float(prompt("theta (rad)", f"{default_theta:.6f}"))
        return ApertureShape("ellipse", (x0, y0, a, b, theta))

    if shape_kind == "circle":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        r = float(prompt("r (pix)", "4.0"))
        return ApertureShape("circle", (x0, y0, r))

    if shape_kind == "rect":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        w = float(prompt("width (pix)", "6.0"))
        h = float(prompt("height (pix)", "6.0"))
        theta = float(prompt("theta (rad)", f"{default_theta:.6f}"))
        return ApertureShape("rect", (x0, y0, w, h, theta))

    if shape_kind == "ellipse_annulus":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        a_in = float(prompt("a_in (pix)", "6.0"))
        b_in = float(prompt("b_in (pix)", "6.0"))
        a_out = float(prompt("a_out (pix)", "10.0"))
        b_out = float(prompt("b_out (pix)", "10.0"))
        theta = float(prompt("theta (rad)", f"{default_theta:.6f}"))
        return ApertureShape("ellipse_annulus", (x0, y0, a_in, b_in, a_out, b_out, theta))

    if shape_kind == "circle_annulus":
        x0 = float(prompt("x0", f"{default_center[0]:.2f}"))
        y0 = float(prompt("y0", f"{default_center[1]:.2f}"))
        r_in = float(prompt("r_in (pix)", "6.0"))
        r_out = float(prompt("r_out (pix)", "10.0"))
        return ApertureShape("circle_annulus", (x0, y0, r_in, r_out))

    raise ValueError(f"Unknown shape kind: {shape_kind}")


def prompt_aperture_current(shape: ApertureShape) -> ApertureShape:
    """Prompt for an aperture's parameters, using the current values as defaults."""
    sh = shape.shape
    p = shape.params

    if sh == "ellipse":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        a = float(prompt("a (semi-major, pix)", f"{p[2]:.2f}"))
        b = float(prompt("b (semi-minor, pix)", f"{p[3]:.2f}"))
        theta = float(prompt("theta (rad)", f"{p[4]:.6f}"))
        return ApertureShape("ellipse", (x0, y0, a, b, theta))

    if sh == "circle":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        r = float(prompt("r (pix)", f"{p[2]:.2f}"))
        return ApertureShape("circle", (x0, y0, r))

    if sh == "rect":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        w = float(prompt("width (pix)", f"{p[2]:.2f}"))
        h = float(prompt("height (pix)", f"{p[3]:.2f}"))
        theta = float(prompt("theta (rad)", f"{p[4]:.6f}"))
        return ApertureShape("rect", (x0, y0, w, h, theta))

    if sh == "ellipse_annulus":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        a_in = float(prompt("a_in (pix)", f"{p[2]:.2f}"))
        b_in = float(prompt("b_in (pix)", f"{p[3]:.2f}"))
        a_out = float(prompt("a_out (pix)", f"{p[4]:.2f}"))
        b_out = float(prompt("b_out (pix)", f"{p[5]:.2f}"))
        theta = float(prompt("theta (rad)", f"{p[6]:.6f}"))
        return ApertureShape("ellipse_annulus", (x0, y0, a_in, b_in, a_out, b_out, theta))

    if sh == "circle_annulus":
        x0 = float(prompt("x0", f"{p[0]:.2f}"))
        y0 = float(prompt("y0", f"{p[1]:.2f}"))
        r_in = float(prompt("r_in (pix)", f"{p[2]:.2f}"))
        r_out = float(prompt("r_out (pix)", f"{p[3]:.2f}"))
        return ApertureShape("circle_annulus", (x0, y0, r_in, r_out))

    raise ValueError(f"Unknown shape kind: {sh}")


def _click_center(
    img: np.ndarray,
    title: str,
    *,
    wavelength_controller: Optional[WhiteLightRangeController] = None,
) -> Tuple[float, float]:
    """Pop up a figure and return a single clicked (x, y) center."""
    fig, _, _ = _white_light_two_panel(
        img,
        title + "\nClick once in either panel to set center; close window if needed",
        wavelength_controller=wavelength_controller,
    )
    # Block until one click is received.
    pts = plt.ginput(1, timeout=-1)
    plt.close(fig)
    if not pts:
        # If user closes window, keep current center by raising a controlled error
        raise RuntimeError("No click received (window closed).")
    x, y = pts[0]
    return float(x), float(y)


def _shape_center(shape: ApertureShape) -> Tuple[float, float]:
    return float(shape.params[0]), float(shape.params[1])


def _shape_from_drag(shape_kind: str, p0: Tuple[float, float], p1: Tuple[float, float]) -> ApertureShape:
    shape_kind = shape_kind.lower().strip()
    x0, y0 = p0
    x1, y1 = p1
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    eps = 0.5

    if shape_kind == "circle":
        r = max(eps, float(np.hypot(x1 - x0, y1 - y0)))
        return ApertureShape("circle", (float(x0), float(y0), r))

    if shape_kind == "ellipse":
        return ApertureShape("ellipse", (float(xc), float(yc), max(dx / 2.0, eps), max(dy / 2.0, eps), 0.0))

    if shape_kind in ("rect", "rectangle", "square"):
        if shape_kind == "square":
            side = max(dx, dy, eps)
            return ApertureShape("rect", (float(xc), float(yc), side, side, 0.0))
        return ApertureShape("rect", (float(xc), float(yc), max(dx, eps), max(dy, eps), 0.0))

    if shape_kind == "circle_annulus":
        r_in = max(eps, min(dx, dy) / 2.0)
        r_out = max(r_in + eps, max(dx, dy) / 2.0)
        return ApertureShape("circle_annulus", (float(xc), float(yc), r_in, r_out))

    if shape_kind == "ellipse_annulus":
        a_out = max(dx / 2.0, eps)
        b_out = max(dy / 2.0, eps)
        return ApertureShape(
            "ellipse_annulus",
            (float(xc), float(yc), max(a_out * 0.6, eps), max(b_out * 0.6, eps), a_out, b_out, 0.0),
        )

    raise ValueError(f"Unknown shape kind: {shape_kind}")


def _draw_shape_by_drag(
    img: np.ndarray,
    shape_kind: str,
    title: str,
    *,
    reference_shapes: Tuple[ApertureShape, ...] = (),
    wavelength_controller: Optional[WhiteLightRangeController] = None,
    celestial_wcs: Optional[WCS] = None,
) -> ApertureShape:
    """Create and adjust an aperture shape on either white-light panel."""
    print(f"{title}: click-drag-release to draw, then move/resize. Press a/Enter when done.")
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title + "\nClick-drag to draw. m=move, e=resize, drag to adjust. a/Enter=accept, r=redraw, q=cancel.",
        wavelength_controller=wavelength_controller,
        celestial_wcs=celestial_wcs,
    )
    state = {
        "start": None,
        "shape": None,
        "patches": [],
        "move_start": None,
        "move_center": None,
        "mode": "move",
        "accepted": False,
        "cancel": False,
    }

    for ax in (ax_left, ax_right):
        for ref in reference_shapes:
            _add_shape_patch(ax, ref, edgecolor="red", linestyle="-", lw=1.8)

    def clear_preview() -> None:
        for patch in state["patches"]:
            try:
                patch.remove()
            except Exception:
                pass
        state["patches"] = []

    def draw_preview(shape: ApertureShape) -> None:
        clear_preview()
        for ax in (ax_left, ax_right):
            patch = _patch_for_shape(shape, edgecolor="cyan", linestyle="-", lw=1.5)
            patches = patch if isinstance(patch, tuple) else (patch,)
            for item in patches:
                ax.add_patch(item)
                state["patches"].append(item)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        xy = (float(event.xdata), float(event.ydata))
        if state["shape"] is None:
            state["start"] = xy
            return
        state["move_start"] = xy
        state["move_center"] = _shape_center(state["shape"])

    def on_motion(event):
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        if state["start"] is not None:
            try:
                shape = _shape_from_drag(shape_kind, state["start"], (float(event.xdata), float(event.ydata)))
            except ValueError:
                return
            draw_preview(shape)
            return
        if state["shape"] is not None and state["move_start"] is not None:
            if state["mode"] == "resize":
                draw_preview(_resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata)))
            else:
                sx, sy = state["move_start"]
                cx, cy = state["move_center"]
                moved = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
                draw_preview(moved)

    def on_release(event):
        if state["start"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            if state["move_start"] is not None and event.inaxes in (ax_left, ax_right) and event.xdata is not None and event.ydata is not None:
                if state["mode"] == "resize":
                    state["shape"] = _resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata))
                else:
                    sx, sy = state["move_start"]
                    cx, cy = state["move_center"]
                    state["shape"] = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
                draw_preview(state["shape"])
            state["move_start"] = None
            state["move_center"] = None
            return
        state["shape"] = _shape_from_drag(shape_kind, state["start"], (float(event.xdata), float(event.ydata)))
        state["start"] = None
        draw_preview(state["shape"])

    def on_key(event):
        if event.key in ("a", "enter", "return"):
            state["accepted"] = True
            plt.close(fig)
        elif event.key == "m":
            state["mode"] = "move"
            fig.suptitle(title + "\nMOVE mode: drag aperture to move. e=resize, a/Enter=accept, r=redraw, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "e":
            state["mode"] = "resize"
            fig.suptitle(title + "\nRESIZE mode: drag to set edge/corner from current center. m=move, a/Enter=accept, r=redraw, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "r":
            state["shape"] = None
            state["start"] = None
            state["move_start"] = None
            state["move_center"] = None
            clear_preview()
            fig.canvas.draw_idle()
        elif event.key == "q":
            state["cancel"] = True
            state["shape"] = None
            plt.close(fig)

    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    clear_preview()
    if state["cancel"] or not state["accepted"] or state["shape"] is None:
        raise RuntimeError("No aperture was drawn.")
    return state["shape"]


def _drag_move_shape(
    img: np.ndarray,
    shape: ApertureShape,
    title: str,
    *,
    reference_shapes: Tuple[ApertureShape, ...] = (),
    wavelength_controller: Optional[WhiteLightRangeController] = None,
    celestial_wcs: Optional[WCS] = None,
) -> ApertureShape:
    """Move an existing aperture by dragging on either panel until accepted."""
    print(f"{title}: click-drag the aperture on either panel to move/resize. Press a/Enter when done.")
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title + "\nm=move, e=resize, drag to adjust. a/Enter=accept, q=cancel.",
        wavelength_controller=wavelength_controller,
        celestial_wcs=celestial_wcs,
    )

    for ax in (ax_left, ax_right):
        for ref in reference_shapes:
            _add_shape_patch(ax, ref, edgecolor="red", linestyle="-", lw=1.8)

    state = {
        "shape": shape,
        "start_xy": None,
        "start_center": None,
        "mode": "move",
        "accepted": False,
        "cancel": False,
        "patches": [],
    }

    def clear_preview() -> None:
        for patch in state["patches"]:
            try:
                patch.remove()
            except Exception:
                pass
        state["patches"] = []

    def draw_preview(current: ApertureShape) -> None:
        clear_preview()
        for ax in (ax_left, ax_right):
            patch = _patch_for_shape(current, edgecolor="cyan", lw=1.8)
            patches = patch if isinstance(patch, tuple) else (patch,)
            for item in patches:
                ax.add_patch(item)
                state["patches"].append(item)
        fig.canvas.draw_idle()

    draw_preview(shape)

    def on_press(event):
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        state["start_xy"] = (float(event.xdata), float(event.ydata))
        state["start_center"] = _shape_center(state["shape"])

    def on_motion(event):
        if state["start_xy"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        if state["mode"] == "resize":
            draw_preview(_resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata)))
        else:
            sx, sy = state["start_xy"]
            cx, cy = state["start_center"]
            moved = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
            draw_preview(moved)

    def on_release(event):
        if state["start_xy"] is None or event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            state["start_xy"] = None
            state["start_center"] = None
            return
        if state["mode"] == "resize":
            state["shape"] = _resize_shape_to_point(state["shape"], float(event.xdata), float(event.ydata))
        else:
            sx, sy = state["start_xy"]
            cx, cy = state["start_center"]
            state["shape"] = _update_shape_center(state["shape"], cx + float(event.xdata) - sx, cy + float(event.ydata) - sy)
        state["start_xy"] = None
        state["start_center"] = None
        draw_preview(state["shape"])

    def on_key(event):
        if event.key in ("a", "enter", "return"):
            state["accepted"] = True
            plt.close(fig)
        elif event.key == "m":
            state["mode"] = "move"
            fig.suptitle(title + "\nMOVE mode: drag aperture to move. e=resize, a/Enter=accept, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "e":
            state["mode"] = "resize"
            fig.suptitle(title + "\nRESIZE mode: drag to set edge/corner from current center. m=move, a/Enter=accept, q=cancel.")
            fig.canvas.draw_idle()
        elif event.key == "q":
            state["cancel"] = True
            plt.close(fig)

    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    clear_preview()
    if state["cancel"] or not state["accepted"]:
        raise RuntimeError("Aperture move cancelled.")
    return state["shape"]


def _auto_background_from_target(target: ApertureShape, kind: str) -> ApertureShape:
    kind = kind.lower().strip()
    p = target.params
    if kind == "circle_annulus":
        if target.shape == "circle":
            x0, y0, r = p
            return ApertureShape("circle_annulus", (x0, y0, r * 1.8, r * 3.0))
        x0, y0 = p[0], p[1]
        scale = max(p[2:4]) if len(p) >= 4 else 4.0
        return ApertureShape("circle_annulus", (x0, y0, scale * 1.8, scale * 3.0))

    if kind == "ellipse_annulus":
        x0, y0 = p[0], p[1]
        if target.shape == "ellipse":
            _, _, a, b, theta = p
        elif target.shape == "circle":
            _, _, r = p
            a, b, theta = r, r, 0.0
        elif target.shape == "rect":
            _, _, w, h, theta = p
            a, b = w / 2.0, h / 2.0
        else:
            a, b, theta = 4.0, 4.0, 0.0
        return ApertureShape("ellipse_annulus", (x0, y0, a * 1.8, b * 1.8, a * 3.0, b * 3.0, theta))

    raise ValueError(f"Cannot auto-create background kind: {kind}")


def _update_shape_center(shape: ApertureShape, x0: float, y0: float) -> ApertureShape:
    p = list(shape.params)
    if len(p) < 2:
        raise ValueError("Shape params must include x0,y0 in first two entries.")
    p[0], p[1] = float(x0), float(y0)
    return ApertureShape(shape.shape, tuple(p))


def _resize_shape_to_point(shape: ApertureShape, x: float, y: float) -> ApertureShape:
    """Resize a shape around its current center using the pointer as an edge/corner."""
    sh = shape.shape
    p = list(shape.params)
    x0, y0 = float(p[0]), float(p[1])
    dx = abs(float(x) - x0)
    dy = abs(float(y) - y0)
    eps = 0.5

    if sh == "circle":
        return ApertureShape(sh, (x0, y0, max(float(np.hypot(float(x) - x0, float(y) - y0)), eps)))

    if sh == "ellipse":
        return ApertureShape(sh, (x0, y0, max(dx, eps), max(dy, eps), p[4]))

    if sh == "rect":
        return ApertureShape(sh, (x0, y0, max(2.0 * dx, eps), max(2.0 * dy, eps), p[4]))

    if sh == "circle_annulus":
        r_in_old, r_out_old = max(float(p[2]), eps), max(float(p[3]), eps)
        ratio = min(0.95, r_in_old / r_out_old)
        r_out = max(float(np.hypot(float(x) - x0, float(y) - y0)), r_in_old + eps, eps)
        return ApertureShape(sh, (x0, y0, max(r_out * ratio, eps), r_out))

    if sh == "ellipse_annulus":
        a_in_old, b_in_old = max(float(p[2]), eps), max(float(p[3]), eps)
        a_out_old, b_out_old = max(float(p[4]), eps), max(float(p[5]), eps)
        a_ratio = min(0.95, a_in_old / a_out_old)
        b_ratio = min(0.95, b_in_old / b_out_old)
        a_out = max(dx, eps)
        b_out = max(dy, eps)
        return ApertureShape(sh, (x0, y0, max(a_out * a_ratio, eps), max(b_out * b_ratio, eps), a_out, b_out, p[6]))

    raise ValueError(f"Unknown shape: {sh}")


def _shape_after_drag(
    shape: ApertureShape,
    mode: str,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
) -> ApertureShape:
    """Return a moved or resized shape for one pointer drag."""
    if mode == "resize":
        return _resize_shape_to_point(shape, end_xy[0], end_xy[1])
    if mode != "move":
        raise ValueError(f"Unknown aperture edit mode: {mode}")
    x0, y0 = _shape_center(shape)
    return _update_shape_center(
        shape,
        x0 + float(end_xy[0]) - float(start_xy[0]),
        y0 + float(end_xy[1]) - float(start_xy[1]),
    )


def _shape_parameter_labels(shape: ApertureShape) -> Tuple[str, ...]:
    labels = {
        "circle": ("x0", "y0", "radius"),
        "ellipse": ("x0", "y0", "semi-major", "semi-minor", "theta rad"),
        "rect": ("x0", "y0", "width", "height", "theta rad"),
        "circle_annulus": ("x0", "y0", "inner radius", "outer radius"),
        "ellipse_annulus": (
            "x0",
            "y0",
            "inner a",
            "inner b",
            "outer a",
            "outer b",
            "theta rad",
        ),
    }
    try:
        return labels[shape.shape]
    except KeyError as exc:
        raise ValueError(f"Unknown shape: {shape.shape}") from exc


def _shape_from_parameter_values(
    shape: ApertureShape,
    values: Tuple[float, ...],
) -> ApertureShape:
    expected = len(_shape_parameter_labels(shape))
    if len(values) != expected:
        raise ValueError(f"{shape.shape} requires {expected} values")
    values = tuple(float(value) for value in values)
    if not np.all(np.isfinite(values)):
        raise ValueError("All aperture values must be finite")

    if shape.shape == "circle" and values[2] <= 0:
        raise ValueError("Radius must be positive")
    if shape.shape in {"ellipse", "rect"} and (values[2] <= 0 or values[3] <= 0):
        raise ValueError("Aperture dimensions must be positive")
    if shape.shape == "circle_annulus":
        if values[2] <= 0 or values[3] <= values[2]:
            raise ValueError("Outer radius must be larger than positive inner radius")
    if shape.shape == "ellipse_annulus":
        if min(values[2:6]) <= 0 or values[4] <= values[2] or values[5] <= values[3]:
            raise ValueError("Outer annulus axes must exceed positive inner axes")
    return ApertureShape(shape.shape, values)


def _shape_outer_half_extents(shape: ApertureShape) -> Tuple[float, float]:
    p = shape.params
    if shape.shape == "circle":
        return float(p[2]), float(p[2])
    if shape.shape == "ellipse":
        return float(p[2]), float(p[3])
    if shape.shape == "rect":
        return float(p[2]) / 2.0, float(p[3]) / 2.0
    if shape.shape == "circle_annulus":
        return float(p[3]), float(p[3])
    if shape.shape == "ellipse_annulus":
        return float(p[4]), float(p[5])
    raise ValueError(f"Unknown shape: {shape.shape}")


def _convert_aperture_shape(
    shape: ApertureShape,
    new_kind: str,
) -> ApertureShape:
    """Convert shape geometry while preserving its center and approximate extent."""
    new_kind = new_kind.lower().strip().replace(" ", "_")
    x0, y0 = _shape_center(shape)
    half_x, half_y = _shape_outer_half_extents(shape)
    half_x = max(abs(half_x), 0.5)
    half_y = max(abs(half_y), 0.5)
    theta = 0.0
    if shape.shape in {"ellipse", "rect"}:
        theta = float(shape.params[4])
    elif shape.shape == "ellipse_annulus":
        theta = float(shape.params[6])

    if new_kind == "circle":
        return ApertureShape("circle", (x0, y0, 0.5 * (half_x + half_y)))
    if new_kind == "ellipse":
        return ApertureShape("ellipse", (x0, y0, half_x, half_y, theta))
    if new_kind == "rect":
        return ApertureShape("rect", (x0, y0, 2.0 * half_x, 2.0 * half_y, theta))
    if new_kind == "square":
        side = 2.0 * max(half_x, half_y)
        return ApertureShape("rect", (x0, y0, side, side, theta))
    if new_kind == "circle_annulus":
        outer = max(half_x, half_y)
        return ApertureShape("circle_annulus", (x0, y0, 0.6 * outer, outer))
    if new_kind == "ellipse_annulus":
        return ApertureShape(
            "ellipse_annulus",
            (x0, y0, 0.6 * half_x, 0.6 * half_y, half_x, half_y, theta),
        )
    raise ValueError(f"Unknown shape kind: {new_kind}")


def _review_apertures_in_window(
    img: np.ndarray,
    apertures: TargetBackgroundApertures,
    title: str,
    *,
    wavelength_controller: Optional[WhiteLightRangeController] = None,
    celestial_wcs: Optional[WCS] = None,
) -> Optional[TargetBackgroundApertures]:
    """Review and edit both apertures in one self-contained blocking window."""
    fig, ax_left, ax_right = _white_light_two_panel(
        img,
        title,
        wavelength_controller=wavelength_controller,
        celestial_wcs=celestial_wcs,
        right_margin=0.70,
    )

    state = {
        "apertures": apertures,
        "active": "target",
        "mode": "move",
        "drag_start": None,
        "drag_shape": None,
        "accepted": False,
        "cancelled": False,
        "patches": [],
        "modal": None,
        "modal_axes": [],
        "modal_artists": [],
        "modal_widgets": [],
    }

    button_specs = (
        ("accept", "Accept", 0.845),
        ("move_target", "Move target", 0.715),
        ("resize_target", "Resize target", 0.655),
        ("shape_target", "Target shape...", 0.595),
        ("move_background", "Move background", 0.465),
        ("resize_background", "Resize background", 0.405),
        ("shape_background", "Background shape...", 0.345),
        ("enter_values", "Enter values...", 0.245),
        ("cancel", "Cancel", 0.185),
    )
    buttons = {}
    for name, label, y0 in button_specs:
        button_ax = fig.add_axes([0.805, y0, 0.18, 0.048])
        buttons[name] = Button(button_ax, label)
    group_artists = [
        fig.text(0.805, 0.785, "TARGET", fontsize=10, fontweight="bold", color="darkred"),
        fig.text(
            0.805,
            0.535,
            "BACKGROUND",
            fontsize=10,
            fontweight="bold",
            color="darkorange",
        ),
    ]

    def active_shape() -> ApertureShape:
        current = state["apertures"]
        return current.target if state["active"] == "target" else current.background

    def set_active_shape(shape: ApertureShape) -> None:
        current = state["apertures"]
        if state["active"] == "target":
            state["apertures"] = TargetBackgroundApertures(
                target=shape,
                background=current.background,
            )
        else:
            state["apertures"] = TargetBackgroundApertures(
                target=current.target,
                background=shape,
            )

    def clear_patches() -> None:
        for patch in state["patches"]:
            try:
                patch.remove()
            except Exception:
                pass
        state["patches"] = []

    def add_preview(shape: ApertureShape, *, edgecolor: str, lw: float) -> None:
        for ax in (ax_left, ax_right):
            patch = _patch_for_shape(shape, edgecolor=edgecolor, lw=lw)
            patches = patch if isinstance(patch, tuple) else (patch,)
            for item in patches:
                ax.add_patch(item)
                state["patches"].append(item)

    def set_main_controls_visible(visible: bool) -> None:
        for button in buttons.values():
            button.set_active(visible)
            button.ax.set_visible(visible)
        for artist in group_artists:
            artist.set_visible(visible)

    def clear_modal() -> None:
        modal_widgets = list(state["modal_widgets"])
        for widget in modal_widgets:
            try:
                widget.disconnect_events()
            except Exception:
                pass
        for widget in modal_widgets:
            if isinstance(widget, TextBox) and widget.capturekeystrokes:
                try:
                    widget.stop_typing()
                except Exception:
                    pass
        for axes in state["modal_axes"]:
            try:
                axes.remove()
            except Exception:
                pass
        for artist in state["modal_artists"]:
            try:
                artist.remove()
            except Exception:
                pass
        state["modal"] = None
        state["modal_axes"] = []
        state["modal_artists"] = []
        state["modal_widgets"] = []
        set_main_controls_visible(True)

    def update_title_and_buttons() -> None:
        active = str(state["active"])
        mode = str(state["mode"])
        fig.suptitle(
            f"{title}\n{active.upper()} {mode.upper()}: drag in either panel. "
            "Use the controls at right, then Accept."
        )
        active_button = f"{mode}_{active}"
        for name, button in buttons.items():
            if name == "accept":
                color = "#b7e4c7"
            elif name == "cancel":
                color = "#f4cccc"
            elif name == active_button:
                color = "#9ecae1"
            else:
                color = "0.92"
            button.ax.set_facecolor(color)

    def draw_preview() -> None:
        clear_patches()
        current = state["apertures"]
        target_lw = 2.8 if state["active"] == "target" else 1.8
        background_lw = 2.8 if state["active"] == "background" else 1.6
        add_preview(current.target, edgecolor="red", lw=target_lw)
        add_preview(current.background, edgecolor="orange", lw=background_lw)
        update_title_and_buttons()
        fig.canvas.draw_idle()

    def set_mode(active: str, mode: str) -> None:
        if active not in {"target", "background"}:
            raise ValueError(f"Unknown active aperture: {active}")
        if mode not in {"move", "resize"}:
            raise ValueError(f"Unknown aperture edit mode: {mode}")
        state["active"] = active
        state["mode"] = mode
        state["drag_start"] = None
        state["drag_shape"] = None
        draw_preview()

    def change_shape(active: str, new_kind: str) -> None:
        state["active"] = active
        current = active_shape()
        normalized_kind = new_kind.lower().strip().replace(" ", "_")
        if active == "background" and normalized_kind.startswith("auto_"):
            annulus_kind = normalized_kind.removeprefix("auto_")
            replacement = _auto_background_from_target(
                state["apertures"].target,
                annulus_kind,
            )
        else:
            replacement = _convert_aperture_shape(current, normalized_kind)
        set_active_shape(replacement)
        state["mode"] = "resize"
        draw_preview()

    def set_values(active: str, values: Tuple[float, ...]) -> None:
        state["active"] = active
        set_active_shape(_shape_from_parameter_values(active_shape(), values))
        draw_preview()

    def apply_drag(start_xy: Tuple[float, float], end_xy: Tuple[float, float]) -> None:
        set_active_shape(
            _shape_after_drag(active_shape(), str(state["mode"]), start_xy, end_xy)
        )
        draw_preview()

    def accept(_event=None) -> None:
        if state["modal"] is not None:
            return
        state["accepted"] = True
        plt.close(fig)

    def cancel(_event=None) -> None:
        state["cancelled"] = True
        plt.close(fig)

    def modal_artist(x: float, y: float, text: str, **kwargs):
        artist = fig.text(x, y, text, **kwargs)
        state["modal_artists"].append(artist)
        return artist

    def modal_axes(bounds):
        axes = fig.add_axes(bounds)
        state["modal_axes"].append(axes)
        return axes

    def open_shape_selector(active: str) -> None:
        clear_modal()
        set_main_controls_visible(False)
        state["modal"] = "shape"
        state["active"] = active
        if active == "target":
            choices = ("circle", "ellipse", "rect", "square")
        else:
            choices = (
                "auto ellipse annulus",
                "auto circle annulus",
                "ellipse annulus",
                "circle annulus",
                "ellipse",
                "circle",
                "rect",
            )
        modal_artist(
            0.805,
            0.855,
            f"Choose {active} shape",
            fontsize=10,
            fontweight="bold",
        )
        radio_ax = modal_axes([0.805, 0.36, 0.18, 0.46])
        current_name = active_shape().shape.replace("_", " ")
        active_index = choices.index(current_name) if current_name in choices else 0
        radio = RadioButtons(radio_ax, choices, active=active_index)
        apply_button = Button(modal_axes([0.805, 0.275, 0.085, 0.052]), "Apply")
        back_button = Button(modal_axes([0.905, 0.275, 0.08, 0.052]), "Back")

        def apply_selection(_event=None) -> None:
            selected = str(radio.value_selected).replace(" ", "_")
            change_shape(active, selected)
            clear_modal()
            draw_preview()

        apply_button.on_clicked(apply_selection)
        back_button.on_clicked(lambda _event: (clear_modal(), draw_preview()))
        state["modal_widgets"] = [radio, apply_button, back_button]
        fig.canvas.draw_idle()

    def open_value_editor() -> None:
        active = str(state["active"])
        shape = active_shape()
        labels = _shape_parameter_labels(shape)
        clear_modal()
        set_main_controls_visible(False)
        state["modal"] = "values"
        modal_artist(
            0.805,
            0.865,
            f"{active.title()} values\n{shape.shape.replace('_', ' ')}",
            fontsize=10,
            fontweight="bold",
        )
        error_artist = modal_artist(0.805, 0.105, "", fontsize=8, color="darkred")
        text_boxes = []
        start_y = 0.785
        spacing = 0.072
        for index, (label, value) in enumerate(zip(labels, shape.params)):
            text_box = TextBox(
                modal_axes([0.895, start_y - index * spacing, 0.09, 0.043]),
                label,
                initial=f"{float(value):.7g}",
            )
            text_boxes.append(text_box)
        apply_button = Button(modal_axes([0.805, 0.16, 0.085, 0.052]), "Apply")
        back_button = Button(modal_axes([0.905, 0.16, 0.08, 0.052]), "Back")

        def apply_values(_event=None) -> None:
            try:
                values = tuple(float(box.text) for box in text_boxes)
                set_values(active, values)
            except (TypeError, ValueError) as exc:
                error_artist.set_text(str(exc))
                fig.canvas.draw_idle()
                return
            clear_modal()
            draw_preview()

        apply_button.on_clicked(apply_values)
        back_button.on_clicked(lambda _event: (clear_modal(), draw_preview()))
        state["modal_widgets"] = [*text_boxes, apply_button, back_button]
        fig.canvas.draw_idle()

    def on_press(event) -> None:
        if event.inaxes not in (ax_left, ax_right) or event.xdata is None or event.ydata is None:
            return
        state["drag_start"] = (float(event.xdata), float(event.ydata))
        state["drag_shape"] = active_shape()

    def update_from_event(event) -> None:
        if (
            state["drag_start"] is None
            or state["drag_shape"] is None
            or event.inaxes not in (ax_left, ax_right)
            or event.xdata is None
            or event.ydata is None
        ):
            return
        edited = _shape_after_drag(
            state["drag_shape"],
            str(state["mode"]),
            state["drag_start"],
            (float(event.xdata), float(event.ydata)),
        )
        set_active_shape(edited)
        draw_preview()

    def on_motion(event) -> None:
        update_from_event(event)

    def on_release(event) -> None:
        update_from_event(event)
        state["drag_start"] = None
        state["drag_shape"] = None

    def on_key(event) -> None:
        key = str(event.key or "").lower()
        if state["modal"] is not None:
            if key == "escape":
                clear_modal()
                draw_preview()
            return
        if key in {"a", "enter", "return"}:
            accept()
        elif key == "t":
            set_mode("target", "move")
        elif key == "b":
            set_mode("background", "move")
        elif key == "m":
            set_mode(str(state["active"]), "move")
        elif key == "e":
            set_mode(str(state["active"]), "resize")
        elif key == "q":
            cancel()

    buttons["accept"].on_clicked(accept)
    buttons["move_target"].on_clicked(lambda _event: set_mode("target", "move"))
    buttons["resize_target"].on_clicked(lambda _event: set_mode("target", "resize"))
    buttons["shape_target"].on_clicked(lambda _event: open_shape_selector("target"))
    buttons["move_background"].on_clicked(lambda _event: set_mode("background", "move"))
    buttons["resize_background"].on_clicked(
        lambda _event: set_mode("background", "resize")
    )
    buttons["shape_background"].on_clicked(
        lambda _event: open_shape_selector("background")
    )
    buttons["enter_values"].on_clicked(lambda _event: open_value_editor())
    buttons["cancel"].on_clicked(cancel)
    cids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    fig._kcwi_aperture_review_widgets = buttons
    fig._kcwi_aperture_review_state = state
    fig._kcwi_aperture_review_actions = {
        "set_mode": set_mode,
        "apply_drag": apply_drag,
        "change_shape": change_shape,
        "set_values": set_values,
        "accept": accept,
        "cancel": cancel,
    }
    draw_preview()
    plt.show()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    clear_modal()
    clear_patches()
    plt.close(fig)
    return state["apertures"] if state["accepted"] else None


def _preview_apertures_blocking(img: np.ndarray,
                               aps: TargetBackgroundApertures,
                               title: str,
                               *,
                               wavelength_controller: Optional[WhiteLightRangeController] = None) -> None:
    """Show apertures overlay in a blocking window (no file write)."""
    fig, _, _ = _white_light_two_panel(
        img,
        title,
        apertures=aps,
        wavelength_controller=wavelength_controller,
    )
    plt.show()
    plt.close(fig)


def review_apertures(img: np.ndarray,
                     apertures: TargetBackgroundApertures,
                     side_label: str,
                     show: bool = False,
                     wavelength_controller: Optional[WhiteLightRangeController] = None,
                     celestial_wcs: Optional[WCS] = None) -> TargetBackgroundApertures:
    """Show the unified aperture editor and return only after in-window acceptance."""
    reviewed = _review_apertures_in_window(
        img,
        apertures,
        title=f"{side_label} aperture editor",
        wavelength_controller=wavelength_controller,
        celestial_wcs=celestial_wcs,
    )
    if reviewed is None:
        raise RuntimeError("Aperture editor closed without accepting the apertures.")
    return reviewed


def interactive_define_apertures(img: np.ndarray,
                                 side_label: str,
                                 show: bool = False,
                                 wavelength_controller: Optional[WhiteLightRangeController] = None,
                                 celestial_wcs: Optional[WCS] = None) -> TargetBackgroundApertures:
    """Interactively define target + background apertures (independent) with iterative recentering.

    Workflow
    --------
    1) Choose shape + size parameters for TARGET and BACKGROUND.
    2) Optionally click to set their centers.
    3) Preview overlay.
    4) Iterate:
        - recenter target by clicking
        - recenter background by clicking
        - edit numeric parameters
      until user approves.

    The pipeline will only proceed once the user approves the apertures.
    """
    print(f"\n[{side_label}] Define TARGET aperture")
    tgt_kind = prompt("Target shape (circle/ellipse/rect/square)", "ellipse").lower().strip()
    while True:
        try:
            tgt = _draw_shape_by_drag(
                img,
                tgt_kind,
                title=f"{side_label} TARGET aperture",
                wavelength_controller=wavelength_controller,
                celestial_wcs=celestial_wcs,
            )
        except RuntimeError as exc:
            print(exc)
            if not prompt("Retry target aperture? (y/n)", "y").lower().startswith("y"):
                raise SystemExit("User quit during target aperture definition.")
            continue
        break

    print(f"\n[{side_label}] Define BACKGROUND region")
    bkg_kind = prompt("Background shape (auto_ellipse_annulus/auto_circle_annulus/ellipse_annulus/circle_annulus/ellipse/circle/rect)", "auto_ellipse_annulus").lower().strip()
    if bkg_kind == "auto_ellipse_annulus":
        bkg = _auto_background_from_target(tgt, "ellipse_annulus")
        try:
            bkg = _drag_move_shape(
                img,
                bkg,
                title=f"{side_label} BACKGROUND region",
                reference_shapes=(tgt,),
                wavelength_controller=wavelength_controller,
                celestial_wcs=celestial_wcs,
            )
        except RuntimeError:
            print("BACKGROUND not changed.")
    elif bkg_kind == "auto_circle_annulus":
        bkg = _auto_background_from_target(tgt, "circle_annulus")
        try:
            bkg = _drag_move_shape(
                img,
                bkg,
                title=f"{side_label} BACKGROUND region",
                reference_shapes=(tgt,),
                wavelength_controller=wavelength_controller,
                celestial_wcs=celestial_wcs,
            )
        except RuntimeError:
            print("BACKGROUND not changed.")
    else:
        draw_kind = bkg_kind
        if draw_kind in ("ellipse", "circle", "rect", "rectangle", "square"):
            print("Background will be drawn as a filled region, not an annulus.")
        while True:
            try:
                bkg = _draw_shape_by_drag(
                    img,
                    draw_kind,
                    title=f"{side_label} BACKGROUND region",
                    reference_shapes=(tgt,),
                    wavelength_controller=wavelength_controller,
                    celestial_wcs=celestial_wcs,
                )
            except RuntimeError as exc:
                print(exc)
                if not prompt("Retry background region? (y/n)", "y").lower().startswith("y"):
                    raise SystemExit("User quit during background definition.")
                continue
            break

    aps = TargetBackgroundApertures(target=tgt, background=bkg)
    return review_apertures(
        img,
        aps,
        side_label=side_label,
        show=show,
        wavelength_controller=wavelength_controller,
        celestial_wcs=celestial_wcs,
    )
