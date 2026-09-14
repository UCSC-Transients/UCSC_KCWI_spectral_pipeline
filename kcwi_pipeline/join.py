from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def concat_join(
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    *,
    sort_by_wavelength: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """KCWI join: simple concatenation (NO overlap stitching)."""
    lam = np.concatenate([np.asarray(lam_blue).ravel(), np.asarray(lam_red).ravel()])
    flux = np.concatenate([np.asarray(flux_blue).ravel(), np.asarray(flux_red).ravel()])
    if sort_by_wavelength:
        s = np.argsort(lam)
        lam, flux = lam[s], flux[s]
    return lam, flux


def plot_join_diagnostic(
    objname: str,
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    *,
    outpng: Path,
    show: bool,
) -> None:
    """Plot BLUE and RED together; supports saving a zoomed-in version.

    While the plot window is open:
      - Use the matplotlib toolbar to zoom/pan.
      - Press 'z' to save a zoomed PNG using the current axis limits.
    """
    outpng = Path(outpng)
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    if np.asarray(lam_blue).size:
        ax.plot(lam_blue, flux_blue, lw=1.0, label="Blue")
    if np.asarray(lam_red).size:
        ax.plot(lam_red, flux_red, lw=1.0, label="Red")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"{objname}: BLUE+RED diagnostic (press 'z' to save zoom)")
    ax.legend()
    ax.grid(alpha=0.2)

    # Always save the full-range diagnostic
    fig.savefig(outpng, dpi=200, bbox_inches="tight")

    def save_zoomed() -> Path:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        fn = outpng.with_name(
            outpng.stem + f"_zoom_x{x0:.0f}-{x1:.0f}_y{y0:.3g}-{y1:.3g}" + outpng.suffix
        )
        fig.savefig(fn, dpi=200, bbox_inches="tight")
        return fn

    def on_key(event):
        if event.key == "z":
            try:
                fn = save_zoomed()
                ax.set_title(f"{objname}: BLUE+RED diagnostic [zoom saved]")
                fig.canvas.draw_idle()
                print(f"Saved zoomed plot: {fn}")
            except Exception as e:
                print(f"Could not save zoomed plot: {e}")

    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    if show:
        plt.show()
    else:
        plt.close(fig)

    fig.canvas.mpl_disconnect(cid)


def interactive_rescale_and_approve_flux(
    *,
    objname: str,
    lam_blue: np.ndarray,
    flux_blue: np.ndarray,
    lam_red: np.ndarray,
    flux_red: np.ndarray,
    outdir: Path,
    show: bool = True,
    interactive: bool = True,
    output_suffix: str = "",
) -> Tuple[float, float]:
    """Iteratively rescale BLUE/RED until user approves.

    Returns (blue_scale, red_scale). These multiplicative factors are applied
    to the provided flux arrays *before* saving the final joined spectrum.

    Controls (interactive=True):
      - Use toolbar to zoom/pan.
      - Press 'z' to save zoomed view.
      - Press 'a' (or Enter) in the figure to approve current scaling and continue.
      - Press 'q' in the figure to abort.
      - If figure key events are missed, terminal prompts also provide approve/abort.

    After closing without approval, you'll be prompted in the terminal to update
    scales and the plot will reopen.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scale_path = outdir / f"{objname}_join_scale{output_suffix}.txt"

    blue_scale = 1.0
    red_scale = 1.0

    if not interactive:
        with open(scale_path, "w") as f:
            f.write(f"blue_scale {blue_scale}\nred_scale {red_scale}\n")
        return blue_scale, red_scale

    while True:
        if not show:
            print("\nRescale options:")
            print("  [a] Approve current scaling and continue")
            print("  [1] Multiply RED by factor")
            print("  [2] Multiply BLUE by factor")
            print("  [3] Set RED scale (absolute)")
            print("  [4] Set BLUE scale (absolute)")
            print("  [q] Abort pipeline")
            print("  [Enter] Keep current scales")
            choice = input("Choice: ").strip()
            if choice.lower() == "a":
                with open(scale_path, "w") as f:
                    f.write(f"blue_scale {blue_scale}\n")
                    f.write(f"red_scale {red_scale}\n")
                return blue_scale, red_scale
            if choice.lower() == "q":
                raise RuntimeError("User aborted join rescale approval (terminal 'q').")
            if choice == "1":
                red_scale *= float(input(f"Multiply RED by factor (current {red_scale}): ").strip())
            elif choice == "2":
                blue_scale *= float(input(f"Multiply BLUE by factor (current {blue_scale}): ").strip())
            elif choice == "3":
                red_scale = float(input(f"Set RED scale (current {red_scale}): ").strip())
            elif choice == "4":
                blue_scale = float(input(f"Set BLUE scale (current {blue_scale}): ").strip())
            else:
                continue

        lam_blue = np.asarray(lam_blue)
        lam_red = np.asarray(lam_red)
        flux_blue = np.asarray(flux_blue)
        flux_red = np.asarray(flux_red)

        fig, ax = plt.subplots(figsize=(11, 5.8))
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.24, top=0.88)
        line_b, = ax.plot(lam_blue, flux_blue * blue_scale, lw=1.0, label=f"Blue x{blue_scale:.4g}")
        line_r, = ax.plot(lam_red, flux_red * red_scale, lw=1.0, label=f"Red x{red_scale:.4g}")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.set_title(f"{objname}: adjust join scaling (sliders; a/Enter=approve, z=save zoom, q=abort)")
        ax.grid(alpha=0.2)
        legend = ax.legend()

        scale_state = {"blue": blue_scale, "red": red_scale}
        decision = {"approved": False, "abort": False}

        ax_blue_slider = fig.add_axes([0.16, 0.125, 0.58, 0.028])
        ax_red_slider = fig.add_axes([0.16, 0.08, 0.58, 0.028])
        ax_reset = fig.add_axes([0.77, 0.08, 0.08, 0.075])
        ax_approve = fig.add_axes([0.86, 0.08, 0.09, 0.075])
        blue_init = float(np.clip(np.log10(max(blue_scale, 1e-6)), -1.0, 1.0))
        red_init = float(np.clip(np.log10(max(red_scale, 1e-6)), -1.0, 1.0))
        blue_slider = Slider(ax_blue_slider, "Blue scale", -1.0, 1.0, valinit=blue_init, valstep=0.005)
        red_slider = Slider(ax_red_slider, "Red scale", -1.0, 1.0, valinit=red_init, valstep=0.005)
        reset_button = Button(ax_reset, "Reset")
        approve_button = Button(ax_approve, "Approve")

        def refresh() -> None:
            nonlocal legend
            scale_state["blue"] = 10.0 ** float(blue_slider.val)
            scale_state["red"] = 10.0 ** float(red_slider.val)
            line_b.set_ydata(flux_blue * scale_state["blue"])
            line_r.set_ydata(flux_red * scale_state["red"])
            line_b.set_label(f"Blue x{scale_state['blue']:.4g}")
            line_r.set_label(f"Red x{scale_state['red']:.4g}")
            if legend is not None:
                legend.remove()
            legend = ax.legend()
            fig.canvas.draw_idle()

        def reset(_event=None) -> None:
            blue_slider.set_val(0.0)
            red_slider.set_val(0.0)

        def approve(_event=None) -> None:
            decision["approved"] = True
            plt.close(fig)

        def save_zoomed() -> Path:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            fn = outdir / f"{objname}_join_scaling{output_suffix}_zoom_x{x0:.0f}-{x1:.0f}_y{y0:.3g}-{y1:.3g}.png"
            fig.savefig(fn, dpi=200, bbox_inches="tight")
            return fn

        def on_key(event):
            if event.key in ("a", "enter", "return"):
                approve()
            elif event.key == "q":
                decision["abort"] = True
                plt.close(fig)
            elif event.key == "z":
                try:
                    fn = save_zoomed()
                    ax.set_title(f"{objname}: adjust join scaling [zoom saved]")
                    fig.canvas.draw_idle()
                    print(f"Saved zoomed plot: {fn}")
                except Exception as e:
                    print(f"Could not save zoomed plot: {e}")

        blue_slider.on_changed(lambda _val: refresh())
        red_slider.on_changed(lambda _val: refresh())
        reset_button.on_clicked(reset)
        approve_button.on_clicked(approve)
        fig._kcwi_join_widgets = (blue_slider, red_slider, reset_button, approve_button)
        cid = fig.canvas.mpl_connect("key_press_event", on_key)

        fig.savefig(outdir / f"{objname}_join_scaling{output_suffix}.png", dpi=200, bbox_inches="tight")
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if decision["abort"]:
            raise RuntimeError("User aborted join rescale approval (pressed 'q').")

        if decision["approved"]:
            blue_scale = float(scale_state["blue"])
            red_scale = float(scale_state["red"])
            with open(scale_path, "w") as f:
                f.write(f"blue_scale {blue_scale}\n")
                f.write(f"red_scale {red_scale}\n")
            return blue_scale, red_scale

        blue_scale = float(scale_state["blue"])
        red_scale = float(scale_state["red"])

        print("\nRescale options:")
        print("  [a] Approve current scaling and continue")
        print("  [1] Multiply RED by factor")
        print("  [2] Multiply BLUE by factor")
        print("  [3] Set RED scale (absolute)")
        print("  [4] Set BLUE scale (absolute)")
        print("  [q] Abort pipeline")
        print("  [Enter] Re-open plot with same scales")
        choice = input("Choice: ").strip()

        if choice.lower() == "a":
            with open(scale_path, "w") as f:
                f.write(f"blue_scale {blue_scale}\n")
                f.write(f"red_scale {red_scale}\n")
            return blue_scale, red_scale
        elif choice.lower() == "q":
            raise RuntimeError("User aborted join rescale approval (terminal 'q').")
        elif choice == "1":
            fac = float(input(f"Multiply RED by factor (current {red_scale}): ").strip())
            red_scale *= fac
        elif choice == "2":
            fac = float(input(f"Multiply BLUE by factor (current {blue_scale}): ").strip())
            blue_scale *= fac
        elif choice == "3":
            red_scale = float(input(f"Set RED scale (current {red_scale}): ").strip())
        elif choice == "4":
            blue_scale = float(input(f"Set BLUE scale (current {blue_scale}): ").strip())
        else:
            pass
