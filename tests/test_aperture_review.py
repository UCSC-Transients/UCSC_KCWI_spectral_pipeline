import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from astropy.wcs import WCS

from kcwi_pipeline import apertures as aperture_module
from kcwi_pipeline.config import ApertureShape, TargetBackgroundApertures


def _apertures() -> TargetBackgroundApertures:
    return TargetBackgroundApertures(
        target=ApertureShape("circle", (3.0, 3.0, 2.0)),
        background=ApertureShape("circle_annulus", (3.0, 3.0, 4.0, 6.0)),
    )


def _click_widget(widget) -> None:
    canvas = widget.ax.get_figure(root=True).canvas
    canvas.draw()
    x, y = widget.ax.transAxes.transform((0.5, 0.5))
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent("button_press_event", canvas, x, y, button=1),
    )
    canvas.callbacks.process(
        "button_release_event",
        MouseEvent("button_release_event", canvas, x, y, button=1),
    )


def _celestial_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    return wcs


def test_review_window_accepts_without_terminal_prompt(monkeypatch) -> None:
    def fail_prompt(*_args, **_kwargs):
        raise AssertionError("terminal prompt should not be used when Accept is clicked")

    def accept_in_window() -> None:
        fig = plt.gcf()
        assert set(fig._kcwi_aperture_review_widgets) == {
            "accept",
            "move_target",
            "resize_target",
            "shape_target",
            "move_background",
            "resize_background",
            "shape_background",
            "enter_values",
            "cancel",
        }
        fig._kcwi_aperture_review_widgets["accept"]._observers.process(
            "clicked",
            None,
        )

    monkeypatch.setattr(aperture_module, "prompt", fail_prompt)
    monkeypatch.setattr(aperture_module.plt, "show", accept_in_window)

    original = _apertures()
    reviewed = aperture_module.review_apertures(
        np.ones((8, 8), dtype=float),
        original,
        "BLUE saved",
    )

    assert reviewed == original


def test_review_window_reports_cursor_ra_dec_in_sexagesimal(monkeypatch) -> None:
    def inspect_coordinates() -> None:
        fig = plt.gcf()
        canvas = fig.canvas
        canvas.draw()
        image_axis = fig.axes[0]
        x, y = image_axis.transData.transform((0.0, 0.0))
        canvas.callbacks.process(
            "motion_notify_event",
            MouseEvent("motion_notify_event", canvas, x, y),
        )

        coordinate_text = fig._kcwi_coordinate_text.get_text()
        assert "RA   10:00:00.00" in coordinate_text
        assert "Dec  +02:00:00.00" in coordinate_text
        fig._kcwi_aperture_review_actions["accept"]()

    monkeypatch.setattr(aperture_module.plt, "show", inspect_coordinates)
    aperture_module.review_apertures(
        np.ones((8, 8), dtype=float),
        _apertures(),
        "RED WCS",
        celestial_wcs=_celestial_wcs(),
    )


def test_review_window_moves_and_resizes_before_accepting(monkeypatch) -> None:
    def edit_in_window() -> None:
        fig = plt.gcf()
        actions = fig._kcwi_aperture_review_actions
        actions["set_mode"]("target", "move")
        actions["apply_drag"]((3.0, 3.0), (5.0, 4.0))
        actions["set_mode"]("background", "resize")
        actions["apply_drag"]((3.0, 3.0), (11.0, 3.0))
        actions["accept"]()

    monkeypatch.setattr(aperture_module.plt, "show", edit_in_window)
    reviewed = aperture_module.review_apertures(
        np.ones((14, 14), dtype=float),
        _apertures(),
        "RED transformed",
    )

    assert reviewed.target.params == (5.0, 4.0, 2.0)
    assert np.allclose(reviewed.background.params, (3.0, 3.0, 16.0 / 3.0, 8.0))


def test_review_window_changes_shapes_and_accepts_entered_values(monkeypatch) -> None:
    def edit_in_window() -> None:
        fig = plt.gcf()
        widgets = fig._kcwi_aperture_review_widgets
        state = fig._kcwi_aperture_review_state

        widgets["shape_target"]._observers.process("clicked", None)
        target_shapes, apply_shape, _ = state["modal_widgets"]
        target_shapes.set_active(1)  # ellipse
        apply_shape._observers.process("clicked", None)

        widgets["enter_values"]._observers.process("clicked", None)
        *value_boxes, apply_values, _ = state["modal_widgets"]
        for box, value in zip(value_boxes, (5.0, 6.0, 3.0, 1.5, 0.25)):
            box.set_val(str(value))
        apply_values._observers.process("clicked", None)

        widgets["shape_background"]._observers.process("clicked", None)
        background_shapes, apply_shape, _ = state["modal_widgets"]
        background_shapes.set_active(0)  # automatic ellipse annulus
        apply_shape._observers.process("clicked", None)
        widgets["accept"]._observers.process("clicked", None)

    monkeypatch.setattr(aperture_module.plt, "show", edit_in_window)
    reviewed = aperture_module.review_apertures(
        np.ones((14, 14), dtype=float),
        _apertures(),
        "BLUE new",
    )

    assert reviewed.target.shape == "ellipse"
    assert reviewed.target.params == (5.0, 6.0, 3.0, 1.5, 0.25)
    assert reviewed.background.shape == "ellipse_annulus"
    assert reviewed.background.params[:2] == (5.0, 6.0)


def test_value_editor_disables_overlapping_main_buttons(monkeypatch) -> None:
    def edit_in_window() -> None:
        fig = plt.gcf()
        widgets = fig._kcwi_aperture_review_widgets
        state = fig._kcwi_aperture_review_state

        widgets["enter_values"]._observers.process("clicked", None)
        *value_boxes, apply_values, _ = state["modal_widgets"]

        assert all(not button.active for button in widgets.values())
        _click_widget(value_boxes[1])
        assert fig.canvas.mouse_grabber is None
        assert state["active"] == "target"
        assert state["mode"] == "move"

        for box, value in zip(value_boxes, (5.0, 6.0, 3.0)):
            box.set_val(str(value))
        _click_widget(apply_values)

        assert state["modal"] is None
        assert not state["cancelled"]
        assert all(button.active for button in widgets.values())
        _click_widget(widgets["accept"])

    monkeypatch.setattr(aperture_module.plt, "show", edit_in_window)
    reviewed = aperture_module.review_apertures(
        np.ones((14, 14), dtype=float),
        _apertures(),
        "RED new",
    )

    assert reviewed.target.params == (5.0, 6.0, 3.0)


def test_closing_review_window_does_not_open_terminal_approval(monkeypatch) -> None:
    def fail_prompt(*_args, **_kwargs):
        raise AssertionError("the removed terminal approval prompt must not be used")

    def close_window() -> None:
        plt.close(plt.gcf())

    monkeypatch.setattr(aperture_module.plt, "show", close_window)
    monkeypatch.setattr(aperture_module, "prompt", fail_prompt)

    with pytest.raises(RuntimeError, match="closed without accepting"):
        aperture_module.review_apertures(
            np.ones((8, 8), dtype=float),
            _apertures(),
            "BLUE saved",
        )
