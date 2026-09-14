import matplotlib.pyplot as plt
import numpy as np
import pytest

from kcwi_pipeline import apertures as aperture_module
from kcwi_pipeline.config import ApertureShape, TargetBackgroundApertures


def _apertures() -> TargetBackgroundApertures:
    return TargetBackgroundApertures(
        target=ApertureShape("circle", (3.0, 3.0, 2.0)),
        background=ApertureShape("circle_annulus", (3.0, 3.0, 4.0, 6.0)),
    )


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
