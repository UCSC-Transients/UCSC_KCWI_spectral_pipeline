import json

from kcwi_pipeline.object_workflow import (
    _cr_cleaned_path,
    _cr_mask_path,
    _load_extraction_state,
    _write_extraction_state,
)


def test_invalid_extraction_state_is_replaced_atomically(tmp_path) -> None:
    state_path = tmp_path / "extraction_state.json"
    state_path.write_text('{"sides": {"BLUE": {"broken": ', encoding="utf-8")

    state = _load_extraction_state(state_path)
    assert state == {}

    replacement = {"sides": {"BLUE": {"complete": True}}}
    _write_extraction_state(state_path, replacement)

    with open(state_path, "r", encoding="utf-8") as f:
        assert json.load(f) == replacement
    assert not state_path.with_name(".extraction_state.json.tmp").exists()


def test_cr_products_are_stored_beside_original_cube(tmp_path) -> None:
    source_path = tmp_path / "BLUE" / "KB.test_icubes.fits"

    assert _cr_cleaned_path(tmp_path, "BLUE", source_path) == (
        tmp_path / "BLUE" / "KB.test_icubes_crclean.fits"
    )
    assert _cr_mask_path(tmp_path, "BLUE", source_path) == (
        tmp_path / "BLUE" / "KB.test_icubes_crmask.fits"
    )
