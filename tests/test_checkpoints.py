import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "custom_nodes"))

from h3_continuity.checkpoints import (  # noqa: E402
    H3ConditioningLoad,
    H3ConditioningSave,
    load_conditioning,
)


def conditioning():
    return [
        [
            torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 7,
            {
                "mask": torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
                "minimax_refs": [
                    {
                        "kind": "audio",
                        "ref_audio_t": 5,
                        "audio_latent": torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4),
                        "nested": (None, True, -0.0, 2**60),
                    }
                ],
                "minimax_keyframes": (
                    {"resolved_frame_index": 17.0, "latent": torch.ones(1, 24, 1, 2, 2)},
                ),
            },
        ]
    ]


def test_conditioning_round_trip_keeps_nested_types_tensor_dtype_and_settings(tmp_path):
    positive = conditioning()
    negative = [[torch.zeros(1, 2, 3, dtype=torch.float16), {"scale": 0.125}]]
    settings = r'{"seed": 9007199254740993, "cfg": 1.0, "label": "a \"take\""}'
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)):
        saved, report = H3ConditioningSave().save(
            positive, "chain/take.safetensors", settings, negative=negative
        )
        restored_positive, restored_negative, restored_settings = load_conditioning(saved)
        node_positive, node_negative, node_settings, load_report = H3ConditioningLoad().load(saved)

    assert saved == "chain/take.safetensors"
    assert "Saved" in report and "Loaded" in load_report
    assert restored_settings == settings == node_settings
    assert isinstance(restored_positive, list)
    assert isinstance(restored_positive[0], list)
    meta = restored_positive[0][1]
    assert isinstance(meta["minimax_keyframes"], tuple)
    assert isinstance(meta["minimax_refs"][0]["nested"], tuple)
    assert meta["minimax_refs"][0]["nested"] == (None, True, -0.0, 2**60)
    for actual, expected in (
        (restored_positive[0][0], positive[0][0]),
        (meta["mask"], positive[0][1]["mask"]),
        (meta["minimax_refs"][0]["audio_latent"], positive[0][1]["minimax_refs"][0]["audio_latent"]),
        (meta["minimax_keyframes"][0]["latent"], positive[0][1]["minimax_keyframes"][0]["latent"]),
        (restored_negative[0][0], negative[0][0]),
        (node_positive[0][0], positive[0][0]),
        (node_negative[0][0], negative[0][0]),
    ):
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


def test_missing_negative_loads_as_empty_conditioning_and_save_avoids_collisions(tmp_path):
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)):
        first, _ = H3ConditioningSave().save(conditioning(), "take.safetensors")
        second, _ = H3ConditioningSave().save(conditioning(), "take.safetensors")
        _, negative, _, _ = H3ConditioningLoad().load(first)
    assert first == "take.safetensors"
    assert second == "take_2.safetensors"
    assert negative == []
    assert (tmp_path / first).stat().st_size > 0
    assert (tmp_path / second).stat().st_size > 0


@pytest.mark.parametrize("filename", ["../escape.safetensors", "wrong.json", "nested/../../escape.safetensors"])
def test_save_rejects_paths_outside_output_or_wrong_extension(tmp_path, filename):
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)):
        with pytest.raises(ValueError):
            H3ConditioningSave().save(conditioning(), filename)


@pytest.mark.parametrize("settings", ["[]", "null", "{", '{"value": NaN}'])
def test_save_requires_valid_finite_json_object_settings(tmp_path, settings):
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)):
        with pytest.raises(ValueError):
            H3ConditioningSave().save(conditioning(), "take.safetensors", settings)


def test_save_rejects_objects_outside_safe_conditioning_value_set(tmp_path):
    invalid = [[torch.zeros(1), {"not_supported": object()}]]
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)):
        with pytest.raises(ValueError, match="Unsupported conditioning value"):
            H3ConditioningSave().save(invalid, "take.safetensors")


def test_load_rejects_non_conditioning_safetensors(tmp_path):
    import safetensors.torch

    safetensors.torch.save_file({"tensor": torch.ones(1)}, str(tmp_path / "wrong.safetensors"))
    with patch("folder_paths.get_output_directory", return_value=str(tmp_path)):
        with pytest.raises(ValueError, match="not a supported"):
            load_conditioning("wrong.safetensors")
