"""CPU checks for the client-side tactile clip assembly."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tactile_history", ROOT / "policy/openpi/tactile_history.py")
th = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(th)


def obs_at(step):
    return {
        "observation/state": np.zeros(8, np.float32),
        th.LEFT_IMAGE_KEY: np.full((4, 4, 3), step, np.uint8),
        th.RIGHT_IMAGE_KEY: np.full((4, 4, 3), 100 + step, np.uint8),
        "prompt": "grasp",
    }


def frame_ids(clip):
    return [int(frame[0, 0, 0]) for frame in clip]


def test_clip_matches_training_offsets_and_start_padding():
    history = th.TactileHistory((-6, -4, -2, 0))
    for step in range(12):
        out = history.push(obs_at(step))
        expected = [max(step + offset, 0) for offset in (-6, -4, -2, 0)]
        assert frame_ids(out[th.LEFT_CLIP_KEY]) == expected
        assert frame_ids(out[th.RIGHT_CLIP_KEY]) == [100 + e for e in expected]
        clip = out[th.LEFT_CLIP_KEY]
        assert clip.shape == (4, 4, 4, 3) and clip.dtype == np.uint8 and clip.flags.c_contiguous
        assert th.LEFT_IMAGE_KEY not in out and th.RIGHT_IMAGE_KEY not in out
        assert out["prompt"] == "grasp"


def test_reset_starts_a_new_episode():
    history = th.TactileHistory()
    for step in range(9):
        history.push(obs_at(step))
    history.reset()
    out = history.push(obs_at(50))
    assert frame_ids(out[th.LEFT_CLIP_KEY]) == [50, 50, 50, 50]


def test_input_is_not_modified_and_bad_inputs_fail():
    history = th.TactileHistory()
    obs = obs_at(3)
    history.push(obs)
    assert th.LEFT_IMAGE_KEY in obs
    with pytest.raises(KeyError):
        history.push({"prompt": "x"})
    with pytest.raises(ValueError):
        th.TactileHistory((-2, 0, 2))
