"""Helpers of the training entry point that need no model."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tacguide import configs
from tacguide import train_distill


def test_experiment_names():
    assert configs.parse_name("gc_C").w == 1.0
    assert configs.parse_name("gc_D_w1.5").w == 1.5
    assert configs.parse_name("gc_D_w2").kind == "D"
    assert configs.parse_name("gc_A").w is None and not configs.parse_name("gc_B").is_student
    assert configs.parse_name("in_usb_D_w3").task == "in_usb"
    for bad in ("gc", "gc_D", "gc_D_w", "gc_E", "gc_D_wx"):
        with pytest.raises(ValueError):
            configs.parse_name(bad)


def test_prune_keeps_newest_and_final(tmp_path):
    for step in (1000, 2000, 3000, 4000, 10000):
        (tmp_path / str(step)).mkdir()
    (tmp_path / "wandb_id.txt").write_text("x")
    assert train_distill.prune_checkpoints(tmp_path, keep_last=2, final_step=10000) == [1000, 2000]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["10000", "3000", "4000", "wandb_id.txt"]
    (tmp_path / "5000").mkdir()
    assert train_distill.prune_checkpoints(tmp_path, keep_last=2, final_step=10000) == [3000]


def test_latest_checkpoint(tmp_path):
    assert configs.latest_checkpoint(tmp_path / "missing") is None
    for name in ("5000", "10000", "tmp_15000"):
        (tmp_path / name).mkdir()
    assert configs.latest_checkpoint(tmp_path).name == "10000"
