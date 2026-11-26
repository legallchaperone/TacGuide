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


def test_teacher_step_selects_that_checkpoint(tmp_path):
    import types

    for step in (10000, 20000):
        (tmp_path / str(step)).mkdir()
        (tmp_path / str(step) / "model.safetensors").touch()
    config = types.SimpleNamespace(checkpoint_dir=tmp_path, num_train_steps=10000, name="pi05", pytorch_weight_path="base")
    pick = lambda **kw: train_distill.resolve_teacher_dir(None, config, allow_partial=False, base_standin=False, **kw)  # noqa: E731
    assert pick().name == "10000"            # default: the config final step, not the latest
    assert pick(step=20000).name == "20000"
    with pytest.raises(FileNotFoundError):
        pick(step=15000)


def test_torchrun_command_keeps_the_arguments():
    command = train_distill.torchrun_command(4, ["gc_D_w2", "--steps", "20000", "--nproc", "4"])
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=4" in command and command[-5:] == ["gc_D_w2", "--steps", "20000", "--nproc", "4"]
    assert command[-6].endswith("train_distill.py")


def test_wait_for_calm_waits_only_under_pressure(tmp_path):
    readings = iter([30.0, 12.0, 0.5])
    naps = []
    assert train_distill.wait_for_calm(read=lambda: next(readings), sleep=naps.append) == 30.0
    assert naps == [15.0, 15.0]
    assert train_distill.wait_for_calm(read=lambda: 0.0, sleep=naps.append) == 0.0
    stat = tmp_path / "memory"
    stat.write_text("some avg10=1.00 avg60=2.00 avg300=3.00 total=1\nfull avg10=7.25 avg60=2.00 avg300=3.00 total=1\n")
    assert train_distill.memory_pressure(stat) == 7.25
    assert train_distill.memory_pressure(tmp_path / "missing") == 0.0


def test_load_lock_is_exclusive():
    import threading
    import time

    order = []

    def worker(tag):
        with train_distill.load_lock():
            order.append(("in", tag))
            time.sleep(0.2)
            order.append(("out", tag))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(order[i][0] == "in" and order[i + 1] == ("out", order[i][1]) for i in range(0, 6, 2))
