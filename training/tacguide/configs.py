"""Experiment names -> training configs.

This is the only file that knows how an experiment name maps onto the bundled
training code, so a different registry for gc_A / gc_B only needs a change here.

  gc_A        vision + tactile teacher   = official `--task gc --model openvtla`
  gc_B        vision-only student        = official `--task gc --model pi05`
  gc_C        distilled student, w = 1
  gc_D_w<w>   extrapolated student, e.g. gc_D_w1.5, gc_D_w2
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

_TRAINING_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

WANDB_PROJECT = "TacGuide"
BASE_MODELS = {"A": "openvtla", "B": "pi05"}
_NAME = re.compile(r"^(?P<task>[a-z0-9_]+?)_(?P<kind>A|B|C|D_w(?P<w>\d+(?:\.\d+)?))$")


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    kind: str  # "A", "B", "C" or "D"
    w: float | None  # None for A and B

    @property
    def is_student(self) -> bool:
        return self.kind in ("C", "D")


def parse_name(name: str) -> Experiment:
    match = _NAME.match(name)
    if not match:
        raise ValueError(f"'{name}' is not an experiment name; expected gc_A, gc_B, gc_C or gc_D_w<number>")
    kind = match["kind"]
    if kind == "C":
        return Experiment(name, match["task"], "C", 1.0)
    if kind.startswith("D"):
        return Experiment(name, match["task"], "D", float(match["w"]))
    return Experiment(name, match["task"], kind, None)


def base_config(name: str, *, exp_name: str = "run1", resume: bool = False, num_workers: int = 2):
    """Training config of a teacher (gc_A) or the plain student (gc_B), as the official entry point builds it."""
    import run  # training/run.py; also sets HF_LEROBOT_HOME and the runtime script path

    experiment = parse_name(name)
    if experiment.is_student:
        raise ValueError(f"{name} is a distilled student; use student_config()")
    args = argparse.Namespace(
        task=experiment.task,
        model=BASE_MODELS[experiment.kind],
        exp_name=exp_name,
        resume=resume,
        num_workers=num_workers,
    )
    return run.build_config(args)


def student_config(
    name: str,
    *,
    exp_name: str = "run1",
    resume: bool = False,
    overwrite: bool = False,
    num_workers: int = 2,
    wandb: bool = True,
):
    """Config for gc_C / gc_D_w*.

    Everything that defines the optimisation (steps, batch size, learning rate,
    initial weights, model) is taken from gc_B, so the student always tracks B.
    The data pipeline is gc_A's, because the frozen teacher needs the tactile clips;
    the student and teacher B only ever see the camera images picked out of that batch.
    """
    from openpi.training import config as _config

    experiment = parse_name(name)
    if not experiment.is_student:
        raise ValueError(f"{name} is not a distilled student")
    weak = base_config(f"{experiment.task}_B", exp_name=exp_name, num_workers=num_workers)
    strong = base_config(f"{experiment.task}_A", exp_name=exp_name, num_workers=num_workers)
    assets = strong.data.assets
    if assets.assets_dir is None:
        # The student has its own config name; keep reading the teacher's statistics.
        assets = dataclasses.replace(assets, assets_dir=str(strong.assets_dirs))
    return dataclasses.replace(
        weak,
        name=name,
        exp_name=exp_name,
        project_name=WANDB_PROJECT,
        data=dataclasses.replace(strong.data, assets=assets),
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=wandb,
    )


def latest_checkpoint(checkpoint_dir) -> pathlib.Path | None:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    steps = [int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return checkpoint_dir / str(max(steps)) if steps else None


def create_policy(name: str, checkpoint_dir, *, exp_name: str = "run1"):
    """Build the serving policy for a trained experiment, plus its norm stats.

    gc_C and gc_D_w* are served with gc_B's config: their checkpoints are ordinary
    vision-only checkpoints.
    """
    from openpi.policies import policy_config
    from openpi.training import checkpoints as _checkpoints

    experiment = parse_name(name)
    config_name = f"{experiment.task}_B" if experiment.is_student else name
    train_config = base_config(config_name, exp_name=exp_name)
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return policy, norm_stats
