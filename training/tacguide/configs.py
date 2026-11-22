"""Experiment names -> training configs.

This is the only file that knows how an experiment name maps onto the bundled
training code, so a different registry for gc_A / gc_B only needs a change here.

  gc_A        vision + tactile teacher   outputs/training/openvtla/gc/gc_A
  gc_B        vision-only student        outputs/training/pi05/gc/gc_B
  gc_C        distilled student, w = 1   outputs/training/pi05/gc/gc_C
  gc_D_w<w>   extrapolated student       outputs/training/pi05/gc/gc_D_w2, gc_D_w1.5, ...
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

# training/run.py points HF_LEROBOT_HOME at data/training/lerobot when it is imported.
# lerobot reads that variable once, at its own import, so `run` has to come first:
# every entry point imports this module before anything from openpi.
import run  # noqa: E402

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


def base_config(name: str, *, exp_name: str | None = None, resume: bool = False, num_workers: int = 2):
    """Training config of a teacher (gc_A) or the plain student (gc_B), as `training/run.py` builds it.

    Runs live in outputs/training/<model>/<task>/<exp_name>; the run is named after
    the experiment unless `exp_name` says otherwise.
    """
    experiment = parse_name(name)
    if experiment.is_student:
        raise ValueError(f"{name} is a distilled student; use student_config()")
    args = argparse.Namespace(
        task=experiment.task,
        model=BASE_MODELS[experiment.kind],
        exp_name=exp_name or name,
        resume=resume,
        num_workers=num_workers,
    )
    return run.build_config(args)


def student_config(
    name: str,
    *,
    exp_name: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
    num_workers: int = 2,
    wandb: bool = False,
):
    """Config for gc_C / gc_D_w*.

    A student is a vision-only pi0.5 run: it is gc_B's config under another run name
    (outputs/training/pi05/<task>/<name>), so steps, batch size, learning rate, initial
    weights and model always track B, and `training/serve.py --model pi05 --exp-name
    <name>` serves it like any other vision-only run.
    Only the data pipeline is gc_A's, because the frozen teacher needs the tactile
    clips; the student and teacher B only ever see the camera images picked out of it.
    """
    experiment = parse_name(name)
    if not experiment.is_student:
        raise ValueError(f"{name} is not a distilled student")
    weak = base_config(f"{experiment.task}_B", exp_name=exp_name or name, num_workers=num_workers)
    strong = base_config(f"{experiment.task}_A", num_workers=num_workers)
    assets = strong.data.assets
    if assets.assets_dir is None:
        # Read the statistics the teacher was trained with (checked equal to B's at start-up).
        assets = dataclasses.replace(assets, assets_dir=str(strong.assets_dirs))
    return dataclasses.replace(
        weak,
        project_name=WANDB_PROJECT,
        data=dataclasses.replace(strong.data, assets=assets),
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=wandb,
    )


def run_location(name: str, exp_name: str | None = None) -> tuple[str, str, str]:
    """(task, model, run name) under which an experiment's checkpoints live."""
    experiment = parse_name(name)
    model = BASE_MODELS["B" if experiment.is_student else experiment.kind]
    return experiment.task, model, exp_name or name


def latest_checkpoint(checkpoint_dir) -> pathlib.Path | None:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    steps = [int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return checkpoint_dir / str(max(steps)) if steps else None


def create_policy(name: str, *, exp_name: str | None = None, checkpoint_dir=None, step: int | None = None):
    """Serving policy for a trained experiment, built by `training/serve.py`.

    Returns (policy, norm_stats, checkpoint_dir). gc_C and gc_D_w* are ordinary
    vision-only runs and load exactly like gc_B.
    """
    import serve  # training/serve.py

    from openpi.training import checkpoints as _checkpoints

    task, model, run_name = run_location(name, exp_name)
    policy, train_config, checkpoint = serve.load_policy(
        task, model, run_name, step=step, checkpoint_dir=checkpoint_dir
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint) / "assets", data_config.asset_id)
    return policy, norm_stats, pathlib.Path(checkpoint)
