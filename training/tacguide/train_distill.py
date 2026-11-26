"""Train a distilled (gc_C) or extrapolated (gc_D_w*) vision-only student.

From the repository root, on one GPU:

    $TG/venv/bin/python training/tacguide/train_distill.py gc_C
    $TG/venv/bin/python training/tacguide/train_distill.py gc_D_w2
    ... --resume                  # continue from the student's latest checkpoint

The teachers default to the latest checkpoint of gc_A / gc_B (``--teacher-exp``,
``--weak-exp`` pick the run, ``--teacher-dir`` / ``--weak-dir`` give a directory).
gc_C (w = 1) needs only A.

Checks from the plan:

    # teacher hashes before/after 10 steps (also verified at every checkpoint save)
    ... gc_D_w2 --exp-name hashcheck --steps 10 --no-save --no-wandb
    # memory: batch 64 for 100 steps, base weights standing in for A and B
    ... gc_D_w2 --exp-name memtest --steps 100 --no-save --no-wandb --base-standin

Several cards: add ``--nproc 4`` (with CUDA_VISIBLE_DEVICES naming four cards). The script
re-launches itself as four DDP workers; each keeps its own frozen teachers on its own card,
the batch of 64 is split 16 per worker, and model loads are serialised across workers (and
wait while the machine is under memory pressure).

The optimisation loop is the runtime's own ``train_pytorch.train_loop``; this script
only swaps in the student model and guards the frozen teachers.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import sys

_TRAINING_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

logger = logging.getLogger("tacguide")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="gc_C or gc_D_w<number>, e.g. gc_D_w2")
    parser.add_argument("--exp-name", help="run name; defaults to the experiment name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--teacher-exp", help="run name of gc_A (default gc_A)")
    parser.add_argument("--weak-exp", help="run name of gc_B (default gc_B)")
    parser.add_argument("--teacher-dir", help="checkpoint directory of A (contains model.safetensors)")
    parser.add_argument("--weak-dir", help="checkpoint directory of B (contains model.safetensors)")
    parser.add_argument("--allow-partial-teacher", action="store_true", help="accept a teacher that is not at its final step")
    parser.add_argument("--base-standin", action="store_true", help="use the pi0.5 base weights for A and B (tests only)")
    parser.add_argument("--teacher-device", help="put the frozen teachers on another device, e.g. cuda:1")
    parser.add_argument("--nproc", type=int, default=1,
                        help="train on this many GPUs (one process per card, each with its own copy of the teachers)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, help="number of training steps (default: the released value for the task)")
    parser.add_argument("--teacher-step", type=int, help="checkpoint step of A to distil from (default: its final step)")
    parser.add_argument("--weak-step", type=int, help="checkpoint step of B to use (default: its final step)")
    parser.add_argument("--batch-size", type=int, help="override the batch size (tests only)")
    parser.add_argument("--save-interval", type=int, help="override the checkpoint interval")
    parser.add_argument("--keep-last", type=int, default=2, help="checkpoints to keep besides the final one; 0 keeps all")
    parser.add_argument("--no-save", action="store_true", help="do not write checkpoints (tests only)")
    parser.add_argument("--wandb", action="store_true", help="log to the TacGuide wandb project (off by default; logs go to stdout)")
    parser.add_argument("--no-wandb", action="store_true", help=argparse.SUPPRESS)  # accepted for older command lines
    parser.add_argument("--summary", help="write a JSON summary (hashes, peak memory) to this path")
    return parser.parse_args(argv)


def load_teacher(train_config, checkpoint_dir, device, *, base_standin: bool = False):
    """Load a frozen teacher the way the serving path does (bfloat16 backbone, eval mode)."""
    import safetensors.torch

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    from tacguide.distill import freeze

    weight_path = pathlib.Path(checkpoint_dir) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"teacher weights not found: {weight_path}")
    model = PI0Pytorch(train_config.model)
    missing, unexpected = safetensors.torch.load_model(model, str(weight_path), strict=False)
    if base_standin:
        missing = [name for name in missing if "anytouch2_encoder" not in name]
    if missing or unexpected:
        raise RuntimeError(
            f"{weight_path} does not match config '{train_config.name}': "
            f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return freeze(model.to(device))


def resolve_teacher_dir(explicit, train_config, *, allow_partial: bool, base_standin: bool, step: int | None = None):
    from tacguide import configs

    if base_standin:
        return pathlib.Path(train_config.pytorch_weight_path)
    if explicit:
        return pathlib.Path(explicit)
    if step is not None:
        chosen = pathlib.Path(train_config.checkpoint_dir) / str(step)
        if not (chosen / "model.safetensors").is_file():
            raise FileNotFoundError(f"checkpoint {chosen} does not exist (yet)")
        return chosen
    # The teacher is the checkpoint at the config's final step, even if the run was
    # later continued beyond it (e.g. gc_B trained on to 20k for another experiment).
    final = pathlib.Path(train_config.checkpoint_dir) / str(train_config.num_train_steps)
    if (final / "model.safetensors").is_file():
        return final
    latest = configs.latest_checkpoint(train_config.checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {train_config.checkpoint_dir}; pass the directory explicitly")
    if int(latest.name) != train_config.num_train_steps and not allow_partial:
        raise RuntimeError(
            f"latest checkpoint of '{train_config.name}' is step {latest.name}, not the final step "
            f"{train_config.num_train_steps}; wait for it to finish or pass --allow-partial-teacher"
        )
    return latest


def torchrun_command(nproc: int, argv: list[str]) -> list[str]:
    """Command that re-runs this script as `nproc` DDP workers on one node."""
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={nproc}", str(pathlib.Path(__file__).resolve()), *argv]


def memory_pressure(path="/proc/pressure/memory") -> float:
    """Share of the last 10 s in which every task was stalled on memory, in percent (0 if unknown)."""
    try:
        for line in pathlib.Path(path).read_text().splitlines():
            if line.startswith("full"):
                return float(line.split("avg10=")[1].split()[0])
    except (OSError, IndexError, ValueError):
        pass
    return 0.0


def wait_for_calm(limit: float = 2.0, timeout: float = 900.0, read=memory_pressure, sleep=None, log=None) -> float:
    """Block while the machine is stalling on memory, so a model load does not add to it."""
    import time

    sleep = sleep or time.sleep
    waited = 0.0
    while waited < timeout:
        value = read()
        if value <= limit:
            break
        if log:
            log(f"memory pressure {value:.1f}% > {limit}%; waiting before loading a model")
        sleep(15.0)
        waited += 15.0
    return waited


@contextlib.contextmanager
def load_lock():
    """One model load at a time across all workers of this user.

    On this machine several simultaneous multi-GB loads have pushed memory pressure high
    enough for the memory daemon to kill every running job (2026-11-25).
    """
    import fcntl
    import tempfile

    path = pathlib.Path(tempfile.gettempdir()) / f"tacguide_load_{os.getuid()}.lock"
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        wait_for_calm(log=logger.info)
        yield


def prune_checkpoints(checkpoint_dir, *, keep_last: int, final_step: int) -> list[int]:
    """Delete this run's older step directories, keeping the newest `keep_last` and the final one."""
    import shutil

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    steps = sorted(int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit())
    removed = [step for step in steps if step != final_step][:-keep_last]
    for step in removed:
        shutil.rmtree(checkpoint_dir / str(step))
    return removed


def check_norm_stats(student_config, teacher_configs_and_dirs):
    """All models must share one normalised state/action space, or v_A - v_B is meaningless."""
    from openpi.training import checkpoints as _checkpoints

    from tacguide import velocity

    student_data = student_config.data.create(student_config.assets_dirs, student_config.model)
    if student_data.norm_stats is None:
        raise RuntimeError("the student has no normalisation statistics; run the official `stats` step first")
    for label, config, directory in teacher_configs_and_dirs:
        data = config.data.create(config.assets_dirs, config.model)
        stats = _checkpoints.load_norm_stats(pathlib.Path(directory) / "assets", data.asset_id)
        same, why = velocity.norm_stats_equal(student_data.norm_stats, stats)
        if not same:
            raise RuntimeError(f"normalisation of teacher {label} ({directory}) differs from the student's: {why}")
        logger.info("norm stats of teacher %s match the student's", label)


def main(argv=None):
    args = parse_args(argv)

    from tacguide import configs

    experiment = configs.parse_name(args.name)
    if not experiment.is_student:
        raise SystemExit(f"{args.name} is not a distilled student; train A and B with training/run.py")
    w = experiment.w

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if args.nproc > 1 and world_size == 1:
        if args.teacher_device:
            raise SystemExit("--teacher-device cannot be combined with --nproc: every worker keeps its teachers on its own card")
        command = torchrun_command(args.nproc, list(sys.argv[1:] if argv is None else argv))
        os.execv(command[0], command)
    if world_size > 1 and args.teacher_device:
        raise SystemExit("--teacher-device cannot be used with several workers")

    import torch

    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
    import train_pytorch

    from tacguide.distill import DistillStudent

    train_pytorch.init_logging()

    config = configs.student_config(
        args.name,
        exp_name=args.exp_name,
        resume=args.resume,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        wandb=args.wandb and not args.no_wandb,
    )
    overrides = {}
    if args.steps is not None:
        overrides["num_train_steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.save_interval is not None:
        overrides["save_interval"] = args.save_interval
    if overrides:
        config = dataclasses.replace(config, **overrides)

    strong_config = configs.base_config(f"{experiment.task}_A", exp_name=args.teacher_exp)
    weak_config = configs.base_config(f"{experiment.task}_B", exp_name=args.weak_exp)
    need_strong, need_weak = w != 0.0, w != 1.0

    # Under DDP every worker owns one card (the trainer uses cuda:<local rank> too) and keeps
    # its own frozen teachers there; the batch is split across workers by the trainer.
    device = torch.device(f"cuda:{rank if world_size > 1 else 0}" if torch.cuda.is_available() else "cpu")
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else device
    is_main = rank == 0
    # The student's initial weights are loaded here, one worker at a time, instead of by the trainer.
    base_weight_path = config.pytorch_weight_path
    config = dataclasses.replace(config, pytorch_weight_path=None)

    teachers, to_check = {}, []
    for label, needed, teacher_config, explicit, step in (
        ("strong", need_strong, strong_config, args.teacher_dir, args.teacher_step),
        ("weak", need_weak, weak_config, args.weak_dir, args.weak_step),
    ):
        if not needed:
            continue
        directory = resolve_teacher_dir(
            explicit, teacher_config, allow_partial=args.allow_partial_teacher, base_standin=args.base_standin, step=step
        )
        with load_lock():
            logger.info("worker %d: loading frozen teacher %s (%s) from %s", rank, label, teacher_config.name, directory)
            teachers[label] = load_teacher(teacher_config, directory, teacher_device, base_standin=args.base_standin)
        to_check.append((label, teacher_config, directory))
    if args.base_standin:
        logger.warning("BASE STAND-IN: teachers are the pi0.5 base weights; this run is only a memory/plumbing test")
    else:
        check_norm_stats(config, to_check)

    if w == 1.0:
        logger.info("w = 1: target is v_A; teacher B is not loaded")

    hashes = {}
    students = []

    def make_student(model_config):
        import safetensors.torch

        with load_lock():
            student = DistillStudent(model_config)
            if not args.resume and base_weight_path is not None:
                logger.info("worker %d: loading the student's initial weights from %s", rank, base_weight_path)
                safetensors.torch.load_model(student, os.path.join(base_weight_path, "model.safetensors"), strict=True)
        student.attach_teachers(
            strong=teachers.get("strong"),
            weak=teachers.get("weak"),
            w=w,
            batch_image_keys=tuple(strong_config.model.image_keys),
        )
        if not hashes:
            hashes.update(student.teacher_hashes())
            logger.info("teacher parameter hashes at start: %s", hashes)
        students.append(student)
        return student

    def verify_teachers(when: str):
        current = students[-1].teacher_hashes()
        if current != hashes:
            raise RuntimeError(f"frozen teacher parameters changed ({when}): {hashes} -> {current}")
        logger.info("teacher parameter hashes unchanged (%s)", when)

    original_save = train_pytorch.save_checkpoint

    def guarded_save(model, optimizer, global_step, cfg, is_main, data_config):
        saving = (global_step % cfg.save_interval == 0 and global_step > 0) or global_step == cfg.num_train_steps
        if saving and is_main:
            verify_teachers(f"step {global_step}")
        if not args.no_save:
            original_save(model, optimizer, global_step, cfg, is_main, data_config)
            if saving and is_main and args.keep_last > 0:
                prune_checkpoints(cfg.checkpoint_dir, keep_last=args.keep_last, final_step=cfg.num_train_steps)

    original_load = train_pytorch.load_checkpoint

    def locked_load(*load_args, **load_kwargs):
        with load_lock():
            return original_load(*load_args, **load_kwargs)

    original_class = pi0_pytorch.PI0Pytorch
    pi0_pytorch.PI0Pytorch = make_student
    train_pytorch.save_checkpoint = guarded_save
    train_pytorch.load_checkpoint = locked_load
    try:
        train_pytorch.train_loop(config)
    finally:
        pi0_pytorch.PI0Pytorch = original_class
        train_pytorch.save_checkpoint = original_save
        train_pytorch.load_checkpoint = original_load

    verify_teachers("end of training")
    summary = {
        "name": args.name,
        "w": w,
        "steps": config.num_train_steps,
        "batch_size": config.batch_size,
        "teacher_hashes": hashes,
        "teachers_unchanged": True,
        "base_standin": args.base_standin,
        "workers": world_size,
        "worker": rank,
    }
    if torch.cuda.is_available():
        indices = [device.index] if world_size > 1 else range(torch.cuda.device_count())
        summary["peak_memory_gb"] = {
            f"cuda:{index}": {
                "allocated": round(torch.cuda.max_memory_allocated(index) / 2**30, 2),
                "reserved": round(torch.cuda.max_memory_reserved(index) / 2**30, 2),
            }
            for index in indices
        }
    logger.info("summary: %s", json.dumps(summary))
    if args.summary:
        path = pathlib.Path(args.summary)
        if not is_main:
            path = path.with_name(f"{path.stem}.worker{rank}{path.suffix}")
        path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
