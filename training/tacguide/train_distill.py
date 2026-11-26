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

The optimisation loop is the runtime's own ``train_pytorch.train_loop``; this script
only swaps in the student model and guards the frozen teachers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
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

    import os

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit("train_distill.py runs on one GPU per job; do not launch it with torchrun")

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else device

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
        logger.info("loading frozen teacher %s (%s) from %s", label, teacher_config.name, directory)
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
        student = DistillStudent(model_config)
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

    original_class = pi0_pytorch.PI0Pytorch
    pi0_pytorch.PI0Pytorch = make_student
    train_pytorch.save_checkpoint = guarded_save
    try:
        train_pytorch.train_loop(config)
    finally:
        pi0_pytorch.PI0Pytorch = original_class
        train_pytorch.save_checkpoint = original_save

    verify_teachers("end of training")
    summary = {
        "name": args.name,
        "w": w,
        "steps": config.num_train_steps,
        "batch_size": config.batch_size,
        "teacher_hashes": hashes,
        "teachers_unchanged": True,
        "base_standin": args.base_standin,
    }
    if torch.cuda.is_available():
        summary["peak_memory_gb"] = {
            f"cuda:{index}": {
                "allocated": round(torch.cuda.max_memory_allocated(index) / 2**30, 2),
                "reserved": round(torch.cuda.max_memory_reserved(index) / 2**30, 2),
            }
            for index in range(torch.cuda.device_count())
        }
    logger.info("summary: %s", json.dumps(summary))
    if args.summary:
        pathlib.Path(args.summary).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
