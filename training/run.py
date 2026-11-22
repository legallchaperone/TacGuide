"""Convert simulation episodes, compute statistics, or train a policy."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

from config import ACTION_HORIZON, BATCH_SIZE, DATA, MODELS, OUTPUTS, ROOT, RUNTIME, TASKS, WEIGHTS, steps_for

os.environ["HF_LEROBOT_HOME"] = str(DATA / "lerobot")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(RUNTIME / "scripts"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("convert", "stats", "train", "show"))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", choices=MODELS, default="openvtla")
    parser.add_argument("--exp-name", default="run1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-prompt", help="Override each episode's instruction during conversion")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, help="Limit statistics for debugging")
    # TacGuide overrides. Leaving them unset keeps the released OpenViTac defaults.
    parser.add_argument("--steps", type=int, help="Override the number of training steps")
    parser.add_argument("--save-interval", type=int, help="Override the checkpoint interval in steps")
    parser.add_argument("--batch-size", type=int, help="Override the global batch size (smoke tests)")
    parser.add_argument("--log-interval", type=int, help="Override the logging interval in steps")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb (project from --wandb-project)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "TacGuide"))
    parser.add_argument("--freeze-anytouch", action="store_true",
                        help="Freeze the AnyTouch2 encoder; its projector and the VLA still train")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing experiment directory")
    return parser.parse_args()


def validate(args):
    if args.exp_name in ("", ".", "..") or Path(args.exp_name).name != args.exp_name:
        raise ValueError("Experiment name must be a single directory name")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    if args.max_frames is not None and args.max_frames < BATCH_SIZE:
        raise ValueError("max-frames must be at least the batch size")
    if args.resume and args.mode != "train":
        raise ValueError("--resume is only valid for training")
    for name in ("steps", "save_interval", "batch_size", "log_interval"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def build_config(args):
    from openpi.policies import sim_policy
    from openpi.training import config as cfg

    base = cfg.make_anytouch2_config(
        args.task,
        anytouch_checkpoint_path=str(WEIGHTS / "anytouch2" / "checkpoint-4frames.pth"),
        pytorch_weight_path=str(WEIGHTS / "pi05_base_pytorch"),
    )
    if getattr(args, "freeze_anytouch", False):
        base = dataclasses.replace(base, model=dataclasses.replace(base.model, anytouch_trainable=False))
    if args.model == "pi05":
        base = dataclasses.replace(
            base,
            model=dataclasses.replace(
                base.model, image_keys=sim_policy.VISION_IMAGE_KEYS, tactile_encoder_type="siglip"
            ),
            data=cfg.LeRobotSimVisionDataConfig(
                repo_id=args.task,
                base_config=cfg.DataConfig(prompt_from_task=True),
                state_dim=8,
                output_action_dim=8,
                delta_joint_actions=True,
            ),
        )
    steps = getattr(args, "steps", None) or steps_for(args.task)
    save_interval = getattr(args, "save_interval", None) or base.save_interval
    return dataclasses.replace(
        base,
        name=args.model,
        project_name=getattr(args, "wandb_project", None) or base.project_name,
        assets_base_dir=str(DATA / "stats"),
        checkpoint_base_dir=str(OUTPUTS),
        exp_name=f"{args.task}/{args.exp_name}",
        batch_size=getattr(args, "batch_size", None) or BATCH_SIZE,
        num_workers=args.num_workers,
        num_train_steps=steps,
        save_interval=save_interval,
        keep_period=save_interval,
        log_interval=getattr(args, "log_interval", None) or base.log_interval,
        resume=args.resume,
        overwrite=getattr(args, "overwrite", False),
        wandb_enabled=getattr(args, "wandb", False),
    )


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def main():
    args = parse_args()
    validate(args)
    if args.mode == "show":
        print(json.dumps({"task": args.task, "model": args.model, "steps": steps_for(args.task),
                          "batch_size": BATCH_SIZE, "action_horizon": ACTION_HORIZON,
                          "hdf5": str(DATA / "hdf5" / args.task),
                          "lerobot": str(DATA / "lerobot" / args.task)}, indent=2))
        return
    if args.mode == "convert":
        command = [sys.executable, str(RUNTIME / "examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py"),
                   "--hdf5-dir", str(DATA / "hdf5" / args.task),
                   "--output-dir", str(DATA / "lerobot" / args.task), "--repo-id", args.task,
                   "--control-mode", "abs_joint"]
        if args.task_prompt:
            command.extend(["--task-prompt", args.task_prompt])
        subprocess.run(command, cwd=ROOT, check=True)
        return
    dataset_dir = DATA / "lerobot" / args.task
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}; run convert first")
    if args.mode == "train":
        require_file(DATA / "stats" / args.model / args.task / "norm_stats.json")
        if args.model == "openvtla":
            require_file(WEIGHTS / "anytouch2" / "checkpoint-4frames.pth")
        if not args.resume:
            require_file(WEIGHTS / "pi05_base_pytorch" / "model.safetensors")
    config = build_config(args)
    if args.mode == "stats":
        from compute_norm_stats import compute
        compute(config, args.max_frames)
    else:
        from train_pytorch import train_loop
        train_loop(config)


if __name__ == "__main__":
    main()
