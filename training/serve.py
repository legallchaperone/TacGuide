"""Serve one trained policy over websocket for the OpenViTac simulation client.

Run from the repository root, for example:

  CUDA_VISIBLE_DEVICES=0 uv run --project training/openpi_runtime python training/serve.py \
      --task gc --model pi05 --exp-name gc_B --port 8000

The vision-only model (pi05) reads observation/image, observation/wrist_image,
observation/state and prompt. OpenVTLA (openvtla) additionally requires
observation/left_tactile_clip and observation/right_tactile_clip, uint8
[4, 224, 224, 3], holding the frames at offsets [-6, -4, -2, 0]; the client
assembles them (see policy/openpi/tactile_history.py).
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import types

TRAINING = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING))

# Cap CPU thread pools before torch starts: on this 384-core machine an uncapped load takes every job down.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import config as tg_config  # noqa: E402
import run as tg_run  # noqa: E402


def checkpoint_dir_for(task: str, model: str, exp_name: str, step: int | None = None) -> Path:
    """Return outputs/training/<model>/<task>/<exp_name>/<step>; the latest step when step is None."""
    exp_dir = tg_config.OUTPUTS / model / task / exp_name
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    if step is None:
        steps = [int(d.name) for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not steps:
            raise FileNotFoundError(f"No checkpoints in {exp_dir}")
        step = max(steps)
    ckpt = exp_dir / str(step)
    if not (ckpt / "model.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint has no model.safetensors: {ckpt}")
    return ckpt


def load_policy(
    task: str,
    model: str,
    exp_name: str,
    *,
    step: int | None = None,
    checkpoint_dir: str | Path | None = None,
    device: str | None = None,
    num_denoise_steps: int | None = None,
):
    """Build the inference policy for one trained model. Returns (policy, train_config, checkpoint_dir).

    The guided-inference server reuses this to load its strong and weak models.
    """
    from openpi.policies import policy_config

    args = types.SimpleNamespace(task=task, model=model, exp_name=exp_name, num_workers=0, resume=False)
    train_config = tg_run.build_config(args)
    ckpt = Path(checkpoint_dir) if checkpoint_dir else checkpoint_dir_for(task, model, exp_name, step)
    sample_kwargs = {"num_steps": num_denoise_steps} if num_denoise_steps else None
    policy = policy_config.create_trained_policy(
        train_config, ckpt, sample_kwargs=sample_kwargs, pytorch_device=device
    )
    return policy, train_config, ckpt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--model", required=True, choices=tg_config.MODELS)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--step", type=int, help="Checkpoint step; defaults to the latest")
    parser.add_argument("--checkpoint-dir", help="Serve this checkpoint directory instead of --exp-name/--step")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", help="Torch device, e.g. cuda or cuda:0; defaults to cuda when available")
    parser.add_argument("--num-denoise-steps", type=int, help="Flow-matching steps at inference (model default 10)")
    return parser.parse_args()


def main():
    import torch

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.serving import websocket_policy_server

    policy, train_config, ckpt = load_policy(
        args.task,
        args.model,
        args.exp_name,
        step=args.step,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        num_denoise_steps=args.num_denoise_steps,
    )
    metadata = {
        "task": args.task,
        "model": args.model,
        "exp_name": args.exp_name,
        "checkpoint": str(ckpt),
        "action_horizon": train_config.model.action_horizon,
        "needs_tactile_clips": args.model == "openvtla",
    }
    logging.info("Serving %s on %s:%d", ckpt, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
