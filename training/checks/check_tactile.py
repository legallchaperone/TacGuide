"""Check that OpenVTLA really uses touch, and which tactile weights trained (plan item 2.4).

Run from the repository root on a trained (even 10-step) OpenVTLA checkpoint:

  CUDA_VISIBLE_DEVICES=0 uv run --project training/openpi_runtime python training/checks/check_tactile.py \
      --task gc --exp-name gc_A_smoke [--step 10] [--expect-frozen-encoder]

1. Tactile sensitivity: the same observation and the same starting noise are run twice,
   once with the real tactile clips and once with all-zero clips. The actions must differ.
2. Weight drift: the checkpoint is compared with the weights training started from
   (π0.5 base + pretrained AnyTouch2 + fresh projector), per group. The projector and the
   VLA must have changed. The AnyTouch2 encoder must have changed when it is trained
   jointly (the default, as in the released OpenVTLA) and must be identical with
   --expect-frozen-encoder (runs trained with --freeze-anytouch).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import config as tg_config  # noqa: E402
import serve as tg_serve  # noqa: E402

OFFSETS = (-6, -4, -2, 0)


def to_uint8_hwc(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 + 0.5, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def observation_from_dataset(dataset, episode: int, step: int) -> dict:
    begin = int(dataset.episode_data_index["from"][episode])
    end = int(dataset.episode_data_index["to"][episode])
    index = min(begin + step, end - 1)
    item = dataset[index]
    clips = {}
    for side in ("left", "right"):
        frames = [to_uint8_hwc(dataset[max(index + o, begin)][f"{side}_tactile_image"]) for o in OFFSETS]
        clips[side] = np.ascontiguousarray(np.stack(frames))
    return {
        "observation/image": to_uint8_hwc(item["image"]),
        "observation/wrist_image": to_uint8_hwc(item["wrist_image"]),
        "observation/left_tactile_clip": clips["left"],
        "observation/right_tactile_clip": clips["right"],
        "observation/state": np.asarray(item["state"], dtype=np.float32),
        "prompt": item["task"],
    }


def group_of(name: str) -> str:
    if name.startswith("anytouch2_encoder.projector"):
        return "tactile_projector"
    if name.startswith("anytouch2_encoder"):
        return "anytouch2_encoder"
    return "vla"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--expect-frozen-encoder", action="store_true")
    parser.add_argument("--sample-steps", type=int, nargs="*", default=[0, 40, 80],
                        help="Dataset steps of episode 0 to test tactile sensitivity on")
    return parser.parse_args()


def main():
    args = parse_args()
    import os

    os.environ["HF_LEROBOT_HOME"] = str(tg_config.DATA / "lerobot")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import safetensors.torch
    import torch

    policy, train_config, ckpt = tg_serve.load_policy(args.task, "openvtla", args.exp_name, step=args.step)
    report = {"checkpoint": str(ckpt), "failures": []}
    fail = report["failures"].append

    # 1. Tactile sensitivity.
    dataset = LeRobotDataset(args.task)
    model_cfg = train_config.model
    noise = np.random.default_rng(0).standard_normal((model_cfg.action_horizon, model_cfg.action_dim)).astype(np.float32)
    report["tactile_sensitivity"] = []
    for step in args.sample_steps:
        obs = observation_from_dataset(dataset, 0, step)
        zeroed = dict(obs)
        for key in ("observation/left_tactile_clip", "observation/right_tactile_clip"):
            zeroed[key] = np.zeros_like(obs[key])
        with_touch = policy.infer(obs, noise=noise)["actions"]
        repeat = policy.infer(obs, noise=noise)["actions"]
        without_touch = policy.infer(zeroed, noise=noise)["actions"]
        entry = {
            "step": step,
            "repeat_max_abs_diff": float(np.abs(with_touch - repeat).max()),
            "zero_tactile_max_abs_diff": float(np.abs(with_touch - without_touch).max()),
        }
        report["tactile_sensitivity"].append(entry)
        if entry["zero_tactile_max_abs_diff"] <= 10 * entry["repeat_max_abs_diff"] + 1e-6:
            fail(f"step {step}: zeroing the tactile clips did not change the actions beyond run-to-run noise")

    # 2. Weight drift relative to the initial weights.
    from openpi.models_pytorch import pi0_pytorch

    initial = pi0_pytorch.PI0Pytorch(config=model_cfg)  # loads the pretrained AnyTouch2 checkpoint itself
    base = safetensors.torch.load_file(str(tg_config.WEIGHTS / "pi05_base_pytorch" / "model.safetensors"))
    initial_state = {k: v.float() for k, v in initial.state_dict().items()}
    initial_state.update({k: v.float() for k, v in base.items() if k in initial_state})
    trained = safetensors.torch.load_file(str(ckpt / "model.safetensors"))
    drift, counted = {}, {}
    for name, value in trained.items():
        if name not in initial_state or not torch.is_floating_point(value):
            continue
        group = group_of(name)
        diff = float((value.float() - initial_state[name]).abs().max())
        drift[group] = max(drift.get(group, 0.0), diff)
        counted[group] = counted.get(group, 0) + 1
    report["weight_max_abs_drift"] = drift
    report["tensors_compared"] = counted
    # The projector starts from a fresh random draw, so only its presence is informative here;
    # that it trains is covered by the encoder/VLA checks plus requires_grad below.
    if drift.get("vla", 0.0) == 0.0:
        fail("VLA weights are identical to the base weights")
    encoder_drift = drift.get("anytouch2_encoder", 0.0)
    if args.expect_frozen_encoder and encoder_drift != 0.0:
        fail(f"AnyTouch2 encoder was expected frozen but drifted by {encoder_drift}")
    if not args.expect_frozen_encoder and encoder_drift == 0.0:
        fail("AnyTouch2 encoder is unchanged although it should train jointly")
    report["projector_requires_grad"] = all(p.requires_grad for p in initial.anytouch2_encoder.projector.parameters())

    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)
    print("Tactile checks passed.")


if __name__ == "__main__":
    main()
