"""Offline comparison of trained policies on demonstration frames (no simulator needed).

    CUDA_VISIBLE_DEVICES=<gpu> $TG/venv/bin/python training/tacguide/checks/offline_eval.py \
        --models gc_A gc_B gc_C --out $TG/logs/distill/offline_eval.json

Three measurements, all at the 10000-step checkpoints and with the same noise for every model:

  velocity   On shuffled training batches, with one shared (noise, t, x_t): mean squared
             difference of each model's velocity to teacher A's, and to the flow-matching
             target noise - actions (this is the quantity each model was trained on).
  chunks     On frames spread over the demonstrations: the sampled action chunk of each model
             against A's chunk and against what the demonstration did next.
  rest       On frames where the demonstration stands still for the next steps (correct output:
             do not move): the constant offset each model adds per chunk, per joint in mrad and
             as flange position in mm. Same frames and method as checks/diagnostics/bias2.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import os  # noqa: E402

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per core stalls the shared machine
    os.environ.setdefault(_name, os.environ.get("TG_CPU_THREADS", "8"))


import numpy as np

CHUNK_EPISODES = (0, 1, 12, 33, 49, 71, 88)
CHUNK_STEPS = (20, 40, 70, 100, 110, 140)
REST_STEPS = (100, 105, 110)
TACTILE_OFFSETS = (-6, -4, -2, 0)


def flange_mm(q):
    """Franka flange position from the 7 joint angles (modified DH), in millimetres."""
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
    alpha = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
    theta = [*list(q), 0.0]
    pose = np.eye(4)
    for i in range(8):
        ct, st, ca, sa = np.cos(theta[i]), np.sin(theta[i]), np.cos(alpha[i]), np.sin(alpha[i])
        pose = pose @ np.array(
            [[ct, -st, 0, a[i]], [st * ca, ct * ca, -sa, -d[i] * sa], [st * sa, ct * sa, ca, d[i] * ca], [0, 0, 0, 1]]
        )
    return pose[:3, 3] * 1000


def u8(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 + 0.5, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="gc")
    parser.add_argument("--models", nargs="+", default=["gc_A", "gc_B", "gc_C"])
    parser.add_argument("--reference", default="gc_A")
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--draws", type=int, default=4, help="noise draws per frame")
    parser.add_argument("--velocity-batches", type=int, default=4)
    parser.add_argument("--velocity-batch-size", type=int, default=32)
    parser.add_argument("--quick", action="store_true", help="two frames, one draw, one small batch (plumbing test)")
    parser.add_argument("--out")
    args = parser.parse_args()

    from tacguide import configs  # first: sets the dataset home before lerobot is imported
    from tacguide import velocity
    from tacguide.guided import unwrap_policy

    import jax
    import torch

    import openpi.training.data_loader as _data
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    started = time.time()
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    names = list(args.models)
    kinds = {name: configs.parse_name(name).kind for name in names}
    policies = {}
    for name in names:
        policy, _, checkpoint = configs.create_policy(name, step=args.step)
        policies[name] = policy
        print(f"loaded {name} from {checkpoint}", flush=True)
    core = {name: unwrap_policy(policy) for name, policy in policies.items()}
    models = {name: c._model for name, c in core.items()}  # noqa: SLF001
    device = core[names[0]]._pytorch_device  # noqa: SLF001
    result = {"task": args.task, "step": args.step, "models": names, "device": str(device)}

    # ------------------------------------------------------------------ velocity
    student = configs.student_config(f"{args.task}_C")
    batch_size = 4 if args.quick else args.velocity_batch_size
    loader_config = dataclasses.replace(student, batch_size=batch_size, num_workers=0)
    torch.manual_seed(0)
    loader = iter(_data.create_data_loader(loader_config, framework="pytorch", shuffle=True))
    strong_keys = tuple(configs.base_config(f"{args.task}_A").model.image_keys)
    sums = {name: {"to_reference": 0.0, "to_target": 0.0} for name in names}
    count = 0
    for _ in range(1 if args.quick else args.velocity_batches):
        observation, actions = next(loader)
        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(torch.float32).to(device)
        processed = velocity.preprocess_once(observation, strong_keys, train=False)
        noise = torch.randn(actions.shape, device=device)
        t = models[names[0]].sample_time(actions.shape[0], device)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        target = noise - actions
        predicted = {}
        with torch.no_grad():
            for name in names:
                view = velocity.full_view(processed, strong_keys) if kinds[name] == "A" else velocity.vision_view(processed)
                predicted[name] = velocity.predict_velocity(models[name], view, x_t, t)[..., :8]
        for name in names:
            sums[name]["to_reference"] += float(((predicted[name] - predicted[args.reference]) ** 2).mean()) * len(actions)
            sums[name]["to_target"] += float(((predicted[name] - target[..., :8]) ** 2).mean()) * len(actions)
        count += len(actions)
    result["velocity"] = {
        "frames": count,
        "note": "mean squared error in normalised action units over the 8 real action dims",
        **{name: {k: v / count for k, v in sums[name].items()} for name in names},
    }
    print("velocity:", json.dumps(result["velocity"], indent=1), flush=True)

    # ------------------------------------------------------------------ sampled chunks
    dataset = LeRobotDataset(args.task)
    horizon, action_dim = models[names[0]].config.action_horizon, models[names[0]].config.action_dim
    rng = np.random.default_rng(0)
    draws = 1 if args.quick else args.draws
    noises = [rng.standard_normal((horizon, action_dim)).astype(np.float32) for _ in range(draws)]

    def episode(index):
        begin = int(dataset.episode_data_index["from"][index])
        end = int(dataset.episode_data_index["to"][index])
        states = np.stack([np.asarray(dataset[begin + i]["state"], dtype=np.float32) for i in range(min(end - begin, 260))])
        return begin, states

    def observation_for(begin, states, step, with_tactile):
        item = dataset[begin + step]
        obs = {
            "observation/state": states[step],
            "observation/image": u8(item["image"]),
            "observation/wrist_image": u8(item["wrist_image"]),
            "prompt": item["task"],
        }
        if with_tactile:
            for side in ("left", "right"):
                frames = [u8(dataset[max(begin + step + o, begin)][f"{side}_tactile_image"]) for o in TACTILE_OFFSETS]
                obs[f"observation/{side}_tactile_clip"] = np.ascontiguousarray(np.stack(frames))
        return obs

    def chunks_at(begin, states, step):
        out = {}
        for name in names:
            obs = observation_for(begin, states, step, kinds[name] == "A")
            out[name] = np.stack([np.asarray(policies[name].infer(dict(obs), noise=nz)["actions"]) for nz in noises])
        future = np.stack([states[min(step + k + 1, len(states) - 1)] for k in range(horizon)])
        return out, future  # [draws, horizon, 8], [horizon, 8]

    episodes = CHUNK_EPISODES[:1] if args.quick else CHUNK_EPISODES
    cache = {index: episode(index) for index in episodes}

    chunk_steps = CHUNK_STEPS[:2] if args.quick else CHUNK_STEPS
    stats = {name: {"vs_ref_joint": [], "vs_ref_flange20": [], "vs_demo_joint": [], "vs_demo_flange20": [],
                    "vs_demo_gripper": [], "vs_ref_gripper": []} for name in names}
    for index in episodes:
        begin, states = cache[index]
        for step in chunk_steps:
            if step >= len(states) - 1:
                continue
            chunk, future = chunks_at(begin, states, step)
            reference = chunk[args.reference]
            for name in names:
                c = chunk[name]
                stats[name]["vs_ref_joint"].append(np.abs(c[:, :20, :7] - reference[:, :20, :7]).mean() * 1000)
                stats[name]["vs_demo_joint"].append(np.abs(c[:, :20, :7] - future[None, :20, :7]).mean() * 1000)
                stats[name]["vs_ref_gripper"].append(np.abs(c[:, :20, 7] - reference[:, :20, 7]).mean() * 1000)
                stats[name]["vs_demo_gripper"].append(np.abs(c[:, :20, 7] - future[None, :20, 7]).mean() * 1000)
                for d in range(len(c)):
                    stats[name]["vs_ref_flange20"].append(np.linalg.norm(flange_mm(c[d, 19, :7]) - flange_mm(reference[d, 19, :7])))
                    stats[name]["vs_demo_flange20"].append(np.linalg.norm(flange_mm(c[d, 19, :7]) - flange_mm(future[19, :7])))
        print(f"chunks: episode {index} done ({time.time() - started:.0f}s)", flush=True)
    result["chunks"] = {
        "frames": len(stats[names[0]]["vs_demo_joint"]),
        "draws_per_frame": draws,
        "note": "first 20 chunk steps; joint = mean abs error in mrad, gripper in mm, flange20 = distance in mm at chunk step 20",
        **{name: {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in stats[name].items()} for name in names},
    }
    print("chunks:", json.dumps(result["chunks"], indent=1), flush=True)

    # ------------------------------------------------------------------ rest frames
    rest_steps = REST_STEPS[:1] if args.quick else REST_STEPS
    errors = {name: [] for name in names}   # [frames*draws, horizon, 7] mrad
    flange = {name: [] for name in names}   # [frames*draws, 3] mm at chunk step 20
    still = []
    for index in episodes:
        begin, states = cache[index]
        for step in rest_steps:
            chunk, future = chunks_at(begin, states, step)
            still.append(float(np.abs(future[:10, :7] - states[step, :7]).max() * 1000))
            for name in names:
                errors[name].extend((chunk[name][:, :, :7] - future[None, :, :7]) * 1000)
                flange[name].extend(flange_mm(c[19, :7]) - flange_mm(future[19, :7]) for c in chunk[name])
        print(f"rest: episode {index} done ({time.time() - started:.0f}s)", flush=True)
    result["rest"] = {
        "frames": len(still),
        "draws_per_frame": draws,
        "demo_motion_next_10_steps_mrad_max": float(np.max(still)),
        "note": "error = prediction - demonstration; joints in mrad, flange xyz in mm at chunk step 20",
    }
    for name in names:
        e = np.asarray(errors[name])
        f = np.asarray(flange[name])
        result["rest"][name] = {
            "joint_offset_mean_first10": e[:, :10].mean((0, 1)).round(3).tolist(),
            "joint_offset_std_first10": e[:, :10].std((0, 1)).round(3).tolist(),
            "joint_offset_mean_by_chunk_index": {str(k + 1): e[:, k].mean(0).round(3).tolist() for k in (0, 4, 9, 19)},
            "flange_offset_mean_xyz": f.mean(0).round(3).tolist(),
            "flange_offset_std_xyz": f.std(0).round(3).tolist(),
        }
    print("rest:", json.dumps(result["rest"], indent=1), flush=True)
    result["seconds"] = round(time.time() - started)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print("done in", result["seconds"], "s")


if __name__ == "__main__":
    main()
