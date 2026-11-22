"""Check a converted LeRobot dataset against its source HDF5 episodes (plan item 2.2).

Run from the repository root after `training/run.py convert`:

  uv run --project training/openpi_runtime python training/checks/check_conversion.py \
      --task gc --episodes 3 --out data/training/checks/gc

For each sampled episode it checks that
  * the fields have the names, shapes and types the policy interface expects;
  * state equals the HDF5 joints [7 arm joints, left gripper joint];
  * the action at step t equals the state at step t+1 (the last step repeats itself);
  * every image equals the HDF5 frame decoded the way the converter decodes it,
    and reports whether a red/blue swap would match better;
and writes one preview video per episode (head | wrist | left tactile | right tactile)
to compare by eye with the preview video saved at collection time.
Exit code 0 means every numeric check passed.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import config as tg_config  # noqa: E402

os.environ["HF_LEROBOT_HOME"] = str(tg_config.DATA / "lerobot")
sys.path.insert(0, str(tg_config.RUNTIME / "examples" / "UniVTAC"))

IMAGE_KEYS = ("image", "wrist_image", "left_tactile_image", "right_tactile_image")
EXPECTED_SHAPES = {"state": (8,), "actions": (8,), **{key: (224, 224, 3) for key in IMAGE_KEYS}}


def to_uint8_hwc(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 + 0.5, 0, 255)
    return image.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--episodes", type=int, default=3, help="How many episodes to sample")
    parser.add_argument("--out", type=Path, required=True, help="Directory for preview videos and the report")
    parser.add_argument("--image-tolerance", type=float, default=3.0,
                        help="Largest allowed mean absolute pixel difference (images are stored compressed)")
    parser.add_argument("--frame-step", type=int, default=5, help="Compare every N-th image frame")
    return parser.parse_args()


def main():
    args = parse_args()
    import convert_sim_to_lerobot as base
    import h5py
    import imageio.v2 as imageio
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    hdf5_dir = tg_config.DATA / "hdf5" / args.task
    paths = base.sorted_hdf5_paths(hdf5_dir, None)
    dataset = LeRobotDataset(args.task)
    meta = dataset.meta
    report = {"task": args.task, "hdf5_episodes": len(paths), "dataset_episodes": meta.total_episodes,
              "fps": meta.fps, "frames": meta.total_frames, "episodes": [], "failures": []}
    fail = report["failures"].append

    if len(paths) != meta.total_episodes:
        fail(f"episode count differs: {len(paths)} hdf5 files, {meta.total_episodes} dataset episodes")
    for key, shape in EXPECTED_SHAPES.items():
        got = tuple(meta.features[key]["shape"]) if key in meta.features else None
        if got != shape:
            fail(f"feature {key}: expected shape {shape}, got {got}")

    picks = np.unique(np.linspace(0, meta.total_episodes - 1, min(args.episodes, meta.total_episodes)).astype(int))
    args.out.mkdir(parents=True, exist_ok=True)
    starts = dataset.episode_data_index["from"]
    ends = dataset.episode_data_index["to"]
    for ep in picks:
        ep = int(ep)
        begin, end = int(starts[ep]), int(ends[ep])
        path = paths[ep]
        entry = {"episode": ep, "hdf5": path.name, "frames": end - begin}
        with h5py.File(path, "r") as h5:
            joint = np.asarray(h5["embodiment/joint"], dtype=np.float32)
            prompt = base.read_episode_prompt(h5, None, path)
            expected_state = base.make_abs_joint_state(joint, "left")
            if len(joint) != end - begin:
                fail(f"episode {ep}: {len(joint)} hdf5 frames, {end - begin} dataset frames")
            states, actions, frames_out = [], [], []
            image_err = dict.fromkeys(IMAGE_KEYS, 0.0)
            swapped_err = dict.fromkeys(IMAGE_KEYS, 0.0)
            compared = 0
            for t, index in enumerate(range(begin, end)):
                item = dataset[index]
                states.append(np.asarray(item["state"], dtype=np.float32))
                actions.append(np.asarray(item["actions"], dtype=np.float32))
                if t == 0 and item["task"] != prompt:
                    fail(f"episode {ep}: prompt differs: {item['task']!r} vs {prompt!r}")
                images = {key: to_uint8_hwc(item[key]) for key in IMAGE_KEYS}
                frames_out.append(np.concatenate([images[key] for key in IMAGE_KEYS], axis=1))
                if t % args.frame_step == 0 and t < len(joint):
                    compared += 1
                    for key in IMAGE_KEYS:
                        source = base.resize_image(
                            base.decode_sim_rgb_jpeg(h5[base.IMAGE_STREAMS[key]], t), (224, 224)
                        ).astype(np.float32)
                        image_err[key] += float(np.abs(images[key] - source).mean())
                        swapped_err[key] += float(np.abs(images[key][..., ::-1] - source).mean())
        states, actions = np.stack(states), np.stack(actions)
        n = min(len(states), len(expected_state))
        entry["state_max_abs_err"] = float(np.abs(states[:n] - expected_state[:n]).max())
        entry["action_vs_next_state_max_abs_err"] = float(np.abs(actions[:-1] - states[1:]).max())
        entry["last_action_vs_last_state_max_abs_err"] = float(np.abs(actions[-1] - states[-1]).max())
        entry["image_mean_abs_err"] = {key: image_err[key] / max(compared, 1) for key in IMAGE_KEYS}
        entry["image_mean_abs_err_if_red_blue_swapped"] = {key: swapped_err[key] / max(compared, 1) for key in IMAGE_KEYS}
        if entry["state_max_abs_err"] > 1e-6:
            fail(f"episode {ep}: state differs from hdf5 joints by {entry['state_max_abs_err']}")
        if entry["action_vs_next_state_max_abs_err"] > 1e-6 or entry["last_action_vs_last_state_max_abs_err"] > 1e-6:
            fail(f"episode {ep}: action[t] is not state[t+1]")
        for key in IMAGE_KEYS:
            if entry["image_mean_abs_err"][key] > args.image_tolerance:
                fail(f"episode {ep}: {key} differs from hdf5 by {entry['image_mean_abs_err'][key]:.2f}")
        video = args.out / f"episode_{ep:03d}_{path.stem}.mp4"
        imageio.mimsave(video, frames_out, fps=meta.fps, macro_block_size=None)
        entry["preview_video"] = str(video)
        report["episodes"].append(entry)

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)
    print("All conversion checks passed. Compare the preview videos with the collection previews by eye.")


if __name__ == "__main__":
    main()
