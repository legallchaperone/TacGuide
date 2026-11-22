"""Check the dataset written by convert_official_lerobot.py against the public dataset it came from.

  uv run --project training/openpi_runtime python training/checks/check_official_conversion.py \
      --source data/official_openvitac_lerobot/grasp_chip --task gc --out data/training/checks/gc

Same checks as check_conversion.py, with the public LeRobot dataset as the reference instead
of HDF5 files: field shapes, state = [7 joints, left finger metres], action[t] = state[t+1],
images equal to the source video frames resized to 224, and preview videos for three episodes.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import config as tg_config  # noqa: E402
import convert_official_lerobot as adapter  # noqa: E402

os.environ["HF_LEROBOT_HOME"] = str(tg_config.DATA / "lerobot")
IMAGE_KEYS = tuple(adapter.VIDEO_KEYS)


def to_uint8_hwc(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 + 0.5, 0, 255)
    return image.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=10)
    args = parser.parse_args()

    import imageio.v2 as imageio
    import pandas as pd
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    source = args.source.expanduser().resolve()
    info = json.loads((source / "meta/info.json").read_text())
    source_episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line]
    dataset = LeRobotDataset(args.task)
    meta = dataset.meta
    report = {"source_episodes": len(source_episodes), "dataset_episodes": meta.total_episodes,
              "source_frames": info["total_frames"], "dataset_frames": meta.total_frames, "fps": meta.fps,
              "episodes": [], "failures": []}
    fail = report["failures"].append
    if (len(source_episodes), info["total_frames"], info["fps"]) != (meta.total_episodes, meta.total_frames, meta.fps):
        fail("episode count, frame count or fps differs from the source")
    for key in IMAGE_KEYS:
        if tuple(meta.features[key]["shape"]) != (224, 224, 3):
            fail(f"{key} has shape {meta.features[key]['shape']}")
    for key in ("state", "actions"):
        if tuple(meta.features[key]["shape"]) != (8,):
            fail(f"{key} has shape {meta.features[key]['shape']}")

    args.out.mkdir(parents=True, exist_ok=True)
    picks = np.unique(np.linspace(0, meta.total_episodes - 1, args.episodes).astype(int))
    for ep in map(int, picks):
        begin, end = int(dataset.episode_data_index["from"][ep]), int(dataset.episode_data_index["to"][ep])
        table = pd.read_parquet(source / info["data_path"].format(episode_chunk=0, episode_index=ep))
        expected_state = np.concatenate([
            np.stack(table["observation.joint_position"].to_numpy()),
            np.stack(table["observation.finger_position"].to_numpy())[:, :1],
        ], axis=1).astype(np.float32)
        videos = {key: adapter.read_video(source / info["video_path"].format(
            episode_chunk=0, video_key=video_key, episode_index=ep)) for key, video_key in adapter.VIDEO_KEYS.items()}
        states, actions, mosaic = [], [], []
        image_err = dict.fromkeys(IMAGE_KEYS, 0.0)
        compared = 0
        for t, index in enumerate(range(begin, end)):
            item = dataset[index]
            states.append(np.asarray(item["state"], dtype=np.float32))
            actions.append(np.asarray(item["actions"], dtype=np.float32))
            if t == 0 and item["task"] != source_episodes[ep]["tasks"][0]:
                fail(f"episode {ep}: prompt differs")
            images = {key: to_uint8_hwc(item[key]) for key in IMAGE_KEYS}
            mosaic.append(np.concatenate([images[key] for key in IMAGE_KEYS], axis=1))
            if t % args.frame_step == 0:
                compared += 1
                for key in IMAGE_KEYS:
                    image_err[key] += float(np.abs(images[key].astype(np.float32) - videos[key][t]).mean())
        states, actions = np.stack(states), np.stack(actions)
        entry = {
            "episode": ep, "frames": end - begin,
            "state_max_abs_err": float(np.abs(states - expected_state).max()),
            "action_vs_next_state_max_abs_err": float(np.abs(actions[:-1] - states[1:]).max()),
            "last_action_vs_last_state_max_abs_err": float(np.abs(actions[-1] - states[-1]).max()),
            "max_joint_step_rad": float(np.abs(np.diff(states[:, :7], axis=0)).max()),
            "gripper_range_m": [float(states[:, 7].min()), float(states[:, 7].max())],
            "image_mean_abs_err": {key: image_err[key] / compared for key in IMAGE_KEYS},
        }
        if entry["state_max_abs_err"] > 1e-6 or entry["action_vs_next_state_max_abs_err"] > 1e-6 \
                or entry["last_action_vs_last_state_max_abs_err"] > 1e-6:
            fail(f"episode {ep}: state or action mismatch")
        if max(entry["image_mean_abs_err"].values()) > 1.0:
            fail(f"episode {ep}: images differ from the source video")
        video = args.out / f"official_episode_{ep:03d}.mp4"
        imageio.mimsave(video, mosaic, fps=meta.fps, macro_block_size=None)
        cv2.imwrite(str(args.out / f"official_episode_{ep:03d}_mid.png"), mosaic[len(mosaic) * 2 // 3][..., ::-1])
        entry["preview_video"] = str(video)
        report["episodes"].append(entry)
    (args.out / "official_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
