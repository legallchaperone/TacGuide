"""Rewrite the public OpenViTac LeRobot dataset into the layout `training/run.py` trains on.

The public release (Simulation/GelSight_mini/<task>, LeRobot v2.1) stores videos at the
source resolution, has no action column, and its 8-D observation.state ends with a gripper
open fraction. Training and the sim client instead use

  state   = [7 arm joints (rad), left finger position (m)]
  actions = the state of the next frame (the last frame repeats itself)
  image, wrist_image, left_tactile_image, right_tactile_image = 224x224 RGB

which is what the bundled HDF5 converter writes. This script produces exactly that from
the public dataset, so the rest of the pipeline (stats, train, serve) is unchanged.

  uv run --project training/openpi_runtime python training/convert_official_lerobot.py \
      --source data/official_openvitac_lerobot/grasp_chip --task gc
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as tg_config  # noqa: E402

VIDEO_KEYS = {
    "image": "observation.images.head",
    "wrist_image": "observation.images.wrist",
    "left_tactile_image": "observation.images.tactile_left",
    "right_tactile_image": "observation.images.tactile_right",
}
SIZE = 224


def read_video(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    return [cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for frame in iio.imiter(path, plugin="pyav")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Root of the public LeRobot dataset")
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    source = args.source.expanduser().resolve()
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    output = tg_config.DATA / "lerobot" / args.task
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    image_feature = {"dtype": "image", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channel"]}
    features = {key: dict(image_feature) for key in VIDEO_KEYS}
    features["state"] = {"dtype": "float32", "shape": (8,), "names": ["state"]}
    features["actions"] = {"dtype": "float32", "shape": (8,), "names": ["actions"]}
    dataset = LeRobotDataset.create(repo_id=args.task, root=output, robot_type="franka", fps=info["fps"],
                                    features=features, image_writer_processes=5, image_writer_threads=10)
    total = 0
    for episode in episodes:
        index = episode["episode_index"]
        chunk = index // info["chunks_size"]
        table = pd.read_parquet(source / info["data_path"].format(episode_chunk=chunk, episode_index=index))
        joints = np.stack(table["observation.joint_position"].to_numpy()).astype(np.float32)
        fingers = np.stack(table["observation.finger_position"].to_numpy()).astype(np.float32)
        state = np.concatenate([joints, fingers[:, :1]], axis=1)
        actions = np.concatenate([state[1:], state[-1:]], axis=0)
        videos = {
            key: read_video(source / info["video_path"].format(episode_chunk=chunk, video_key=video_key,
                                                               episode_index=index))
            for key, video_key in VIDEO_KEYS.items()
        }
        for key, frames in videos.items():
            if len(frames) != len(state):
                raise ValueError(f"episode {index}: {key} has {len(frames)} frames, table has {len(state)} rows")
        for t in range(len(state)):
            frame = {key: videos[key][t] for key in VIDEO_KEYS}
            frame.update(state=state[t], actions=actions[t], task=episode["tasks"][0])
            dataset.add_frame(frame)
        dataset.save_episode()
        total += len(state)
        print(f"episode {index}: {len(state)} frames", flush=True)
    print(f"Done. Wrote {len(episodes)} episodes and {total} frames to {output}")


if __name__ == "__main__":
    main()
