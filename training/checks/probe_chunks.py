"""Open-loop probe: at chosen steps of a demonstration, compare the served policy's action
chunk with what the demonstration did next.

  python training/checks/probe_chunks.py --task gc --port 8000 [--tactile] [--episode 0] --steps 0 40 80 ...

Prints, for chunk positions 1, 20 and 50: how far the predicted arm target is from the current
state versus how far the demonstration moved, and the predicted versus recorded gripper target.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import config as tg_config  # noqa: E402

os.environ["HF_LEROBOT_HOME"] = str(tg_config.DATA / "lerobot")
_spec = importlib.util.spec_from_file_location("tactile_history", tg_config.ROOT / "policy/openpi/tactile_history.py")
tactile_history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tactile_history)


def u8(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 + 0.5, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tactile", action="store_true")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi_client import websocket_client_policy

    ds = LeRobotDataset(args.task)
    begin = int(ds.episode_data_index["from"][args.episode])
    end = int(ds.episode_data_index["to"][args.episode])
    states = np.stack([np.asarray(ds[i]["state"], dtype=np.float32) for i in range(begin, end)])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    history = tactile_history.TactileHistory() if args.tactile else None
    wanted = set(args.steps)
    print("step | pred arm move (mrad) at k=1,20,50 | demo arm move | pred gripper (mm) k=1,20,50 | demo gripper | state gripper")
    for step in range(0, min(max(wanted) + 1, end - begin)):
        if history is None and step not in wanted:
            continue
        item = ds[begin + step]
        obs = {"observation/state": states[step], "observation/image": u8(item["image"]),
               "observation/wrist_image": u8(item["wrist_image"]), "prompt": item["task"]}
        if history is not None:
            obs["observation/left_tactile_image"] = u8(item["left_tactile_image"])
            obs["observation/right_tactile_image"] = u8(item["right_tactile_image"])
            obs = history.push(obs)
        if step not in wanted:
            continue
        chunk = np.asarray(client.infer(obs)["actions"])
        ks = (0, 19, 49)
        future = [states[min(step + k + 1, len(states) - 1)] for k in ks]
        pred_move = [np.abs(chunk[k, :7] - states[step, :7]).max() * 1000 for k in ks]
        demo_move = [np.abs(f[:7] - states[step, :7]).max() * 1000 for f in future]
        print(f"{step:4d} | " + " ".join(f"{v:6.1f}" for v in pred_move) + " | " + " ".join(f"{v:6.1f}" for v in demo_move)
              + " | " + " ".join(f"{chunk[k, 7] * 1000:5.1f}" for k in ks) + " | "
              + " ".join(f"{f[7] * 1000:5.1f}" for f in future) + f" | {states[step, 7] * 1000:5.1f}")


if __name__ == "__main__":
    main()
