"""Send dataset observations to a running policy server the way the sim client does.

  python training/checks/check_server.py --task gc --port 8000 [--tactile] [--queries 5]

Walks through episode 0 one step at a time. With --tactile every step goes through the
client's TactileHistory (policy/openpi/tactile_history.py) and the server gets 4-frame
clips; the server is queried every 20 steps, like the sim client's open-loop horizon.
Checks that each reply holds a finite action chunk [T >= 20, 8], and reports how far the
first predicted joint target is from the recorded next state.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import config as tg_config  # noqa: E402

os.environ["HF_LEROBOT_HOME"] = str(tg_config.DATA / "lerobot")
_spec = importlib.util.spec_from_file_location("tactile_history", tg_config.ROOT / "policy/openpi/tactile_history.py")
tactile_history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tactile_history)


def to_uint8_hwc(image) -> np.ndarray:
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
    parser.add_argument("--tactile", action="store_true", help="Send tactile clips (OpenVTLA server)")
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--open-loop-horizon", type=int, default=20)
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi_client import websocket_client_policy

    dataset = LeRobotDataset(args.task)
    begin, end = int(dataset.episode_data_index["from"][0]), int(dataset.episode_data_index["to"][0])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("server metadata:", client.get_server_metadata())
    history = tactile_history.TactileHistory() if args.tactile else None
    results, failures = [], []
    for step, index in enumerate(range(begin, end)):
        if len(results) >= args.queries:
            break
        item = dataset[index]
        obs = {
            "observation/state": np.asarray(item["state"], dtype=np.float32),
            "observation/image": to_uint8_hwc(item["image"]),
            "observation/wrist_image": to_uint8_hwc(item["wrist_image"]),
            "prompt": item["task"],
        }
        if history is not None:
            obs["observation/left_tactile_image"] = to_uint8_hwc(item["left_tactile_image"])
            obs["observation/right_tactile_image"] = to_uint8_hwc(item["right_tactile_image"])
            obs = history.push(obs)
        if step == 0:
            obs["reset"] = True
        if step % args.open_loop_horizon:
            continue
        start = time.monotonic()
        actions = np.asarray(client.infer(obs)["actions"])
        entry = {
            "step": step,
            "shape": list(actions.shape),
            "seconds": round(time.monotonic() - start, 3),
            "first_action_minus_next_state_max_abs": float(
                np.abs(actions[0] - np.asarray(item["actions"], dtype=np.float32)).max()
            ),
        }
        results.append(entry)
        if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < args.open_loop_horizon:
            failures.append(f"step {step}: bad action shape {actions.shape}")
        if not np.all(np.isfinite(actions)):
            failures.append(f"step {step}: non-finite actions")
    print(json.dumps({"queries": results, "failures": failures}, indent=2))
    if failures or not results:
        raise SystemExit(1)
    print("Server check passed.")


if __name__ == "__main__":
    main()
