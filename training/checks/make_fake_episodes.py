"""Write small synthetic OpenViTac-style episodes to exercise the pipeline before real data exists.

  python training/checks/make_fake_episodes.py --task pd --episodes 4 --frames 90

Files go to data/training/hdf5/<task>/. Use a task you are not training on, and delete
the hdf5, lerobot and stats directories of that task afterwards.
"""

import argparse
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as tg_config  # noqa: E402

STREAMS = ("observation/head/rgb", "observation/wrist/rgb",
           "tactile/left_tactile/rgb_marker", "tactile/right_tactile/rgb_marker")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=tg_config.TASKS)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    out = tg_config.DATA / "hdf5" / args.task
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        t = np.linspace(0, 1, args.frames, dtype=np.float32)[:, None]
        joint = np.sin(t * (1 + np.arange(9, dtype=np.float32)) + ep).astype(np.float32) * 0.5
        with h5py.File(out / f"{ep}.hdf5", "w") as h5:
            h5.attrs["instruction"] = "synthetic pipeline test"
            h5["step"] = np.arange(args.frames) * 2
            h5["embodiment/joint"] = joint
            for s, key in enumerate(STREAMS):
                frames = []
                for f in range(args.frames):
                    image = np.zeros((240, 320, 3), np.uint8)
                    image[..., 0] = 40 + 50 * s          # red grows with the stream index
                    image[..., 2] = (f * 255) // args.frames  # blue grows with time
                    cv2.circle(image, (int(40 + 3 * f), 60 + 30 * ep), 20, (255, 255, 255), -1)
                    image += rng.integers(0, 4, image.shape, dtype=np.uint8)
                    ok, encoded = cv2.imencode(".jpg", image)
                    assert ok
                    frames.append(encoded.tobytes())
                h5[key] = np.asarray(frames, dtype=f"S{max(map(len, frames))}")
    print(f"Wrote {args.episodes} episodes to {out}")


if __name__ == "__main__":
    main()
