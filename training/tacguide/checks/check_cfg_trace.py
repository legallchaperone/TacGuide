"""Offline equivalence check for guided inference on a recorded evaluation trace.

    CUDA_VISIBLE_DEVICES=<gpu> $TG/venv/bin/python training/tacguide/checks/check_cfg_trace.py \
        --strong gc_A --weak gc_B --step 20000 --trace $TG/logs/eval_debug/gcA20k_10000

One process loads the strong and the weak model once, then replays the trace's observations
(q*.npz, in order, first one with "reset": True, noise seed as given) three ways:

  the strong policy served alone                      -> reference
  guided inference with w = 1                         -> must equal the reference element for element
  guided inference with each value in --also (2 1.5 3) -> how far the actions move, for information

It also reports how close the reference is to the action chunks recorded in the trace, i.e.
whether this replay reproduces what the evaluation server returned at the time.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import os  # noqa: E402

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per core stalls the shared machine
    os.environ.setdefault(_name, os.environ.get("TG_CPU_THREADS", "8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--also", type=float, nargs="*", default=[2.0, 1.5, 3.0])
    parser.add_argument("--prompt")
    parser.add_argument("--out")
    args = parser.parse_args()

    from tacguide import configs
    from tacguide.guided import GuidedPolicy, SeededNoisePolicy

    import numpy as np

    task = configs.parse_name(args.strong).task
    prompt = args.prompt
    if prompt is None:
        tasks = pathlib.Path(__file__).resolve().parents[3] / "data" / "training" / "lerobot" / task / "meta" / "tasks.jsonl"
        prompt = json.loads(tasks.read_text().splitlines()[0])["task"]
    files = sorted(glob.glob(str(pathlib.Path(args.trace) / "q*.npz")))
    if not files:
        raise SystemExit(f"no q*.npz in {args.trace}")
    requests, recorded = [], []
    for index, path in enumerate(files):
        record = np.load(path, allow_pickle=True)
        obs = {key.replace("__", "/"): np.ascontiguousarray(record[key]) for key in record.files if key.startswith("observation__")}
        obs["prompt"] = prompt
        if index == 0:
            obs["reset"] = True
        requests.append(obs)
        recorded.append(np.asarray(record["action_chunk"], dtype=np.float64))
    recorded = np.stack(recorded)

    strong, strong_stats, strong_dir = configs.create_policy(args.strong, step=args.step)
    weak, weak_stats, weak_dir = configs.create_policy(args.weak, step=args.step)

    def replay(policy):
        session = SeededNoisePolicy(policy, seed=args.seed).create_session()
        return np.stack([np.asarray(session.infer(dict(obs))["actions"], dtype=np.float64) for obs in requests])

    reference = replay(strong)  # before GuidedPolicy replaces the strong policy's sampling function
    guided = GuidedPolicy(strong, weak, w=1.0, strong_norm_stats=strong_stats, weak_norm_stats=weak_stats)
    at_one = replay(guided)
    steps = min(reference.shape[1], recorded.shape[1])
    result = {
        "strong": str(strong_dir),
        "weak": str(weak_dir),
        "trace": args.trace,
        "requests": len(requests),
        "noise_seed": args.seed,
        "prompt": prompt,
        "guided_w1_identical_to_strong_alone": bool(np.array_equal(at_one, reference)),
        "guided_w1_max_abs_diff": float(np.abs(at_one - reference).max()),
        "strong_alone_vs_recorded": {
            "max_abs_diff": float(np.abs(reference[:, :steps] - recorded[:, :steps]).max()),
            "mean_abs_diff": float(np.abs(reference[:, :steps] - recorded[:, :steps]).mean()),
        },
        "shift_from_strong_first20_steps": {},
    }
    for w in args.also:
        guided._sampler.w = float(w)  # noqa: SLF001
        chunk = replay(guided)
        diff = np.abs(chunk[:, :20] - reference[:, :20])
        result["shift_from_strong_first20_steps"][str(w)] = {
            "arm_mean_mrad": float(diff[..., :7].mean() * 1000),
            "arm_max_mrad": float(diff[..., :7].max() * 1000),
            "gripper_mean_mm": float(diff[..., 7].mean() * 1000),
        }
    result["passed"] = result["guided_w1_identical_to_strong_alone"]
    print(json.dumps(result, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
