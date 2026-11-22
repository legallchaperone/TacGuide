"""Check the guided-inference serving path on trained checkpoints (plan item 2.6).

    $TG/venv/bin/python training/tacguide/checks/check_serve.py --strong pd_A --strong-exp gc_A_smoke \
        --weak pd_B --weak-exp gc_B_smoke

Policies are built exactly as `serve_guided.py` builds them (through `training/serve.py`,
with the checkpoints' own transforms and normalisation), then fed one raw client
observation. With the same noise seed:

  guided, w = 1  must return the same actions as the strong policy served alone
  guided, w = 0  must return the same actions as the weak policy served alone
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--strong-exp")
    parser.add_argument("--weak-exp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    from tacguide import configs
    from tacguide.guided import GuidedPolicy, SeededNoisePolicy

    import numpy as np

    strong, strong_stats, strong_dir = configs.create_policy(args.strong, exp_name=args.strong_exp)
    weak, weak_stats, weak_dir = configs.create_policy(args.weak, exp_name=args.weak_exp)

    rng = np.random.default_rng(0)
    vision = {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-0.5, 0.5, 8).astype(np.float32),
        "prompt": "synthetic pipeline test",
    }
    needs_tactile = configs.parse_name(args.strong).kind == "A"
    tactile = {
        f"observation/{side}_tactile_clip": np.ascontiguousarray(rng.integers(0, 255, (4, 224, 224, 3), dtype=np.uint8))
        for side in ("left", "right")
    }
    observation = {**vision, **(tactile if needs_tactile else {}), "reset": True}

    def first_actions(policy, obs):
        session = SeededNoisePolicy(policy, seed=args.seed).create_session()
        return np.asarray(session.infer(dict(obs))["actions"])

    # References first: GuidedPolicy replaces the strong policy's sampling function.
    only_strong = first_actions(strong, observation)
    only_weak = first_actions(weak, {**vision, "reset": True})

    guided = GuidedPolicy(strong, weak, w=1.0, strong_norm_stats=strong_stats, weak_norm_stats=weak_stats)
    at_one = first_actions(guided, observation)
    guided._sampler.w = 0.0  # noqa: SLF001
    at_zero = first_actions(guided, observation)
    guided._sampler.w = 2.0  # noqa: SLF001
    at_two = first_actions(guided, observation)

    results = {
        "strong": str(strong_dir),
        "weak": str(weak_dir),
        "client_sends_tactile": needs_tactile,
        "actions_shape": list(only_strong.shape),
        "guided_w1_equals_strong_alone": bool(np.array_equal(at_one, only_strong)),
        "guided_w0_equals_weak_alone": bool(np.array_equal(at_zero, only_weak)),
        "strong_vs_weak_max_abs_diff": float(np.abs(only_strong - only_weak).max()),
        "guided_w2_vs_strong_max_abs_diff": float(np.abs(at_two - only_strong).max()),
        "metadata": {k: v for k, v in guided.metadata.items() if isinstance(v, (int, float, str, bool))},
    }
    results["passed"] = (
        results["guided_w1_equals_strong_alone"]
        and results["guided_w0_equals_weak_alone"]
        and results["strong_vs_weak_max_abs_diff"] > 0
        and results["guided_w2_vs_strong_max_abs_diff"] > 0
        and only_strong.shape[0] >= 20 and only_strong.shape[1] == 8
    )
    print(json.dumps(results, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    raise SystemExit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
