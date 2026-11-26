"""Policy server for guided inference (plan item 2.6).

    # CFG(A, B): the client must send tactile, use the vision+tactile deploy config
    $TG/venv/bin/python training/tacguide/serve_guided.py \
        --strong gc_A --weak gc_B --w 2 --port 8000

    # zero-training extrapolation: no tactile, use the vision deploy config
    $TG/venv/bin/python training/tacguide/serve_guided.py \
        --strong gc_C --weak gc_B --w 2 --port 8000

    # a single model behind the same seeded-noise wrapper (for like-for-like comparisons)
    $TG/venv/bin/python training/tacguide/serve_guided.py --strong gc_A --port 8000

Models are loaded through ``training/serve.py``. Each side defaults to the latest checkpoint
of its run (``--strong-exp``, ``--weak-exp``),
or takes a directory (``--strong-dir``, ``--weak-dir``). With ``--seed`` (default 0) the
k-th action request of an episode always gets the same initial noise, so
"guided with w = 1" and "strong model alone" can be compared episode by episode.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

_TRAINING_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

logger = logging.getLogger("tacguide")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strong", required=True, help="experiment name of the strong model, e.g. gc_A or gc_C")
    parser.add_argument("--weak", help="experiment name of the weak model, e.g. gc_B; omit to serve --strong alone")
    parser.add_argument("--w", type=float, help="guidance weight; required with --weak")
    parser.add_argument("--strong-exp", help="run name; defaults to the experiment name")
    parser.add_argument("--weak-exp", help="run name; defaults to the experiment name")
    parser.add_argument("--step", type=int, help="checkpoint step for both models; defaults to the latest")
    parser.add_argument("--strong-dir")
    parser.add_argument("--weak-dir")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0, help="noise seed; -1 leaves the noise unseeded")
    args = parser.parse_args(argv)
    if (args.weak is None) != (args.w is None):
        parser.error("--weak and --w go together")
    return args


def build_policy(args):
    from tacguide import configs
    from tacguide.guided import GuidedPolicy, SeededNoisePolicy

    policy, strong_stats, strong_dir = configs.create_policy(
        args.strong, exp_name=args.strong_exp, checkpoint_dir=args.strong_dir, step=args.step
    )
    logger.info("strong: %s from %s", args.strong, strong_dir)
    description = {"strong": args.strong, "strong_checkpoint": str(strong_dir)}
    if args.weak is not None:
        weak, weak_stats, weak_dir = configs.create_policy(
            args.weak, exp_name=args.weak_exp, checkpoint_dir=args.weak_dir, step=args.step
        )
        logger.info("weak: %s from %s, w = %s", args.weak, weak_dir, args.w)
        policy = GuidedPolicy(policy, weak, w=args.w, strong_norm_stats=strong_stats, weak_norm_stats=weak_stats)
        description.update({"weak": args.weak, "weak_checkpoint": str(weak_dir), "w": args.w})
    if args.seed >= 0:
        policy = SeededNoisePolicy(policy, seed=args.seed)
        description["noise_seed"] = args.seed
    return policy, description


def limit_cpu_threads(default: int = 8) -> int:
    """Cap CPU threads before torch is imported.

    Unlimited, model construction starts one thread per core (384 on the training machine). On a
    busy machine those threads starve each other and everything else: a load that takes a minute
    took 5 to 15, the load average passed 190 and all of the user's jobs were killed (2026-11-25/26).
    """
    import os

    count = int(os.environ.get("TG_CPU_THREADS", default))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(count)
    return count


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    threads = limit_cpu_threads()

    import torch

    torch.set_num_threads(threads)
    logger.info("CPU threads limited to %d", threads)

    from openpi.serving import websocket_policy_server

    policy, description = build_policy(args)
    metadata = {**(getattr(policy, "metadata", None) or {}), "tacguide": description}
    logger.info("serving %s on %s:%d", description, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
