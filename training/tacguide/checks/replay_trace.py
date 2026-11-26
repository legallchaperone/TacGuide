"""Replay a recorded evaluation trace against running policy servers and compare their answers.

    $TG/venv/bin/python training/tacguide/checks/replay_trace.py \
        --trace $TG/logs/eval_debug/gcA20k_10000 --ports 8001 8002

The trace directory holds q*.npz files (one per policy request: the observation that was
sent and the action chunk that came back). Each server gets the same requests in the same
order, the first one with "reset": True, so servers started with the same noise seed draw the
same noise. The first port is the reference: every other port is compared with it element by
element, and each port is also compared with the recorded chunk.

Use it to check that guided inference with w = 1 equals the strong model served alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

RUNTIME = pathlib.Path(__file__).resolve().parents[2] / "openpi_runtime"
sys.path.insert(0, str(RUNTIME / "packages" / "openpi-client" / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prompt", default=None, help="task instruction; default: the dataset's task string")
    parser.add_argument("--task", default="gc")
    parser.add_argument("--limit", type=int, default=0, help="replay only the first N requests")
    parser.add_argument("--out")
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    prompt = args.prompt
    if prompt is None:
        tasks = pathlib.Path(__file__).resolve().parents[3] / "data" / "training" / "lerobot" / args.task / "meta" / "tasks.jsonl"
        prompt = json.loads(tasks.read_text().splitlines()[0])["task"]
    files = sorted(glob.glob(str(pathlib.Path(args.trace) / "q*.npz")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no q*.npz in {args.trace}")

    answers, metadata = {}, {}
    for port in args.ports:
        client = websocket_client_policy.WebsocketClientPolicy(args.host, port)
        raw = client.get_server_metadata() or {}
        metadata[port] = raw.get("tacguide") or raw.get(b"tacguide") or {}
        chunks = []
        for index, path in enumerate(files):
            record = np.load(path, allow_pickle=True)
            obs = {key.replace("__", "/"): record[key] for key in record.files if key.startswith("observation__")}
            obs["prompt"] = prompt
            if index == 0:
                obs["reset"] = True
            chunks.append(np.asarray(client.infer(obs)["actions"], dtype=np.float64))
        answers[port] = np.stack(chunks)
        print(f"port {port}: {len(chunks)} requests, server {metadata[port]}", flush=True)

    recorded = np.stack([np.load(path, allow_pickle=True)["action_chunk"] for path in files])
    reference = args.ports[0]
    result = {"trace": args.trace, "requests": len(files), "prompt": prompt, "reference_port": reference, "ports": {}}
    for port in args.ports:
        chunk = answers[port]
        steps = min(chunk.shape[1], recorded.shape[1])
        entry = {
            "server": metadata[port],
            "max_abs_diff_to_reference": float(np.abs(chunk - answers[reference]).max()),
            "identical_to_reference": bool(np.array_equal(chunk, answers[reference])),
            "max_abs_diff_to_recorded": float(np.abs(chunk[:, :steps] - recorded[:, :steps]).max()),
            "mean_abs_diff_to_recorded": float(np.abs(chunk[:, :steps] - recorded[:, :steps]).mean()),
        }
        result["ports"][str(port)] = entry
    print(json.dumps(result, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
