"""Replay server: speaks the OpenPI websocket protocol but returns a recorded demonstration's next states.
Usage: python replay_server.py <demo.npz> <episode idx in npz> <port> [query_interval=20]
Query k is assumed to happen at policy step k*interval; it gets states[k*interval+1 : k*interval+51] (padded with the last state).
"""
import sys, numpy as np
sys.path.insert(0, "/root/autodl-tmp/OpenViTac")
from policy.openpi import msgpack_numpy as mp
from websockets.sync.server import serve

z = np.load(sys.argv[1]); ep = sys.argv[2]; port = int(sys.argv[3]); interval = int(sys.argv[4]) if len(sys.argv) > 4 else 20
states = np.concatenate([z[f"q{ep}"], z[f"f{ep}"][:, :1]], axis=1).astype(np.float64)
import os
if os.environ.get("HOLD"):
    states = np.repeat(states[:1], len(states), axis=0)  # hold the initial pose: static scene
print("demo states", states.shape, flush=True)

def handler(ws):
    ws.send(mp.packb({"replay": True, "frames": int(len(states))}))
    k = 0
    for msg in ws:
        obs = mp.unpackb(msg)
        if obs.get("reset"):
            k = 0
        i = k * interval
        idx = np.clip(np.arange(i + 1, i + 51), 0, len(states) - 1)
        st = np.asarray(obs["observation/state"], dtype=np.float64)
        ref = states[min(i, len(states) - 1)]
        print(f"query {k} step {i}: |sim state - demo state| max arm {np.abs(st[:7]-ref[:7]).max():.4f} grip sim {st[7]:.4f} demo {ref[7]:.4f}", flush=True)
        ws.send(mp.packb({"actions": states[idx]}))
        k += 1

with serve(handler, "127.0.0.1", port, max_size=None, ping_interval=None) as s:
    s.serve_forever()
