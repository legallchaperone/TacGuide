"""One-off CPU diagnosis of guided inference on the recorded gc_A 20k trace (no GPU, 16 threads)."""
import os, sys, glob, json, time
for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[n] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.expanduser("~/TacGuide-distill/training"))
from tacguide import configs
from tacguide.guided import GuidedPolicy, SeededNoisePolicy
import numpy as np
import torch
torch.set_num_threads(16)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PROMPT = "pick and insert the HDMI"
TRACE = "/data2/cwx/TacGuide/logs/eval_debug/gcA20k_10000"

def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]*1000

files = sorted(glob.glob(TRACE + "/q*.npz"))[:N]
reqs, rec = [], []
for i, f in enumerate(files):
    r = np.load(f, allow_pickle=True)
    o = {k.replace("__", "/"): np.ascontiguousarray(r[k]) for k in r.files if k.startswith("observation__")}
    o["prompt"] = PROMPT
    if i == 0: o["reset"] = True
    reqs.append(o); rec.append(np.asarray(r["action_chunk"], dtype=np.float64))
rec = np.stack(rec)
t0 = time.time()
strong, ss, sd = configs.create_policy("gc_A", step=20000)
weak, ws, wd = configs.create_policy("gc_B", step=20000)
print("loaded", sd, wd, round(time.time()-t0), "s", flush=True)

def replay(policy):
    s = SeededNoisePolicy(policy, seed=0).create_session()
    out = []
    for o in reqs:
        out.append(np.asarray(s.infer(dict(o))["actions"], dtype=np.float64))
    return np.stack(out)

A = replay(strong); print("A alone done", round(time.time()-t0), "s", flush=True)
B = replay(weak); print("B alone done", round(time.time()-t0), "s", flush=True)
g = GuidedPolicy(strong, weak, w=1.0, strong_norm_stats=ss, weak_norm_stats=ws)
W1 = replay(g); print("guided w=1 done", round(time.time()-t0), "s", flush=True)
g._sampler.w = 0.0; W0 = replay(g); print("guided w=0 done", round(time.time()-t0), "s", flush=True)
g._sampler.w = 2.0; W2 = replay(g); print("guided w=2 done", round(time.time()-t0), "s", flush=True)

steps = min(A.shape[1], rec.shape[1])
def mrad(x): return float(np.abs(x[..., :7]).max()*1000), float(np.abs(x[..., :7]).mean()*1000)
res = {
 "requests": N, "prompt": PROMPT, "device": "cpu",
 "guided_w1_identical_to_A_alone": bool(np.array_equal(W1, A)), "guided_w1_vs_A_max_abs": float(np.abs(W1-A).max()),
 "guided_w0_identical_to_B_alone": bool(np.array_equal(W0, B)), "guided_w0_vs_B_max_abs": float(np.abs(W0-B).max()),
 "A_alone_cpu_vs_recorded_gpu_arm_mrad_max_mean": mrad(A[:, :steps]-rec[:, :steps]),
 "A_vs_B_arm_mrad_max_mean": mrad(A-B),
 "w2_vs_A_arm_mrad_max_mean": mrad(W2-A),
 "w2_vs_linear_extrapolation_of_final_actions_arm_mrad_max_mean": mrad(W2-(B+2*(A-B))),
}
def zs(X, k): return [round(float(fk(X[i, k, :7])[2]), 2) for i in range(len(X))]
def zmin(X): return [round(float(min(fk(X[i, k, :7])[2] for k in range(0, X.shape[1], 2))), 2) for i in range(len(X))]
res["flange_z_mm_at_chunk_step_20"] = {"recorded_A_gpu": zs(rec, 19), "A": zs(A, 19), "B": zs(B, 19), "w2": zs(W2, 19)}
res["flange_z_mm_min_over_chunk"] = {"recorded_A_gpu": zmin(rec), "A": zmin(A), "B": zmin(B), "w2": zmin(W2)}
res["sign_check_w2_minus_A_vs_A_minus_B_flange_z_step20"] = [[round(w-a, 2), round(a-b, 2)] for w, a, b in zip(zs(W2, 19), zs(A, 19), zs(B, 19))]
res["seconds"] = round(time.time()-t0)
print(json.dumps(res, indent=1))
json.dump(res, open("/data2/cwx/TacGuide/logs/distill/cfg_diag_cpu.json", "w"), indent=1)
