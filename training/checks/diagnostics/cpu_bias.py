"""CPU-only: does the constant per-chunk offset depend on the number of denoising steps or on bfloat16?"""
import os, sys, time, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0, os.path.expanduser("~/TacGuide/training"))
import run as tg_run, serve as tg_serve, config as tg_config
import torch
torch.set_num_threads(48)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
ds=LeRobotDataset("gc"); np.set_printoptions(precision=2, suppress=True, linewidth=200)
policy, cfg, ckpt = tg_serve.load_policy("gc","pi05","gc_B",step=10000,device="cpu")
model=policy._model
frames=[]
for ep in (0,33,71):
    b=int(ds.episode_data_index["from"][ep])
    for step in (100,110):
        it=ds[b+step]; frames.append(({"observation/state":np.asarray(it["state"],dtype=np.float32),"observation/image":u8(it["image"]),"observation/wrist_image":u8(it["wrist_image"]),"prompt":it["task"]}, np.asarray(ds[b+step+1]["state"])[:7]))
rng=np.random.default_rng(0); noises=[rng.standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32) for _ in range(2)]
def measure(tag, steps):
    policy._sample_kwargs={"num_steps":steps}; E=[]; t0=time.time()
    for obs,tgt in frames:
        for nz in noises:
            ch=policy.infer(dict(obs),noise=nz)["actions"]; E.append((ch[:10,:7]-tgt)*1000)
    E=np.array(E).reshape(-1,7)
    print("%-34s mean joint offset mrad %s | std %s | %.0fs"%(tag,E.mean(0),E.std(0),time.time()-t0), flush=True)
measure("bfloat16 weights, 10 steps",10)
measure("bfloat16 weights, 50 steps",50)
model.float()
measure("float32 weights, 10 steps",10)
measure("float32 weights, 50 steps",50)
