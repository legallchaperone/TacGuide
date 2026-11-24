import os, numpy as np, cv2, glob
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds=LeRobotDataset("gc")
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
def markers(im):
    g=im.mean(-1); m=(g<45).astype(np.uint8); n,_,st,_=cv2.connectedComponentsWithStats(m,8)
    return int(((st[1:,4]>=6)&(st[1:,4]<=400)).sum()), float((g<45).mean())
L=[];R=[];W=[]
for ep in range(0,100,4):
    it=ds[int(ds.episode_data_index["from"][ep])]
    L.append(u8(it["left_tactile_image"])); R.append(u8(it["right_tactile_image"])); W.append(u8(it["wrist_image"]).mean())
ref=np.median(np.stack(L),0).astype(np.uint8)
np.save("/data2/cwx/TacGuide/logs/eval_debug/tactile_rest_reference_left_224.npy", ref)
print("training step-0 left tactile: marker count", sorted(set(markers(x)[0] for x in L)), "dark fraction %.4f..%.4f"%(min(markers(x)[1] for x in L),max(markers(x)[1] for x in L)), "mean abs diff to reference max %.2f"%max(np.abs(x.astype(float)-ref).mean() for x in L))
print("right:", sorted(set(markers(x)[0] for x in R)), "diff to left ref max %.2f"%max(np.abs(x.astype(float)-ref).mean() for x in R))
print("training step-0 wrist grey mean range %.1f..%.1f"%(min(W),max(W)))
for d in ("trace_A_10000","trace_Ah50_10000"):
    z=np.load(f"/data2/cwx/TacGuide/logs/eval_debug/{d}/q00000.npz"); t=z["observation__left_tactile_clip"][-1]
    print(d,"markers",markers(t),"diff to ref %.2f"%np.abs(t.astype(float)-ref).mean(),"wrist mean %.1f"%z["observation__wrist_image"].mean())
