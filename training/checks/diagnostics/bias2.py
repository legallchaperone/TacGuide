import os, json, numpy as np
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
ds=LeRobotDataset("gc"); np.set_printoptions(precision=2, suppress=True, linewidth=220)
cl=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",8000)
ns=json.load(open("/data2/cwx/TacGuide/outputs/training/pi05/gc/gc_B/10000/assets/gc/norm_stats.json"))["norm_stats"]["actions"]
q01=np.array(ns["q01"][:8]); q99=np.array(ns["q99"][:8]); half=(q99-q01)/2
print("action half-range (rad; last = m):", half.round(4)); print("normalised value of a zero move:", ((0-q01)/(q99-q01)*2-1).round(3))
E=[]  # error per chunk index, joints, in mrad
for ep in (0,1,33,49,71,12,88):
    b=int(ds.episode_data_index["from"][ep]); e=int(ds.episode_data_index["to"][ep])
    S=np.stack([np.asarray(ds[b+i]["state"]) for i in range(min(e-b,240))])
    for step in (100,105,110):   # arm at rest at the bottom for the next 10+ steps in every demo
        it=ds[b+step]; obs={"observation/state":S[step].astype(np.float32),"observation/image":u8(it["image"]),"observation/wrist_image":u8(it["wrist_image"]),"prompt":it["task"]}
        for k in range(4):
            o=dict(obs); 
            if k==0: o["reset"]=True
            ch=np.asarray(cl.infer(o)["actions"])
            tgt=np.stack([S[min(step+i+1,len(S)-1)] for i in range(50)])
            E.append((ch[:,:7]-tgt[:,:7])*1000)
E=np.array(E)
print("frames x draws:",len(E))
for k in (0,1,4,9,19):
    print("chunk index %2d: mean joint error mrad"%k, E[:,k].mean(0), "| std", E[:,k].std(0), "| in normalised units", (E[:,k].mean(0)/1000/half[:7]).round(3))
