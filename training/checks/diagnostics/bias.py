import os, numpy as np
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]*1000
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
ds=LeRobotDataset("gc"); np.set_printoptions(precision=2, suppress=True, linewidth=200)
for port,name in ((8000,"gc_B"),(8001,"gc_A")):
    cl=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",port)
    allerr=[]; alljerr=[]
    print("==",name,": error of the 20th chunk target vs the demo's state 20 steps later, over 8 noise draws per frame (mm: mean xyz, std xyz)")
    for ep in (0,1,33,49,71):
        b=int(ds.episode_data_index["from"][ep]); e=int(ds.episode_data_index["to"][ep])
        S=np.stack([np.asarray(ds[b+i]["state"]) for i in range(min(e-b,220))])
        for step in (40,70,100,110):
            it=ds[b+step]; st=S[step].astype(np.float32)
            obs={"observation/state":st,"observation/image":u8(it["image"]),"observation/wrist_image":u8(it["wrist_image"]),"prompt":it["task"]}
            if port==8001:
                for side in ("left","right"):
                    obs[f"observation/{side}_tactile_clip"]=np.ascontiguousarray(np.stack([u8(ds[max(b+step+o,b)][f"{side}_tactile_image"]) for o in (-6,-4,-2,0)]))
            tgt=S[min(step+20,len(S)-1)]; errs=[]; jerrs=[]
            for k in range(8):
                o=dict(obs)
                if k==0: o["reset"]=True
                ch=np.asarray(cl.infer(o)["actions"]); errs.append(fk(ch[19,:7])-fk(tgt[:7])); jerrs.append((ch[19,:7]-tgt[:7])*1000)
            errs=np.array(errs); allerr.append(errs.mean(0)); alljerr.append(np.array(jerrs).mean(0))
            print(ep,step,"mean",errs.mean(0),"std",errs.std(0),"| joint err mrad mean",np.array(jerrs).mean(0).round(1))
    A=np.array(allerr); J=np.array(alljerr)
    print(name,"OVERALL mean error xyz mm:",A.mean(0),"(std across frames",A.std(0),") ; joint mean mrad",J.mean(0).round(2))
