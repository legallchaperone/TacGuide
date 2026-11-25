import os, sys, numpy as np
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
ds=LeRobotDataset("gc"); c=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",8000)
def q(item, shift):
    f=lambda im: np.clip(im.astype(np.int16)+shift,0,255).astype(np.uint8)
    obs={"observation/state":np.asarray(item["state"],dtype=np.float32),"observation/image":f(u8(item["image"])),"observation/wrist_image":f(u8(item["wrist_image"])),"prompt":item["task"],"reset":True}
    return np.asarray(c.infer(obs)["actions"])
print("ep step | lowest flange point of chunk, original images (x y z mm) | same with +7 brightness | difference mm | demo bottom")
for ep in (3,27,49,71,88):
    b=int(ds.episode_data_index["from"][ep]); e=int(ds.episode_data_index["to"][ep])
    S=np.stack([np.asarray(ds[i]["state"]) for i in range(b,e,1)][:140])
    zb=np.array([fk(s[:7]) for s in S]); bot=zb[zb[:,2].argmin()]
    for step in (40,60):
        it=ds[b+step]; out=[]
        for sh in (0,7):
            ch=q(it,sh); P=np.array([fk(x[:7]) for x in ch]); out.append(P[P[:,2].argmin()])
        print(ep, step, (out[0]*1000).round(1), (out[1]*1000).round(1), ((out[1]-out[0])*1000).round(1), (bot*1000).round(1))
