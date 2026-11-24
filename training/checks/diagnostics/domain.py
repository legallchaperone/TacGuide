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
    return T[:3,3]
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
ds=LeRobotDataset("gc"); c=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",8000)
R="/data2/cwx/TacGuide/logs/eval_debug/replay_ep0"
S=np.stack([np.asarray(ds[i]["state"]) for i in range(0,140)]); P=np.array([fk(s[:7]) for s in S]); bot=P[P[:,2].argmin()]
print("demo ep0 bottom flange (mm):", (bot*1000).round(1))
def run(state,img,wr,prompt):
    ch=np.asarray(c.infer({"observation/state":state,"observation/image":img,"observation/wrist_image":wr,"prompt":prompt,"reset":True})["actions"])
    Q=np.array([fk(x[:7]) for x in ch]); return Q[Q[:,2].argmin()], ch
print("step | lowest chunk point with TRAINING images | with REPLAY (sim-rendered) images | difference mm | gripper k=50 mm train/replay | max arm diff over chunk (mrad)")
for step in (0,20,40,60,80,100):
    z=np.load(f"{R}/q{step:05d}.npz"); it=ds[step]; st=np.asarray(it["state"],dtype=np.float32)
    a,ca=run(st,u8(it["image"]),u8(it["wrist_image"]),it["task"])
    b,cb=run(st,z["observation__image"],z["observation__wrist_image"],it["task"])
    print(step, (a*1000).round(1), (b*1000).round(1), ((b-a)*1000).round(1), "| %.1f / %.1f"%(ca[49,7]*1000,cb[49,7]*1000), "| %.1f"%(np.abs(ca[:,:7]-cb[:,:7]).max()*1000))
