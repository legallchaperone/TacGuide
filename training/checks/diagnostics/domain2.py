import os, io, numpy as np, cv2, av
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
steps=[0,20,40,60,80,100,120,140,160,180,200]
pairs={s:(np.load(f"{R}/q{s:05d}.npz"), ds[s]) for s in steps}
# calibrate: per-channel linear map a*x+b and blur sigma that best map replay -> train, on steps not used for the test
def blur(im,s): return im if s<=0 else cv2.GaussianBlur(im,(0,0),s)
def fit(key,ek,sig):
    X=[];Y=[]
    for s in steps:
        z,it=pairs[s]; X.append(blur(z[ek],sig).reshape(-1,3).astype(float)); Y.append(u8(it[key]).reshape(-1,3).astype(float))
    X=np.concatenate(X);Y=np.concatenate(Y); co=[np.polyfit(X[:,i],Y[:,i],1) for i in range(3)]
    err=np.mean([np.abs(np.polyval(co[i],X[:,i])-Y[:,i]).mean() for i in range(3)]); return co,err
cal={}
for key,ek in (("image","observation__image"),("wrist_image","observation__wrist_image")):
    best=min(((fit(key,ek,s),s) for s in (0,0.4,0.6,0.8,1.0,1.2,1.5)), key=lambda t:t[0][1])
    (co,err),sig=best; cal[ek]=(co,sig); print(key,"best blur sigma",sig,"gain/offset per channel",[(round(a,3),round(b,1)) for a,b in co],"residual mean abs",round(err,2),"| no-correction residual",round(np.mean([np.abs(pairs[s][0][ek].astype(float)-u8(pairs[s][1][key])).mean() for s in steps]),2))
def corr(im,ek,mode):
    co,sig=cal[ek]; x=im
    if mode in ("blur","both"): x=blur(x,sig)
    x=x.astype(float)
    if mode in ("gain","both"): x=np.stack([np.polyval(co[i],x[...,i]) for i in range(3)],-1)
    return np.clip(x+0.5,0,255).astype(np.uint8)
def run(state,img,wr,prompt):
    ch=np.asarray(c.infer({"observation/state":state,"observation/image":np.ascontiguousarray(img),"observation/wrist_image":np.ascontiguousarray(wr),"prompt":prompt,"reset":True})["actions"])
    Q=np.array([fk(x[:7]) for x in ch]); return Q[Q[:,2].argmin()], ch
print("step | lowest-point error vs training-image prediction (mm xyz) and max chunk diff (mrad) for: raw replay | brightness only | blur only | both")
for s in (0,20,40,60):
    z,it=pairs[s]; st=np.asarray(it["state"],dtype=np.float32)
    a,ca=run(st,u8(it["image"]),u8(it["wrist_image"]),it["task"]); out=[]
    for mode in ("raw","gain","blur","both"):
        b,cb=run(st,corr(z["observation__image"],"observation__image",mode),corr(z["observation__wrist_image"],"observation__wrist_image",mode),it["task"])
        out.append("%s %.0f"%(((b-a)*1000).round(1), np.abs(ca[:,:7]-cb[:,:7]).max()*1000))
    print(s," | ".join(out))
