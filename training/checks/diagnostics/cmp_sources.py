"""Compare image sources of the seed-0 scene with official training episode 0.
usage: cmp_sources.py [--predict]
Sources found automatically under eval_debug: replay_ep0 (eval-path render), render_static/*/sent,
expert_0/*.hdf5 (collection-path JPEG, compared at the same frame index)."""
import os, sys, glob, numpy as np, cv2, h5py
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
E="/data2/cwx/TacGuide/logs/eval_debug"; predict="--predict" in sys.argv
ds=LeRobotDataset("gc")
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]
def sharp(im): return cv2.Laplacian(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()
def res(a,b):
    off=a.reshape(-1,3).mean(0)-b.reshape(-1,3).mean(0)
    return "%.2f off %+.1f resid %.2f x%.2f"%(np.abs(a.astype(float)-b).mean(), off.mean(), np.abs(a.astype(float)-off-b).mean(), sharp(a)/sharp(b))
client=None; base={}
if predict:
    from openpi_client import websocket_client_policy
    client=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",8000)
def low(st,img,wr,prompt):
    ch=np.asarray(client.infer({"observation/state":st,"observation/image":np.ascontiguousarray(img),"observation/wrist_image":np.ascontiguousarray(wr),"prompt":prompt,"reset":True})["actions"])
    Q=np.array([fk(x[:7]) for x in ch]); return Q[Q[:,2].argmin()], ch
def report(name, frames):
    # frames: {step: (head224, wrist224, state or None)}
    for s,(h,w,st) in sorted(frames.items()):
        it=ds[s]; th=u8(it["image"]); tw=u8(it["wrist_image"]); line="%-34s step %3d | head %s | wrist %s"%(name,s,res(h,th),res(w,tw))
        if st is not None: line+=" | state diff %.5f"%np.abs(np.asarray(st).ravel()[:8]-np.asarray(it["state"])).max()
        if predict and s in (20,40):
            stt=np.asarray(it["state"],dtype=np.float32)
            if s not in base: base[s]=low(stt,th,tw,it["task"])
            p,ch=low(stt,h,w,it["task"]); line+=" | pred shift mm %s, %.0f mrad"%(((p-base[s][0])*1000).round(1),np.abs(ch[:,:7]-base[s][1][:,:7]).max()*1000)
        print(line)
def from_npz(d, steps=(0,20,40,60)):
    out={}
    for s in steps:
        f=f"{d}/q{s:05d}.npz"
        if os.path.exists(f):
            z=np.load(f); out[s]=(z["observation__image"],z["observation__wrist_image"],z["observation__state"] if "observation__state" in z.files else None)
    return out
report("replay_ep0 (eval path)", from_npz(E+"/replay_ep0"))
for d in sorted(glob.glob(E+"/render_static/*/sent")): report("render_static/"+d.split("/")[-2], from_npz(d))
for f in sorted(glob.glob(E+"/expert_0/*.hdf5")):
    h=h5py.File(f,"r"); j=np.asarray(h["embodiment/joint"]); out={}
    def dec(key,i):
        im=cv2.imdecode(np.frombuffer(h[key][i],dtype=np.uint8),cv2.IMREAD_COLOR)
        return im
    print("expert_0:", f, "frames", len(j), "raw frame shape", dec("observation/head/rgb",0).shape)
    for order in ("as-decoded","swapped"):
        out={}
        for s in (0,20,40,60):
            a=dec("observation/head/rgb",s); b=dec("observation/wrist/rgb",s)
            if order=="swapped": a=a[...,::-1]; b=b[...,::-1]
            out[s]=(cv2.resize(a,(224,224),interpolation=cv2.INTER_AREA),cv2.resize(b,(224,224),interpolation=cv2.INTER_AREA),np.r_[j[s,:8]])
        report("expert_0 JPEG ("+order+")", out)
