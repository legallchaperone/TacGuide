import os, numpy as np, cv2
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds=LeRobotDataset("gc")
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
R="/data2/cwx/TacGuide/logs/eval_debug/replay_ep0"
rows=[]
print("stream step | mean RGB replay / train | state diff | phase-corr shift px (dx,dy) | ECC affine [[a,b,tx],[c,d,ty]] | mean abs diff before / after brightness match")
for step in (0,60,100,200):
    z=np.load(f"{R}/q{step:05d}.npz"); it=ds[step]
    sd=np.abs(z["observation__state"].ravel()-np.asarray(it["state"])).max()
    for k,ek in (("image","observation__image"),("wrist_image","observation__wrist_image")):
        a=z[ek].astype(np.uint8); b=u8(it[k])
        ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32); gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float32)
        ga_b=cv2.GaussianBlur(ga,(0,0),1.2); gb_b=cv2.GaussianBlur(gb,(0,0),1.2)
        (dx,dy),resp=cv2.phaseCorrelate(gb_b,ga_b,cv2.createHanningWindow((224,224),cv2.CV_32F))
        W=np.eye(2,3,dtype=np.float32)
        try:
            cc,W=cv2.findTransformECC(gb_b,ga_b,W,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,300,1e-7),None,5)
        except cv2.error as e:
            cc=-1
        d0=np.abs(a.astype(float)-b).mean(); off=a.reshape(-1,3).mean(0)-b.reshape(-1,3).mean(0)
        d1=np.abs(cv2.GaussianBlur(a,(0,0),1.0).astype(float)-off-cv2.GaussianBlur(b,(0,0),1.0)).mean()
        print(k, step, a.reshape(-1,3).mean(0).round(1), "/", b.reshape(-1,3).mean(0).round(1), "| %.5f"%sd, "| (%.2f, %.2f) resp %.2f"%(dx,dy,resp), "|", W.round(4).tolist(), "cc %.4f"%cc, "| %.1f / %.1f"%(d0,d1))
        if step in (0,100):
            diff=np.clip(np.abs(a.astype(int)-off.round().astype(int)-b)*4,0,255).astype(np.uint8)
            rows.append(np.concatenate([a,b,diff],1))
cv2.imwrite("/data2/cwx/TacGuide/logs/eval_debug/replay_vs_train.png", cv2.resize(np.concatenate(rows,0)[...,::-1],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
import json
m=json.load(open(R+"/metadata.json")); 
def walk(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+"/"+k)
    elif any(s in p.lower() for s in ("camera","resolution","render","width","height","fov","focal")): print(p, str(o)[:160])
walk(m)
