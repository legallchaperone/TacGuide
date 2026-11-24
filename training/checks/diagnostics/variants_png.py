import os, glob, numpy as np, cv2
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds=LeRobotDataset("gc"); it=ds[0]
def u8(x):
    x=np.asarray(x); x=np.transpose(x,(1,2,0)) if x.shape[0]==3 else x
    return np.ascontiguousarray(np.clip(x*255+0.5,0,255).astype(np.uint8))
T={"image":u8(it["image"]),"wrist_image":u8(it["wrist_image"])}
def sharp(im): return cv2.Laplacian(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()
def res(a,b):
    d=np.abs(a.astype(float)-b).mean(); off=a.reshape(-1,3).mean(0)-b.reshape(-1,3).mean(0)
    d2=np.abs(a.astype(float)-off-b).mean()
    return d,off,d2,sharp(a)/sharp(b)
R="/data2/cwx/TacGuide/logs/eval_debug/render_variants"
z=np.load("/data2/cwx/TacGuide/logs/eval_debug/replay_ep0/q00000.npz")
print("variant | head: mean abs diff, offset RGB, diff after removing offset, sharpness ratio | wrist: same")
def line(name,h,w):
    a=res(h,T["image"]); b=res(w,T["wrist_image"])
    print("%-28s | %.2f %s %.2f x%.2f | %.2f %s %.2f x%.2f"%(name,a[0],a[1].round(1),a[2],a[3],b[0],b[1].round(1),b[2],b[3]))
line("replay_ep0 (earlier)", z["observation__image"], z["observation__wrist_image"])
for d in sorted(glob.glob(R+"/*/")):
    h=cv2.imread(d+"000000_image_sent.png"); w=cv2.imread(d+"000000_wrist_image_sent.png")
    if h is None or w is None: print(os.path.basename(d.rstrip("/")),"missing", os.listdir(d)[:5]); continue
    if h.shape[:2]!=(224,224): h=cv2.resize(h,(224,224),interpolation=cv2.INTER_AREA); w=cv2.resize(w,(224,224),interpolation=cv2.INTER_AREA)
    line(os.path.basename(d.rstrip("/")), h[...,::-1], w[...,::-1])
