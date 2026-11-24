import os, numpy as np, cv2
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
rng=np.random.default_rng(0)
def noise(im): return np.clip(im.astype(float)+rng.normal(0,4,im.shape),0,255).astype(np.uint8)
def sharpen(im):
    b=cv2.GaussianBlur(im,(0,0),1.0); return cv2.addWeighted(im,1.6,b,-0.6,0)
def shift(im): return np.roll(im,1,axis=1)
def jpeg(im):
    ok,e=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,90]); return cv2.imdecode(e,cv2.IMREAD_COLOR)
P={"noise sigma 4":noise,"sharpen":sharpen,"shift 1 px":shift,"jpeg q90":jpeg}
def run(st,img,wr,prompt):
    ch=np.asarray(c.infer({"observation/state":st,"observation/image":np.ascontiguousarray(img),"observation/wrist_image":np.ascontiguousarray(wr),"prompt":prompt,"reset":True})["actions"])
    Q=np.array([fk(x[:7]) for x in ch]); return Q[Q[:,2].argmin()], ch
print("ep step | change of the predicted lowest point (mm xyz) and max chunk change (mrad) under: "+" | ".join(P))
for ep in (0,49):
    b0=int(ds.episode_data_index["from"][ep])
    for s in (20,40):
        it=ds[b0+s]; st=np.asarray(it["state"],dtype=np.float32); im=u8(it["image"]); wr=u8(it["wrist_image"])
        a,ca=run(st,im,wr,it["task"]); out=[]
        for name,f in P.items():
            b,cb=run(st,f(im),f(wr),it["task"]); out.append("%s %.0f"%(((b-a)*1000).round(1),np.abs(ca[:,:7]-cb[:,:7]).max()*1000))
        print(ep,s," | ".join(out))
