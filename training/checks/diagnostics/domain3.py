import os, io, numpy as np, cv2, av, fractions
os.environ["HF_LEROBOT_HOME"]="/data2/cwx/TacGuide/data/training/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy
P="/data2/cwx/TacGuide/data/official_openvitac_lerobot/grasp_chip/videos/chunk-000/observation.images.head/episode_000000.mp4"
c0=av.open(P); s=c0.streams.video[0]
print("dataset video: codec", s.codec_context.name, "profile", s.codec_context.profile, "bit_rate", s.bit_rate, "frames", s.frames, "size_bytes", os.path.getsize(P), "bytes/frame", os.path.getsize(P)//max(s.frames,1))
kinds=[]
for i,pk in enumerate(c0.demux(s)):
    if i<40: kinds.append("I" if pk.is_keyframe else "p")
print("first packets keyframe pattern:", "".join(kinds))
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
def h264(im, crf, size=(480,270)):
    big=cv2.resize(im,size,interpolation=cv2.INTER_CUBIC)
    buf=io.BytesIO(); o=av.open(buf,"w",format="mp4"); st=o.add_stream("libx264",rate=60); st.width,st.height=size; st.pix_fmt="yuv420p"; st.options={"crf":str(crf),"preset":"medium"}
    for _ in range(3):
        for p in st.encode(av.VideoFrame.from_ndarray(big,format="rgb24")): o.mux(p)
    for p in st.encode(): o.mux(p)
    o.close(); buf.seek(0); i=av.open(buf); fr=[f.to_ndarray(format="rgb24") for f in i.decode(video=0)][-1]
    return cv2.resize(fr,(224,224),interpolation=cv2.INTER_AREA)
ds=LeRobotDataset("gc"); c=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",8000)
R="/data2/cwx/TacGuide/logs/eval_debug/replay_ep0"
off={"observation__image":np.array([-6.5,-6.5,-7.3]),"observation__wrist_image":np.array([-3.5,-3.8,-4.2])}
def chain(im,ek,crf): return np.clip(h264(im,crf).astype(float)+off[ek]+0.5,0,255).astype(np.uint8)
for crf in (18,23,28,33):
    r=[]
    for s_ in (0,60,100,200):
        z=np.load(f"{R}/q{s_:05d}.npz"); it=ds[s_]
        r.append(np.abs(chain(z["observation__image"],"observation__image",crf).astype(float)-u8(it["image"])).mean())
    print("crf",crf,"head residual vs training frame", round(float(np.mean(r)),2))
def run(st,img,wr,prompt):
    ch=np.asarray(c.infer({"observation/state":st,"observation/image":np.ascontiguousarray(img),"observation/wrist_image":np.ascontiguousarray(wr),"prompt":prompt,"reset":True})["actions"])
    Q=np.array([fk(x[:7]) for x in ch]); return Q[Q[:,2].argmin()], ch
print("step | lowest-point error vs training-image prediction (mm) and max chunk diff (mrad): raw replay | h264 crf23+offset | crf28+offset | crf33+offset")
for s_ in (0,20,40,60):
    z=np.load(f"{R}/q{s_:05d}.npz"); it=ds[s_]; st=np.asarray(it["state"],dtype=np.float32)
    a,ca=run(st,u8(it["image"]),u8(it["wrist_image"]),it["task"]); out=[]
    b,cb=run(st,z["observation__image"],z["observation__wrist_image"],it["task"]); out.append("%s %.0f"%(((b-a)*1000).round(1),np.abs(ca[:,:7]-cb[:,:7]).max()*1000))
    for crf in (23,28,33):
        b,cb=run(st,chain(z["observation__image"],"observation__image",crf),chain(z["observation__wrist_image"],"observation__wrist_image",crf),it["task"])
        out.append("%s %.0f"%(((b-a)*1000).round(1),np.abs(ca[:,:7]-cb[:,:7]).max()*1000))
    print(s_," | ".join(out))
