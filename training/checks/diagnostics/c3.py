import os, sys, glob, json, numpy as np, cv2
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
ds=LeRobotDataset("gc"); E="/data2/cwx/TacGuide/logs/eval_debug"
np.set_printoptions(precision=1, suppress=True, linewidth=220)
cl=websocket_client_policy.WebsocketClientPolicy("127.0.0.1",8001)
def clip(b,t,key):
    return np.ascontiguousarray(np.stack([u8(ds[max(b+t+o,b)][key]) for o in (-6,-4,-2,0)]))
def infer(st,h,w,l,r,prompt):
    return np.asarray(cl.infer({"observation/state":st,"observation/image":np.ascontiguousarray(h),"observation/wrist_image":np.ascontiguousarray(w),"observation/left_tactile_clip":np.ascontiguousarray(l),"observation/right_tactile_clip":np.ascontiguousarray(r),"prompt":prompt,"reset":True})["actions"])
for seed,ep in ((0,0),(1,1)):
    d=f"{E}/c3_A_{seed}"; fs=sorted(glob.glob(d+"/q*.npz"))
    if not fs: print("no data",d); continue
    b=int(ds.episode_data_index["from"][ep]); e=int(ds.episode_data_index["to"][ep]); n=e-b
    DS=np.stack([np.asarray(ds[b+i]["state"]) for i in range(min(n,260))]); DP=np.array([fk(s[:7]) for s in DS])
    print(f"== c3_A_{seed} vs demo {ep}: demo bottom", DP[DP[:,2].argmin()], "demo grip at lift %.1f"%(DS[:,7].min()*1000), "START_CHECK", open(d+"/START_CHECK.json").read()[:200].replace("\n"," ") if os.path.exists(d+"/START_CHECK.json") else "")
    print("step | sim flange | demo flange same step | sim grip / demo grip | chunk@20 target (sim obs) | demo flange at step+20")
    for f in fs[:11]:
        z=np.load(f); s=int(os.path.basename(f)[1:6]); st=z["observation__state"].ravel(); c=z["action_chunk"]
        print(s, fk(st[:7]), DP[min(s,len(DP)-1)], "| %.1f / %.1f |"%(st[7]*1000,DS[min(s,len(DS)-1),7]*1000), fk(c[19,:7]), DP[min(s+20,len(DP)-1)])
    # step-0 swap test: same state; which input stream changes the prediction?
    z=np.load(fs[0]); it=ds[b]; st=np.asarray(it["state"],dtype=np.float32)
    print("state diff at step 0: %.6f"%np.abs(z["observation__state"].ravel()-st).max())
    T=dict(h=u8(it["image"]),w=u8(it["wrist_image"]),l=clip(b,0,"left_tactile_image"),r=clip(b,0,"right_tactile_image"))
    S=dict(h=z["observation__image"],w=z["observation__wrist_image"],l=z["observation__left_tactile_clip"],r=z["observation__right_tactile_clip"])
    for k in "hwlr": print("  pixel diff step0", k, "%.2f"%np.abs(T[k].astype(float)-S[k]).mean(), "sim mean %.1f train mean %.1f"%(S[k].mean(),T[k].mean()))
    base=infer(st,T["h"],T["w"],T["l"],T["r"],it["task"]); demo20=DP[20]; demo50=DP[min(50,len(DP)-1)]
    def rep(name,ch): print("  %-28s target@20 %s  @50 %s | vs train-obs prediction: max joint diff %.0f mrad"%(name,fk(ch[19,:7]),fk(ch[49,:7]),np.abs(ch[:,:7]-base[:,:7]).max()*1000))
    print("  demo actual flange at step 20:",demo20," step 50:",demo50)
    rep("all training inputs",base)
    rep("all sim inputs",infer(st,S["h"],S["w"],S["l"],S["r"],it["task"]))
    rep("only head from sim",infer(st,S["h"],T["w"],T["l"],T["r"],it["task"]))
    rep("only wrist from sim",infer(st,T["h"],S["w"],T["l"],T["r"],it["task"]))
    rep("only tactile from sim",infer(st,T["h"],T["w"],S["l"],S["r"],it["task"]))
    print("  sim's own chunk at step 0 (recorded): target@20", fk(z["action_chunk"][19,:7]), "@50", fk(z["action_chunk"][49,:7]))
