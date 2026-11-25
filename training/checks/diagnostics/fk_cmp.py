import numpy as np, pandas as pd, glob
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T
P="/data2/cwx/TacGuide/data/official_openvitac_lerobot/grasp_chip"
zb=[];off=[];rows=[]
for ep in range(100):
    t=pd.read_parquet(f"{P}/data/chunk-000/episode_{ep:06d}.parquet")
    q=np.stack(t["observation.joint_position"].to_numpy()); ee=np.stack(t["observation.eef_pose_quaternion"].to_numpy())
    dq=np.abs(np.diff(q,axis=0)).max(1); i=40
    while i<len(dq)-10 and not (dq[i:i+10]<1e-4).all(): i+=1
    T=fk(q[i]); zb.append(T[2,3]); off.append(ee[i,:3]-T[:3,3]); rows.append((T[:3,3],ee[i,:3], T[:3,2]))
zb=np.array(zb); off=np.array(off)
print("demo bottom flange z (DH): mean %.4f min %.4f max %.4f std %.4f"%(zb.mean(),zb.min(),zb.max(),zb.std()))
print("recorded ee minus DH flange: mean", off.mean(0).round(4), "std", off.std(0).round(4))
print("demo tool axis z-component at bottom: mean %.4f min %.4f"%(np.mean([r[2][2] for r in rows]), np.max([r[2][2] for r in rows])))
D="/data2/cwx/TacGuide/logs/eval_debug/trace_A_10000"
for s in (0,80,100,120,140,160,200,300,500):
    z=np.load(f"{D}/q{s:05d}.npz"); st=z["observation__state"]; T=fk(st[:7])
    print("eval step %4d flange xyz"%s, T[:3,3].round(4), "tool axis", T[:3,2].round(3), "grip mm %.1f"%(st[7]*1000))
t=pd.read_parquet(f"{P}/data/chunk-000/episode_000049.parquet"); q=np.stack(t["observation.joint_position"].to_numpy())
print("demo ep49 flange xyz at frames 0,80,100,120:", [fk(q[i])[:3,3].round(4).tolist() for i in (0,80,100,120)])
# demo descent time: first frame where flange z within 2 mm of bottom
ts=[]
for ep in range(0,100,5):
    t=pd.read_parquet(f"{P}/data/chunk-000/episode_{ep:06d}.parquet"); q=np.stack(t["observation.joint_position"].to_numpy())
    zz=np.array([fk(x)[2,3] for x in q[:200]]); ts.append(int(np.argmax(zz<zz.min()+0.002)))
print("demo frames to reach bottom:", ts)
