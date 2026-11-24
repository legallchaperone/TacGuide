import numpy as np, glob, pandas as pd, h5py, json, os
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]
E="/data2/cwx/TacGuide/logs/eval_debug"
h=h5py.File(E+"/expert_10000/10000.hdf5","r"); ej=np.asarray(h["embodiment/joint"]); EP=np.array([fk(x[:7]) for x in ej])*1000
chip=np.array([493.7,-95.1])
np.set_printoptions(precision=1, suppress=True, linewidth=220)
for name in ("c2_A_10000","c2_B_10000"):
    fs=sorted(glob.glob(f"{E}/{name}/q*.npz"))
    if not fs: print(name,"no files", os.listdir(f"{E}/{name}")[:8]); continue
    print("==",name,len(fs),"queries; keys",np.load(fs[0]).files)
    print("step | policy flange xyz mm | expert flange at same step | chunk end-of-20 target xyz | chunk lowest point xyz | grip state mm, cmd k20")
    for f in fs[:12]:
        z=np.load(f); s=int(os.path.basename(f)[1:6]); st=z["observation__state"].ravel(); c=z["action_chunk"]
        P=fk(st[:7])*1000; Q=np.array([fk(x[:7]) for x in c])*1000
        print(s, P, EP[min(s,len(EP)-1)], Q[19], Q[Q[:,2].argmin()], "| %.1f %.1f"%(st[7]*1000,c[19,7]*1000))
# training distribution of chip (= demo bottom flange xy)
Pq="/data2/cwx/TacGuide/data/official_openvitac_lerobot/grasp_chip"
B=[]
for ep in range(100):
    t=pd.read_parquet(f"{Pq}/data/chunk-000/episode_{ep:06d}.parquet"); q=np.stack(t["observation.joint_position"].to_numpy()); ee=np.stack(t["observation.eef_pose_quaternion"].to_numpy())
    i=int(ee[:200,2].argmin()); B.append(ee[i,:3]*1000)
B=np.array(B)
print("training grasp xy: x range %.1f..%.1f, y range %.1f..%.1f"%(B[:,0].min(),B[:,0].max(),B[:,1].min(),B[:,1].max()))
d=np.linalg.norm(B[:,:2]-chip,axis=1); o=np.argsort(d)[:5]
print("eval seed 10000 chip",chip,"nearest training grasps (ep, xy, dist mm):",[(int(k),B[k,:2].round(1).tolist(),round(float(d[k]),1)) for k in o])
print("count of training grasps within 5/10/15 mm:", int((d<5).sum()), int((d<10).sum()), int((d<15).sum()))
# typical nearest-neighbour spacing in training
from itertools import combinations
D=np.linalg.norm(B[:,None,:2]-B[None,:,:2],axis=2); np.fill_diagonal(D,1e9); print("training nearest-neighbour spacing: median %.1f mm, 90th pct %.1f mm"%(np.median(D.min(1)),np.percentile(D.min(1),90)))
