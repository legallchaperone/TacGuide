"""Per-episode miss table: python miss_table.py <glob of run dirs, e.g. 'c2_*' 'c3_*'>
For each run dir with q*.npz and a metadata json: chip start xy, policy's first-bottom flange pose,
miss vector (policy minus chip centre; expert grasps at the chip centre, z about 145 mm), gripper width, outcome."""
import sys, glob, json, os, numpy as np
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]*1000
def find(o,key):
    if isinstance(o,dict):
        if key in o: return o[key]
        for v in o.values():
            r=find(v,key)
            if r is not None: return r
    return None
E="/data2/cwx/TacGuide/logs/eval_debug"; rows=[]
print("run | chip xy | first-bottom flange xyz (step) | miss x y (mm), z vs 145 | grip mm at bottom+40 | outcome")
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(f"{E}/{pat}")):
        fs=sorted(glob.glob(d+"/q*.npz"))
        if not fs: continue
        S=np.stack([np.load(f)["observation__state"].ravel() for f in fs]); steps=[int(os.path.basename(f)[1:6]) for f in fs]
        P=np.array([fk(s[:7]) for s in S])
        # first bottom: first query where z stops decreasing (within 3 mm of the minimum over the first 12 queries)
        n=min(12,len(P)); i=int(np.argmax(P[:n,2]<P[:n,2].min()+3.0))
        meta=None
        for m in glob.glob(d+"/*metadata*.json")+glob.glob(d+"/*.json"):
            try:
                j=json.load(open(m)); c=find(j,"chip_start_pose")
                if c is not None: meta=j; break
            except Exception: pass
        chip=np.array(find(meta,"chip_start_pose")[:2])*1000 if meta else np.array([np.nan,np.nan])
        reason=find(meta,"terminal_reason") if meta else None; succ=find(meta,"success") if meta else None
        miss=P[i,:2]-chip; k=min(i+2,len(S)-1)
        rows.append((os.path.basename(d),miss))
        print("%-16s | %s | %s (%d) | %+.1f %+.1f, z %+.1f | %.1f | %s %s"%(os.path.basename(d),chip.round(1),P[i].round(1),steps[i],miss[0],miss[1],P[i,2]-145.0,S[k,7]*1000,reason,succ))
M=np.array([m for _,m in rows if np.isfinite(m).all()])
if len(M): print("mean miss x y: %+.1f %+.1f mm; mean |miss| %.1f mm; std x y %.1f %.1f; n=%d"%(M[:,0].mean(),M[:,1].mean(),np.linalg.norm(M,axis=1).mean(),M[:,0].std(),M[:,1].std(),len(M)))
