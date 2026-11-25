# first_bottom.py <export dir> ...: first-bottom flange position vs chip start pose, from the per-query trace
import numpy as np, glob, math, json, sys
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088]; d=[0.333,0,0.316,0,0.384,0,0]; al=[0,-math.pi/2,math.pi/2,math.pi/2,-math.pi/2,math.pi/2,math.pi/2]
    T=np.eye(4)
    for i in range(7):
        ct,st=math.cos(q[i]),math.sin(q[i]); ca,sa=math.cos(al[i]),math.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return (T@np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.107],[0,0,0,1]]))[:3,3]
def find(o,key):
    if isinstance(o,dict):
        for k,v in o.items():
            if k==key: return v
            r=find(v,key)
            if r is not None: return r
for d in sys.argv[1:]:
    fs=sorted(glob.glob(d+"/q*.npz"))
    if not fs: print(d,"no trace"); continue
    sts=np.array([np.load(f)["observation__state"] for f in fs]); xyz=np.array([fk(s[:7]) for s in sts]); steps=np.array([int(f[-9:-4]) for f in fs])
    r=json.load(open(d+"/result.json")); chip=find(json.load(open(d+"/metadata.json")),"chip_start_pose")
    low=np.where(xyz[:,2]<0.150)[0]
    if len(low)==0: print(d.split("/")[-1],r["result"],r.get("terminal_reason"),"never reached below 0.150; min z %.4f"%xyz[:,2].min()); continue
    i=low[0]; j=i
    while j+1<len(xyz) and steps[j+1]-steps[i]<=60 and xyz[j+1,2]<xyz[j,2]-2e-4: j+=1
    dx,dy=(xyz[j,0]-chip[0])*1000,(xyz[j,1]-chip[1])*1000
    print("%s %s/%s actions=%s | first bottom step %d z=%.1f mm, offset from chip dx=%+.1f dy=%+.1f mm | gripper min %.1f mm"%(d.split("/")[-1],r["result"],r.get("terminal_reason"),r.get("actions"),steps[j],xyz[j,2]*1000,dx,dy,sts[:,7].min()*1000))
