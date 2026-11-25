import h5py, numpy as np
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]
h=h5py.File("/data2/cwx/TacGuide/logs/eval_debug/expert_10000/10000.hdf5","r")
keys=[]; h.visit(lambda n: keys.append(n) if isinstance(h[n],h5py.Dataset) else None)
print([ (k,h[k].shape) for k in keys][:25]); print(dict(h.attrs).keys())
j=np.asarray(h["embodiment/joint"]); P=np.array([fk(x[:7]) for x in j]); n=len(j)
dq=np.abs(np.diff(j[:,:7],axis=0)).max(1); i=40
while i<n-11 and not (dq[i:i+10]<1e-4).all(): i+=1
k=i
while k<n-1 and dq[k]<1e-4: k+=1
print("frames",n,"first still frame",i,"lift starts",k)
print("expert bottom flange xyz (mm):",(P[i]*1000).round(1),"joints",j[i,:7].round(4))
print("expert gripper left finger mm: at descent end %.1f, at lift start %.1f, min over episode %.1f"%(j[i,7]*1000,j[k,7]*1000,j[:,7].min()*1000))
pol=np.array([0.4864,-0.1016,0.1384])
print("policy bottom flange (mm):",(pol*1000).round(1),"-> policy minus expert (mm):",((pol-P[i])*1000).round(1),"; policy gripper 28.3 vs expert %.1f"%(j[k,7]*1000))
print("chip centre (metadata) 493.7, -95.1 ; expert flange minus chip centre xy (mm):",((P[i,:2]-np.array([0.4937,-0.0951]))*1000).round(1))
print("expert z every 20 frames:",(P[::20,2]*1000).round(0).tolist()[:14])
