import numpy as np, glob
def fk(q):
    a=[0,0,0,0.0825,-0.0825,0,0.088,0]; d=[0.333,0,0.316,0,0.384,0,0,0.107]; al=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
    th=list(q)+[0.0]; T=np.eye(4)
    for i in range(8):
        ct,st,ca,sa=np.cos(th[i]),np.sin(th[i]),np.cos(al[i]),np.sin(al[i])
        T=T@np.array([[ct,-st,0,a[i]],[st*ca,ct*ca,-sa,-d[i]*sa],[st*sa,ct*sa,ca,d[i]*ca],[0,0,0,1]])
    return T[:3,3]
D="/data2/cwx/TacGuide/logs/eval_debug/trace_A_10000"
fs=sorted(glob.glob(D+"/q*.npz")); S=[];C=[]
for f in fs:
    z=np.load(f); S.append(z["observation__state"]); C.append(z["action_chunk"])
S=np.stack(S);C=np.stack(C)
print("query | commanded k=20 flange z | achieved z at next query | arm tracking err mrad (achieved - commanded k=20) | grip cmd k=20 mm -> achieved mm")
for j in range(0,14):
    cz=fk(C[j,19,:7])[2]; az=fk(S[j+1,:7])[2]
    print(j*20, "%.4f %.4f"%(cz,az), np.round((S[j+1,:7]-C[j,19,:7])*1000,1), "| %.1f -> %.1f"%(C[j,19,7]*1000,S[j+1,7]*1000))
print("chunk at step 80: flange z along k:", np.round([fk(C[4,k,:7])[2] for k in range(0,50,3)],4))
print("chunk at step 60: flange z along k:", np.round([fk(C[3,k,:7])[2] for k in range(0,50,3)],4))
print("chunk at step 100: flange z along k:", np.round([fk(C[5,k,:7])[2] for k in range(0,50,3)],4))
