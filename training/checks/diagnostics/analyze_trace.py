import numpy as np, glob, pandas as pd, json
D="/data2/cwx/TacGuide/logs/eval_debug/trace_A_10000"
fs=sorted(glob.glob(D+"/q*.npz"))
z=np.load(fs[0]); print("keys", list(z.keys()))
S=[];C=[]
for f in fs:
    z=np.load(f); S.append(z["observation__state"]); C.append(z["action_chunk"])
S=np.stack(S); C=np.stack(C)
steps=[int(f.split("/q")[-1][:5]) for f in fs]
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print("query step | state arm(7) | grip mm | chunk: arm move mrad at k=1,10,20,30,50 | grip mm at k=1,10,20,30,50")
for i,s in enumerate(steps[:16]+steps[25:60:8]):
    j=steps.index(s)
    mv=[np.abs(C[j,k,:7]-S[j,:7]).max()*1000 for k in (0,9,19,29,49)]
    print(s, S[j,:7].round(4), round(S[j,7]*1000,1), "|", np.round(mv,1), "|", np.round(C[j,[0,9,19,29,49],7]*1000,1))
# demos
P="/data2/cwx/TacGuide/data/official_openvitac_lerobot/grasp_chip"
bot=[];gfin=[];hold=[];zb=[];q4=[]
for ep in range(100):
    t=pd.read_parquet(f"{P}/data/chunk-000/episode_{ep:06d}.parquet")
    q=np.stack(t["observation.joint_position"].to_numpy()); f=np.stack(t["observation.finger_position"].to_numpy())[:,0]
    ee=np.stack(t["observation.eef_pose_quaternion"].to_numpy())
    dq=np.abs(np.diff(q,axis=0)).max(1)
    # first still run >=10 frames after frame 40
    i=40
    while i<len(dq)-10 and not (dq[i:i+10]<1e-4).all(): i+=1
    j=i
    while j<len(dq) and dq[j]<1e-4: j+=1
    bot.append(q[i]); zb.append(ee[i,2]); gfin.append(f[j]); hold.append((i,j))
    # frames between gripper settling and lift
    k=i
    while k<j and abs(f[k]-f[j])>2e-4: k+=1
    q4.append((j-k, f[i]))
bot=np.stack(bot)
print("demo bottom joints mean", bot.mean(0).round(4)); print("std", bot.std(0).round(4)); print("min", bot.min(0).round(4)); print("max", bot.max(0).round(4))
print("demo bottom ee z: mean %.4f min %.4f max %.4f"%(np.mean(zb),np.min(zb),np.max(zb)))
print("demo final grip mm: mean %.1f min %.1f max %.1f; pct10/50/90"%(np.mean(gfin)*1000,np.min(gfin)*1000,np.max(gfin)*1000), np.percentile(np.array(gfin)*1000,[10,50,90]).round(1))
h=np.array(hold); print("demo still start mean %.0f, lift start mean %.0f, duration min/mean/max"%(h[:,0].mean(),h[:,1].mean()), (h[:,1]-h[:,0]).min(), (h[:,1]-h[:,0]).mean(), (h[:,1]-h[:,0]).max())
w=np.array([a for a,_ in q4]); print("frames gripper is settled before lift: min/mean/max", w.min(), w.mean(), w.max())
# nearest demo bottom pose to eval bottom
eb=np.array([-0.1382, 0.4409, -0.0582, -2.3778, 0.0538, 2.812, 0.531])
d=np.abs(bot-eb).max(1); k=int(d.argmin()); print("eval bottom vs nearest demo bottom (ep %d): max joint diff %.4f rad"%(k,d[k])); print(" demo", bot[k].round(4), "z", round(zb[k],4), "final grip", round(gfin[k]*1000,1))
# linear map joints->z around bottom from demos to estimate eval z
A=np.c_[bot,np.ones(len(bot))]; coef,*_=np.linalg.lstsq(A,np.array(zb),rcond=None)
print("eval bottom ee z estimated from demos (linear fit): %.4f ; fit resid max %.4f"%(np.r_[eb,1]@coef, np.abs(A@coef-np.array(zb)).max()))
jb=steps.index(100) if 100 in steps else 5
print("eval state at step 100", S[jb].round(4))
