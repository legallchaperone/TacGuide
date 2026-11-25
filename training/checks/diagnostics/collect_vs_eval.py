import h5py, numpy as np, cv2
h=h5py.File("/data2/cwx/TacGuide/logs/eval_debug/expert_10000/10000.hdf5","r")
z=np.load("/data2/cwx/TacGuide/logs/eval_debug/trace_A_10000/q00000.npz")
def sharp(im): return cv2.Laplacian(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()
print("state diff frame0:", np.abs(np.asarray(h["embodiment/joint"][0,:8])-z["observation__state"].ravel()).max())
for key,ek in (("observation/head/rgb","observation__image"),("observation/wrist/rgb","observation__wrist_image")):
    raw=cv2.imdecode(np.frombuffer(h[key][0],dtype=np.uint8),cv2.IMREAD_COLOR); e=z[ek]
    for name,im in (("as-decoded",raw),("swapped",raw[...,::-1])):
        a=cv2.resize(im,(224,224),interpolation=cv2.INTER_AREA)
        off=a.reshape(-1,3).mean(0)-e.reshape(-1,3).mean(0)
        print(key, raw.shape, name, "collect-JPEG mean", a.reshape(-1,3).mean(0).round(1), "eval-sent mean", e.reshape(-1,3).mean(0).round(1), "| mean abs diff %.2f, after offset %.2f, sharpness ratio x%.2f"%(np.abs(a.astype(float)-e).mean(), np.abs(a.astype(float)-off-e).mean(), sharp(np.ascontiguousarray(a))/sharp(e)))
for s in (0,20,40):
    raw=cv2.imdecode(np.frombuffer(h["observation/head/rgb"][s],dtype=np.uint8),cv2.IMREAD_COLOR)
    print("expert_10000 head frame",s,"mean (as decoded)",raw.reshape(-1,3).mean(0).round(1),"carpet patch",cv2.resize(raw,(224,224),interpolation=cv2.INTER_AREA)[150:220,120:220].reshape(-1,3).mean(0).round(1))
print("jpeg bytes/frame head:", np.mean([len(h["observation/head/rgb"][i]) for i in range(0,400,40)]))
