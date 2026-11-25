# Summarise a trace dir: per query, state vs action chunk, and tactile image sharpness.
import numpy as np, glob, sys
d = sys.argv[1]
fs = sorted(glob.glob(d + "/q*.npz"))
print("queries:", len(fs))
z = np.load(fs[0]); print("keys/shapes:", {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print("step | state[0:7] | grip state | chunk[0]-state max|arm| | chunk[19]-state max|arm| | chunk[49]-state max|arm| | grip cmd 0/19/49 | tactile L/R std, dark-pixel frac")
prev = None
for f in fs:
    z = np.load(f); st = z["observation__state"].astype(float).ravel(); a = z["action_chunk"].astype(float)
    step = int(f.split("q")[-1].split(".")[0])
    def dm(i): return np.abs(a[i, :7] - st[:7]).max()
    tk = [k for k in z.files if "tactile" in k]
    tinfo = []
    for k in tk:
        im = z[k].astype(float); last = im[-1] if im.ndim == 4 else im
        tinfo.append("%.1f/%.3f" % (last.std(), (last.mean(axis=-1) < 40).mean()))
    moved = "" if prev is None else " dq_since_prev=%.4f" % np.abs(st[:7] - prev[:7]).max()
    if step % 100 == 0 or step < 200:
        print("%5d | %s | %.4f | %.4f | %.4f | %.4f | %.4f %.4f %.4f | %s%s" % (step, st[:7], st[7], dm(0), dm(19), dm(49), a[0, 7], a[19, 7], a[49, 7], " ".join(tinfo), moved))
    prev = st
