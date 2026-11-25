# Assemble per-seed folders for the training thread: export/<name>_<seed>/ with q*.npz, START_CHECK.json,
# metadata.json, result.json and the episode video. Usage: export_results.py <tag>=<name> [...]
import json, glob, os, shutil, sys
for arg in sys.argv[1:]:
    tag, name = arg.split("=")
    for f in sorted(glob.glob(f"/root/autodl-tmp/results/{tag}/[0-9]*.json")):
        r = json.load(open(f)); seed = r["seed"]; dst = f"/root/autodl-tmp/export/{name}_{seed}"
        if os.path.exists(dst + "/result.json"):
            continue
        os.makedirs(dst, exist_ok=True)
        tr = f"/root/autodl-tmp/trace/{tag}/{seed}/try{r['tries']}"
        for q in glob.glob(tr + "/*"):
            shutil.copy(q, dst)
        out = r["output"]
        for g in [out + "/metadata.json"] + glob.glob(out + "/video/*.mp4"):
            if os.path.exists(g): shutil.copy(g, dst)
        json.dump(r, open(dst + "/result.json", "w"))
        print("exported", dst, len(os.listdir(dst)), "files")
