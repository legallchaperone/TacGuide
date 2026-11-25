# Summarise every grasp_chip eval output dir: per-seed results, counts, errors.
import json, glob, os, sys
root = "/root/autodl-tmp/OpenViTac/eval_result/openpi/grasp_chip"
for d in sorted(glob.glob(root + "/*/*/")):
    p = d + "force_task_progress.json"
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    att = s["attempts"]
    res = {int(k): v["result"] for k, v in att.items()}
    suc = sorted(k for k, v in res.items() if v == "success")
    fail = sorted(k for k, v in res.items() if v == "fail")
    err = sorted(k for k, v in res.items() if v not in ("success", "fail"))
    print(f"{d.split('/')[-3]}/{d.split('/')[-2]}: status={s['status']} first={s['first_seed']} max={s['max_seed']} "
          f"done={len(suc)+len(fail)} success={len(suc)} fail={len(fail)} error={len(err)}")
    if "-v" in sys.argv:
        print("   success seeds:", suc); print("   error seeds:", err)
