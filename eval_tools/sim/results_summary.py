import json, glob, os, sys
root = "/root/autodl-tmp/results"
for d in sorted(glob.glob(root + "/*/")):
    rs = [json.load(open(f)) for f in sorted(glob.glob(d + "[0-9]*.json"))]
    suc = [r["seed"] for r in rs if r["result"] == "success"]
    rej = len(glob.glob(d + "rejected/*_try?.json"))
    reasons = {}
    for r in rs: reasons[r.get("terminal_reason")] = reasons.get(r.get("terminal_reason"), 0) + 1
    print(f"{os.path.basename(d[:-1])}: done={len(rs)} success={len(suc)} rejected_attempts={rej} reasons={reasons}")
    if "-v" in sys.argv: print("   success seeds:", suc)
