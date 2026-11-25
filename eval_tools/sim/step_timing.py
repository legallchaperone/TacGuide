import re,sys,statistics as st
for f in sys.argv[1:]:
    t=open(f,errors="ignore").read()
    seen={}
    for s,a,r in re.findall(r"Step\s+(\d+)\(.*?\), action\s+(\d+), FPS\s+[\d.]+, Running\s+([\d.]+)s",t): seen[int(s)]=(int(a),float(r))
    steps=sorted(seen)
    if len(steps)<3: print(f,"no steps"); continue
    d=[seen[s][1]-seen[steps[i-1]][1] for i,s in enumerate(steps) if i>0 and s==steps[i-1]+1]
    med=st.median(d); slow=sorted(x for x in d if x>med*1.6)
    print("/".join(f.split("/")[-5:-1]),"last action",seen[steps[-1]][0],"running %.0fs"%seen[steps[-1]][1],"| step median %.2f | policy-call n=%d median %.2f max %.2f"%(med,len(slow),st.median(slow) if slow else 0,slow[-1] if slow else 0),"| chunks",t.count("OpenPI action chunk"))
    tail=[l for l in t.replace("\r","\n").splitlines() if l.strip() and not l.lstrip().startswith("Step") and "[Warning]" not in l and "action chunk" not in l][-4:]
    for l in tail: print("     ",l[:230])
