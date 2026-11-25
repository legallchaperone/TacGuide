"""Plot training loss curves from the training logs and print a one-line status per run.

  python training/checks/plot_loss.py [--out /data2/cwx/TacGuide/logs/loss_curves.png] [run ...]

Runs default to ga_A ga_B gc_B gc_C gc_D. Loss values are read from the progress lines in
$TG/logs/train_<run>.log, or for the distillation runs from the newest
$TG/queue/logs/train_<run>*.log (last value seen for each step), smoothed with a 100-step mean.
Runs without a log yet are skipped.
"""
import argparse, glob, os, re, subprocess, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TG = os.environ.get("TG", "/data2/cwx/TacGuide")
TOTAL = {"ga_A": 20000, "ga_B": 20000, "gc_B": 20000, "gc_A": 10000, "gc_C": 10000, "gc_D": 10000}
LABEL = {"ga_A": "ga_A  gear assembly, vision + touch", "ga_B": "ga_B  gear assembly, vision only",
         "gc_B": "gc_B  grasp chip, vision only (continued from 10k)", "gc_A": "gc_A  grasp chip, vision + touch",
         "gc_C": "gc_C  grasp chip, distilled student", "gc_D": "gc_D  grasp chip, extrapolated student"}
COLOR = {"ga_A": "#1f77b4", "ga_B": "#ff7f0e", "gc_B": "#2ca02c", "gc_A": "#9467bd", "gc_C": "#d62728", "gc_D": "#8c564b"}
PAT = re.compile(r"loss=([0-9.eE+-]+), lr=([0-9.eE+-]+), step=([0-9]+)")


def read(run):
    path = f"{TG}/logs/train_{run}.log"
    if not os.path.exists(path):
        queued = sorted(glob.glob(f"{TG}/queue/logs/train_{run}*.log"), key=os.path.getmtime)
        if not queued:
            return None
        path = queued[-1]
    data = {}
    with open(path, errors="ignore") as f:
        for m in PAT.finditer(f.read().replace("\r", "\n")):
            data[int(m.group(3))] = float(m.group(1))
    if not data:
        return None
    steps = np.array(sorted(data)); loss = np.array([data[s] for s in steps])
    return steps, loss, os.path.getmtime(path)


def smooth(y, k=100):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0)); out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["ga_A", "ga_B", "gc_B", "gc_C", "gc_D"])
    ap.add_argument("--out", default=f"{TG}/logs/loss_curves.png")
    args = ap.parse_args()
    sessions = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    now = time.time()
    for run in args.runs:
        r = read(run)
        if r is None:
            continue
        steps, loss, mtime = r
        ax.plot(steps, loss, color=COLOR.get(run, "gray"), alpha=0.15, linewidth=0.6)
        ax.plot(steps, smooth(loss), color=COLOR.get(run, "gray"), linewidth=1.8, label=LABEL.get(run, run))
        alive = f"train_{run}:" in sessions or subprocess.run(
            ["pgrep", "-f", f"(exp-name {run} |train_distill.py {run}( |$))"], capture_output=True).returncode == 0
        last = steps[-1]; recent = loss[-100:].mean(); total = TOTAL.get(run, 0)
        print(f"{run}: step {last}/{total}, mean loss of last 100 steps {recent:.4f}, "
              f"{'RUNNING' if alive else 'NOT RUNNING'}, log updated {int((now - mtime) / 60)} min ago, "
              f"nan={'yes' if not np.isfinite(loss).all() else 'no'}")
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("flow-matching loss (log scale)")
    ax.set_title("TacGuide training loss (thick line: 100-step mean) — " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    ax.grid(True, which="both", alpha=0.25); ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
