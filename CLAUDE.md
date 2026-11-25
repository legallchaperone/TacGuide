# TacGuide: notes for Claude working from Simon's Mac

Simon manages agents from his Mac; the work itself runs on two remote machines. Read this before
touching either of them. Reply to Simon in Chinese; give times in US Eastern.

## What the project is

Tactile extrapolation on the OpenViTac benchmark: a teacher with touch (A) guides a student
without touch. Four policies per task, all π0.5-based:

| Name | Model | Inputs | How it is trained |
| --- | --- | --- | --- |
| `<task>_A` | OpenVTLA (π0.5 + AnyTouch2 tactile tokens) | vision + touch | plain flow matching |
| `<task>_B` | π0.5 | vision | plain flow matching |
| `<task>_C` | π0.5 | vision | distilled from frozen A (w = 1) |
| `<task>_D_w*` | π0.5 | vision | regresses v_B + w (v_A − v_B), w > 1 |

Tasks so far: `gc` (Grasp Chip) and `ga` (Gear Assembly). The plan, decisions and results live in
the project's plan document; `training/TACGUIDE.md` has the commands and the list of deviations
from the plan; `training/checks/diagnostics/README.md` has the evaluation findings.

## Repository and branches

`github.com/legallchaperone/TacGuide` is Simon's fork of OpenViTac. It is public: never commit
host names, ports, keys, tokens or personal details.

| Branch | Content | Owner |
| --- | --- | --- |
| `main` | upstream OpenViTac, untouched | nobody; do not commit here |
| `tacguide` | data conversion, A/B training wrapper, policy server, sim client changes, checks, this file | training work |
| `distill-guidance` | based on `tacguide`; adds `training/tacguide/` (distillation loss, guided-inference server, GPU queue) | distillation work |

The repo carries large simulation assets. For training-side work clone sparsely:

```bash
git clone --filter=blob:none --sparse git@github.com:legallchaperone/TacGuide.git
cd TacGuide && git sparse-checkout set training policy scripts task_config docs eval_tools
```

Commit small and often, push after each step. Do not open pull requests unless Simon asks.

## Machines

Both aliases exist only on the Mac (`~/.ssh/config`, key login). Nothing heavy belongs on the Mac.

### Training machine: `ssh cwx`

Shared lab server, 8× RTX PRO 6000 (96 GB each), many other users.

| What | Where |
| --- | --- |
| Checkout used by running A/B training (branch `tacguide`) | `~/TacGuide` |
| Checkout for distillation work (branch `distill-guidance`) | `~/TacGuide-distill` |
| Checkout for edits (safe to switch branches) | `~/TacGuide-edit` |
| Everything large | `/data2/cwx/TacGuide/` |
| Converted datasets, stats | `/data2/cwx/TacGuide/data/training/{lerobot,stats}/<task>` |
| Official downloads | `/data2/cwx/TacGuide/data/official_openvitac_lerobot/<task_name>` |
| Base weights | `/data2/cwx/TacGuide/weights/{pi05_base_pytorch,anytouch2}` |
| Checkpoints | `/data2/cwx/TacGuide/outputs/training/<openvtla\|pi05>/<task>/<run>/<step>` (20 GB each) |
| Shared Python env | `/data2/cwx/TacGuide/venv` (do not `uv sync` or add packages casually) |
| Logs, watcher status, loss plot | `/data2/cwx/TacGuide/logs/` |
| Evaluation traces and sim hand-off files | `/data2/cwx/TacGuide/logs/eval_debug/` |

In each checkout `data`, `weights` and `outputs` are symlinks into `/data2/cwx/TacGuide`.

Rules that have each cost hours when broken:

- Start every command with `source training/tacguide_env.sh`. It puts all caches on the data
  disk and unsets the proxy from `~/.bashrc`, which is dead. GitHub (SSH), PyPI, Google storage and
  ModelScope work directly; huggingface.co does not (download on the Mac and copy over).
- Never edit, switch branches or pull in `~/TacGuide` while training runs from it. Edit in
  `~/TacGuide-edit` (or on the Mac) and push.
- Long jobs: plain ssh plus one tmux server per job, `training/isolated.sh <name> "<command>"`
  (`tmux -L tg-<name> attach` to look; plain `tmux ls` does not list them). **Never use
  `systemd-run --user`**: units under the per-user service are watched by systemd-oomd (memory
  pressure above 50% for 20 s) and get killed within minutes while a model loads from the slow disk.
- Start one model-loading process at a time and wait until it runs before starting the next.
  Several loads in parallel have taken down every job of the account.
- GPUs are shared. Look at `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`
  right before launching; use an empty card, or one Simon has named. A batch-64 run needs about
  46 GB (A) or 54 GB (B, C, D students); a student with both frozen teachers needs a second card.
- `/data2` is a slow shared disk. A 20 GB checkpoint save or load normally takes minutes but has
  taken over an hour under load; that is a stall, not a crash.
- Use only our own directories and processes. Kill by PID, never by pattern.

Training (A and B):

```bash
training/isolated.sh train_ga_A \
  "cd ~/TacGuide; TG_TASK=ga TG_STEPS=20000 TG_SAVE_INTERVAL=2500 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True training/tacguide.sh train ga_A <gpu> 2>&1 | tee -a /data2/cwx/TacGuide/logs/train_ga_A.log"
# resume after an interruption: same command with --resume appended to the tacguide.sh call
```

`training/watch_ab.sh` restarts dead runs from the last checkpoint and writes
`logs/watch_<name>.status`; `training/checks/plot_loss.py` draws `logs/loss_curves.png` and prints
one status line per run. Runs resume only from saved checkpoints, so keep the save interval at
2,500 steps or less. The paper trains 20k steps; the released code defaults to 10k.

### Simulation machine: `ssh autodl`

Rented by the hour (RTX 4090). **Shut it down whenever it is idle**; only Simon can power it on,
switch hosts or save images. Isaac Sim 4.5 does not run on the training machine's GPUs, and π0.5
training does not fit on a 4090, which is why there are two machines.

- Everything under `/root/autodl-tmp` (system disk is small): `OpenViTac/` is this repo on branch
  `tacguide`; `run_env.sh` sets up the runtime; helper scripts are kept in `eval_tools/sim/`.
- The renderer must have DLSS: `launch.log` must not contain "NGX isn't enabled". Without the NGX
  library the camera images are brighter and sharper than the training data and every policy
  misses. The library lives in `/root/autodl-tmp/ngx_lib` and is added in `run_env.sh`.
- About 30% of simulator launches start broken. Run evaluations with `TACGUIDE_VALIDATE=1`: at
  step 0 both tactile images must show exactly 63 markers and the wrist image mean must be below
  132, otherwise the episode aborts and the seed is repeated.
- At most two simulator processes at once (10 GB each).
- Evaluation seeds start at 10000; collection seeds (the training demos) are below that.

## Evaluation flow

1. Simon powers on the sim machine. On `cwx`, start the policy servers on a free card, one at a
   time, each with an explicit `--step` (a run that is still training keeps adding checkpoints):
   `training/tacguide/serve_guided.py --strong <run> --step <n> --port <port>` from
   `~/TacGuide-distill`. Port convention: 8000 B, 8001 A, 8002–8005 CFG(A,B) with w = 1, 1.5, 2, 3,
   8006 C.
2. Tunnel directly from `cwx` to the sim machine (reverse forwards of those ports); routing
   through the Mac is far too slow.
3. On the sim machine: `scripts/eval_policy.py <task> <task config> openpi/abs_joint/<deploy config>
   --openpi_port <port> --start_seed 10000 ...`. Deploy configs: `deploy_grasp_chip_vision` (B, C,
   D), `deploy_grasp_chip_vision_tactile` (A, CFG), and the two `deploy_gear_assembly_*` files.
   The tactile configs make the client send 4-frame tactile clips (offsets −6, −4, −2, 0).
4. Copy traces, videos and metadata to `/data2/cwx/TacGuide/logs/eval_debug/`, stop the servers,
   shut the sim machine down.

## What needs Simon's own word

The permission check on the Mac lets these through only after Simon has written the specific
confirmation himself in the conversation; a go-ahead relayed by another agent does not count.
Read the conversation for it before asking again, and offer the exact command he could run.

- starting or restarting anything on a GPU;
- deleting checkpoints or datasets (permanent);
- adding ssh keys, running downloaded installers.

## Known state of the science (2026-11-25)

- `gc_A`, `gc_B`, `gc_C` at 10k steps fail in closed loop even on training scenes: each action chunk
  carries a constant offset of about 1 mrad on joints 1, 4 and 7, re-planning every 20 steps adds it
  up to a 5–8 mm miss, and the model gets no robot state (`discrete_state_input=False`, as released).
  The demonstrations contain no retries, so a missed grasp is never corrected.
- Open: whether 20k steps shrinks the offset (`gc_B` is being continued), and whether Gear Assembly
  behaves better (`ga_A`, `ga_B` training). Upstream issue #5 reports 0/50 on Grasp Chip for every
  learned policy because of very strict success rules.
