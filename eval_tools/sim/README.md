# Simulation-side evaluation helpers

Scripts used on the simulation machine (under `/root/autodl-tmp`) during the Grasp Chip evaluation of
2026-11-24 and 2026-11-25, kept here so a new host or a new session can pick them up. They assume the
repo at `/root/autodl-tmp/OpenViTac` and the runtime from `run_env.sh`; paths inside are host-specific.

| File | Purpose |
| --- | --- |
| `stage0_all.sh`, `stage1_install.sh`, `run_env.sh` | install Isaac Sim / Isaac Lab / OpenViTac dependencies; runtime environment (adds the NGX library for DLSS) |
| `gate_render.py`, `render_variants.sh`, `static_run.sh`, `replay_ngx.sh` | render checks: camera output, render presets, DLSS on/off |
| `replay_server.py`, `replay_run.sh` | policy server that replays an official demonstration (pipeline control) |
| `expert_run.sh` | run the scripted expert on a seed |
| `run_seeds.sh`, `run_eval.sh`, `plan_cb.sh`, `ens_plan.sh`, `after_ens.sh`, `switch_to_cfg.sh`, `switch_to_ens.sh`, `diag_run.sh`, `link_test.sh` | evaluation loops per seed with retry on a bad simulator start |
| `eval_status.py`, `results_summary.py`, `export_results.py`, `first_bottom.py`, `step_timing.py`, `trace_report.py`, `wait_port.py` | status, summaries, first-grasp offset against the chip pose, timing |
| `patch_*.py` | the scripts that originally inserted the diagnostics now committed in `policy/openpi/` |

The client-side diagnostics in `policy/openpi/deploy_policy.py` and `transforms.py` do nothing unless
these environment variables are set: `TACGUIDE_VALIDATE=1` (start-up check: 63 tactile markers per side,
wrist image mean below 132; optional `TACGUIDE_TACTILE_REF` reference image), `TACGUIDE_TRACE=<dir>`
(dump every query observation and action chunk), `TACGUIDE_RAW=<dir>` (save native-resolution frames).
