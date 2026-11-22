# TacGuide: distillation, extrapolation and guided inference

Additions on top of the bundled OpenVTLA training code (`training/run.py`,
`training/openpi_runtime`). Nothing here edits those files.

| Experiment | What it is | How it is trained |
| --- | --- | --- |
| `gc_A` | teacher, vision + tactile | official: `run.py train --task gc --model openvtla` |
| `gc_B` | plain student, vision only | official: `run.py train --task gc --model pi05` |
| `gc_C` | distilled student, w = 1 | `tacguide/train_distill.py gc_C` |
| `gc_D_w<w>` | extrapolated student | `tacguide/train_distill.py gc_D_w2` |

On the training machine run `source training/tacguide_env.sh` first and use
`$TG/venv/bin/python` from the repository root.

## Training C and D

```bash
$TG/venv/bin/python training/tacguide/train_distill.py gc_C            # needs gc_A only
$TG/venv/bin/python training/tacguide/train_distill.py gc_D_w2         # needs gc_A and gc_B
$TG/venv/bin/python training/tacguide/train_distill.py gc_D_w2 --resume
```

Each step draws noise and a time step once, builds `x_t = t·noise + (1 − t)·actions`
as the runtime does, evaluates the frozen teachers and regresses

    target = v_B + w · (v_A − v_B)

The student is initialised and optimised exactly like `gc_B` (its config is derived
from `gc_B` in `configs.py`), and its checkpoints are ordinary vision-only
checkpoints, served with the `gc_B` config. Details that matter:

- The three models share one preprocessing call, so they see the same random crop,
  rotation and colour jitter.
- The batch comes from the vision + tactile pipeline. The student and B get the two
  cameras picked out of it under the vision model's names; tactile never reaches them.
- A, B and the student must share state/action normalisation statistics. This is
  verified at start-up and the run aborts if they differ.
- Teachers are loaded as in serving (bfloat16 backbone, eval mode), are not part of
  the student module, and their SHA-256 parameter hashes are checked at every
  checkpoint save and at the end.
- The target is computed as `(1 − w)·v_B + w·v_A`, the same value written so that
  w = 1 and w = 0 are exact in floating point.
- The runtime's PyTorch trainer has no EMA, so there is no EMA to switch off. If
  memory is short, `--teacher-device cuda:1` moves the teachers to a second card.

Plan checks:

```bash
# teacher hashes unchanged over 10 steps
... train_distill.py gc_D_w2 --exp-name hashcheck --steps 10 --no-save --no-wandb
# memory: batch 64, 100 steps, base weights standing in for A and B
... train_distill.py gc_D_w2 --exp-name memtest --steps 100 --no-save --no-wandb --base-standin --summary memtest.json
```

## Guided inference

```bash
# CFG(A, B): client sends tactile clips (vision + tactile deploy config)
$TG/venv/bin/python training/tacguide/serve_guided.py --strong gc_A --weak gc_B --w 2 --port 8000
# zero-training extrapolation: no tactile (vision deploy config)
$TG/venv/bin/python training/tacguide/serve_guided.py --strong gc_C --weak gc_B --w 2 --port 8000
```

Every denoising step uses `v = v_weak + w·(v_strong − v_weak)` with one shared
initial noise and one shared `x_t`. With `--seed` (default 0) request k of an episode
always receives the same noise; start a single model through the same script
(`--strong gc_A` alone) to compare it with a guided run episode by episode.

## Scheduler

`gpu_queue.py` (standard library only) starts the best ready job from
`queue.gc.json` whenever a card is free. See the docstring for commands.

## Tests

```bash
$TG/venv/bin/python -m pytest training/tacguide/tests -q
```

They build real `PI0Pytorch` models shrunk to CPU size and check, among others:
target equals `v_A` at w = 1 and `v_B` at w = 0; teachers unchanged after 10
optimiser steps; no tactile in the student's input; guided sampling equal to the
strong model at w = 1 and to the weak model at w = 0, element for element.
