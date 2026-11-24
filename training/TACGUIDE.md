# TacGuide: training A (vision + touch) and B (vision only) on Grasp Chip

This note covers plan items 2.1 to 2.4. Everything builds on the released simulation
training code in this directory (see [README.md](README.md)); the additions are listed
at the end.

## Machine layout (training machine)

- Code: `~/TacGuide`, branch `tacguide`.
- Everything large: `/data2/cwx/TacGuide` (`data/`, `weights/`, `outputs/`, `venv/`,
  `cache/`, `logs/`). The repository's `data`, `weights` and `outputs` are symlinks into it.
- Before any command: `source training/tacguide_env.sh`. It points every cache at the data
  disk and unsets the proxy from `~/.bashrc`, which is dead. GitHub, PyPI and Google storage
  are reachable directly; huggingface.co is not.
- Weights: `weights/pi05_base_pytorch/` was converted from
  `gs://openpi-assets/checkpoints/pi05_base` with upstream openpi's
  `examples/convert_jax_model_to_pytorch.py` (`--config-name pi05_base`).
  `weights/anytouch2/checkpoint-4frames.pth` is from the gated `xxuan01/AnyTouch2-Model`.

## Commands

```bash
training/tacguide.sh prep  <dir with .hdf5 episodes>   # link data, convert, stats, conversion check
training/tacguide.sh smoke gc_A|gc_B <gpu>             # 10 steps, batch 64
training/tacguide.sh train gc_A|gc_B <gpu> [--resume]  # 10k steps, checkpoint every 1k
training/tacguide.sh serve gc_A|gc_B <gpu> [port] [step]
```

`gc_A` is OpenVTLA (`--model openvtla`), `gc_B` is vision-only π0.5 (`--model pi05`).
Checkpoints land in `outputs/training/<model>/gc/<gc_A|gc_B>/<step>/`, 20 GB each.
`TG_STEPS`, `TG_SAVE_INTERVAL` and `TG_WANDB=1` change steps, checkpoint interval and
wandb logging. To train on the public dataset instead of collected HDF5, run
`training/convert_official_lerobot.py --source <dataset> --task gc`, then the two `stats`
commands from the README.

## Finished runs and how to resume a run

| Run | Final checkpoint (UTC) | Wall-clock training time | Last 100 steps |
| --- | --- | --- | --- |
| `gc_A` | 2026-11-24 01:53, `outputs/training/openvtla/gc/gc_A/10000` | about 23 h | mean loss 0.0007 |
| `gc_B` | 2026-11-24 03:48, `outputs/training/pi05/gc/gc_B/10000` | about 25 h | mean loss 0.0007 |

Both started on 2026-11-22 (20:29 and 21:03 UTC, cards 7 and 5) at 6.6 and 7.6 s/step. On
2026-11-23 every process of the account was terminated between 13:35 and 16:00 UTC while the
shared `/data2` disk was overloaded (single checkpoint saves were taking over an hour). The
runs were resumed at 19:35 UTC from steps 7000 and 6000 on cards 1 and 5, shared with other
users' jobs, at about 7.3 s/step, so each run repeated about 1000 steps. Both final
checkpoints carry identical normalisation statistics.

An active run sits in tmux `train_gc_A` / `train_gc_B` and logs to `$TG/logs/train_gc_*.log`.
`training/watch_ab.sh gc_A:7 gc_B:5` (tmux `watch_ab`) writes `$TG/logs/watch_ab.status`
every five minutes and resumes a run that died, on the same card, when that card has room.
To resume by hand, possibly on another card:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True training/tacguide.sh train gc_B <gpu> --resume
```

The cards are shared with another user's 40 GB process, which leaves 2 to 3 GB of headroom
for `gc_B`; `expandable_segments` reduces fragmentation. Only one resumer may act on a run:
stop `watch_ab` before letting another scheduler manage these two runs.

## Policy interface

| Field | Shape | Content |
| --- | --- | --- |
| `observation/state` | `[8]` float32 | 7 arm joints, left finger position |
| `observation/image`, `observation/wrist_image` | `[224, 224, 3]` uint8 RGB | head and wrist camera |
| `observation/left_tactile_clip`, `observation/right_tactile_clip` | `[4, 224, 224, 3]` uint8 | `rgb_marker` at steps t-6, t-4, t-2, t; OpenVTLA only |
| `prompt` | string | the task instruction |
| `reset` | bool, optional | present in the first observation of an episode |
| returned `actions` | `[50, 8]` | absolute joint targets and gripper target |

The model predicts arm joints relative to the current state; the server converts them back
to absolute targets. The sim client executes the first 20 actions of each chunk.

## Checks

| Item | How | Command |
| --- | --- | --- |
| 2.2 conversion | fields, state vs HDF5, action[t] = state[t+1], images vs HDF5, preview videos | `training/checks/check_conversion.py` |
| 2.3 server | chunk shape and finiteness through the websocket | `training/checks/check_server.py` |
| 2.4 touch is used | same input and noise, real vs zeroed tactile clips | `training/checks/check_tactile.py` |
| client clips | client history equals the training loader's clip; client sends clips and `reset` | `pytest training/tests` |

`training/checks/make_fake_episodes.py` writes small synthetic episodes for pipeline tests.

## Where this differs from the written plan

1. Code base: the bundled PyTorch runtime in `training/openpi_runtime` (released with
   OpenViTac on 2026-11-17), not upstream JAX openpi. Tactile integration, converter and
   configs already existed there.
2. AnyTouch2 is trained jointly with the model, and training runs 10k steps, as in the
   released code (Simon, 2026-11-22). The plan's check "AnyTouch2 weights unchanged after 10
   steps" therefore no longer applies; `check_tactile.py` checks that they do change.
   `--freeze-anytouch` remains available and freezes the encoder only.
3. Tactile history is assembled in the sim client, not cached on the server. The client
   asks the server only every 20 steps, so the server never sees the frames in between.
4. The plan's `observation/left_tactile_image` single frames are replaced by the clip
   fields above for OpenVTLA.
5. wandb is off for now; losses are in the training log under `/data2/cwx/TacGuide/logs`.
6. Training data is the public OpenViTac grasp_chip dataset (100 episodes, 43,287 frames),
   not 100 self-collected demonstrations (Simon, 2026-11-22), so collection (stage 1) is
   skipped. Its images come from h264 videos at the source resolution, resized to 224, not
   from the HDF5 JPEG frames. State, action and image content were checked against the
   source with `checks/check_official_conversion.py`.
7. For evaluation, single models are served through the guided-inference server of the
   distillation branch (strong model only) so that all configurations share its seeded
   noise; `serve.py` here is the plain server used for link tests.
8. Environment variables live in `training/tacguide_env.sh`, not `~/.bashrc`, because the
   account is shared with other projects.

## Code added on top of the release

- `run.py`: `--steps`, `--save-interval`, `--batch-size`, `--log-interval`, `--wandb`,
  `--freeze-anytouch`, `--overwrite`. Unset flags keep the released defaults.
- `anytouch2/encoder.py`: `trainable=False` used to freeze the new projector as well; it
  now freezes only the pretrained encoder.
- `serve.py`: single-model policy server; `load_policy()` is reused by guided inference.
- `policy/openpi/tactile_history.py` and `deploy_policy.py`: tactile clips and the `reset`
  flag; enabled by `tactile_history_offsets` in `deploy_grasp_chip_vision_tactile.yml`.
- `convert_official_lerobot.py`, `tacguide.sh`, `tacguide_env.sh`, `checks/`, two test files.
