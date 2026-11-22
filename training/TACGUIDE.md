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
6. Environment variables live in `training/tacguide_env.sh`, not `~/.bashrc`, because the
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
