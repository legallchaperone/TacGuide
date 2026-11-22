"""Checks that need the runtime environment: the sim client wiring and loader/client clip parity.

  uv run --project training/openpi_runtime python -m pytest training/tests/test_client_and_loader.py

The loader parity test needs a converted dataset; set TG_PARITY_TASK (default: pd) and it
is skipped when that dataset does not exist.
"""

import importlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
import config as tg_config  # noqa: E402

torch = pytest.importorskip("torch")


def load_deploy_policy():
    """Import policy/openpi/deploy_policy.py as a package module without Isaac Sim."""
    try:
        import transforms3d  # noqa: F401  (only used by the end-effector modes, absent from the training env)
    except ImportError:
        sys.modules["transforms3d"] = types.ModuleType("transforms3d")
    if "tg_policy" not in sys.modules:
        pkg = types.ModuleType("tg_policy")
        pkg.__path__ = [str(ROOT / "policy")]
        sys.modules["tg_policy"] = pkg
    return importlib.import_module("tg_policy.openpi.deploy_policy")


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.observations = []

    def connect(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    def infer(self, obs):
        self.observations.append(obs)
        return np.zeros((50, 8), dtype=np.float64)


class FakeTask:
    instruction = "grasp the chip"
    device = "cpu"
    _robot_manager = types.SimpleNamespace(gripper_max_qpos=0.039)

    def take_action(self, action, action_type="qpos", **kwargs):
        return True, False


def sim_observation(step):
    def image(value):
        return torch.full((32, 40, 3), value, dtype=torch.uint8)

    return {
        "embodiment": {"joint": torch.zeros(9)},
        "observation": {"head": {"rgb": image(10)}, "wrist": {"rgb": image(20)}},
        "tactile": {
            "left_tactile": {"rgb_marker": image(step)},
            "right_tactile": {"rgb_marker": image(100 + step)},
        },
    }


def make_policy(monkeypatch, *, tactile):
    module = load_deploy_policy()
    monkeypatch.setattr(module, "OpenPiClientRuntime", FakeClient)
    openpi_cfg = {
        "host": "127.0.0.1", "port": 8000, "control_mode": "abs_joint", "action_dim": 8, "state_dim": 8,
        "image_size": 224, "send_tactile": tactile, "image_color_order": "rgb", "open_loop_horizon": 20,
    }
    if tactile:
        openpi_cfg["tactile_history_offsets"] = [-6, -4, -2, 0]
    return module.Policy({"policy_name": "openpi", "openpi": openpi_cfg})


def clip_ids(clip):
    return [int(frame[0, 0, 0]) for frame in clip]


def test_client_sends_clips_every_query_and_resets_between_episodes(monkeypatch):
    policy = make_policy(monkeypatch, tactile=True)
    task = FakeTask()
    for episode in range(2):
        policy.reset()
        for step in range(45):
            policy.eval(task, sim_observation(step))
    sent = policy.client.observations
    assert len(sent) == 6  # steps 0, 20 and 40 of both episodes
    for query, obs in enumerate(sent):
        step = (query % 3) * 20
        expected = [max(step + offset, 0) for offset in (-6, -4, -2, 0)]
        assert clip_ids(obs["observation/left_tactile_clip"]) == expected
        assert clip_ids(obs["observation/right_tactile_clip"]) == [100 + e for e in expected]
        assert obs["observation/left_tactile_clip"].shape == (4, 224, 224, 3)
        assert "observation/left_tactile_image" not in obs
        assert ("reset" in obs) == (step == 0)
        assert obs["prompt"] == "grasp the chip"


def test_vision_only_client_sends_no_tactile(monkeypatch):
    policy = make_policy(monkeypatch, tactile=False)
    policy.reset()
    policy.eval(FakeTask(), sim_observation(0))
    obs = policy.client.observations[0]
    assert not [key for key in obs if "tactile" in key]
    assert obs["reset"] is True


def test_client_clip_equals_training_loader_clip():
    task = os.environ.get("TG_PARITY_TASK", "pd")
    if not (tg_config.DATA / "lerobot" / task).is_dir():
        pytest.skip(f"no converted dataset for task {task}")
    os.environ["HF_LEROBOT_HOME"] = str(tg_config.DATA / "lerobot")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    import run as tg_run
    from openpi.training import data_loader

    spec = importlib.util.spec_from_file_location("tactile_history", ROOT / "policy/openpi/tactile_history.py")
    tactile_history = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tactile_history)

    args = types.SimpleNamespace(task=task, model="openvtla", exp_name="parity", num_workers=0, resume=False)
    train_config = tg_run.build_config(args)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    fps = LeRobotDatasetMetadata(task).fps
    offsets = data_config.observation_history_offsets["left_tactile_image"]
    # Exactly the history request the training loader makes (data_loader.create_torch_dataset).
    loader_view = LeRobotDataset(task, delta_timestamps={
        key: [o / fps for o in data_config.observation_history_offsets[key]]
        for key in ("left_tactile_image", "right_tactile_image")
    })
    assert "observation_history_offsets" in Path(data_loader.__file__).read_text()
    frames = LeRobotDataset(task)

    def uint8(image):
        image = np.asarray(image)
        if image.shape[-3] == 3:
            image = np.moveaxis(image, -3, -1)
        return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)

    last_episode = frames.meta.total_episodes - 1
    for episode in sorted({0, last_episode}):
        begin = int(frames.episode_data_index["from"][episode])
        end = int(frames.episode_data_index["to"][episode])
        history = tactile_history.TactileHistory(offsets)
        for step, index in enumerate(range(begin, min(begin + 12, end))):
            item = frames[index]
            out = history.push({
                tactile_history.LEFT_IMAGE_KEY: uint8(item["left_tactile_image"]),
                tactile_history.RIGHT_IMAGE_KEY: uint8(item["right_tactile_image"]),
            })
            expected = loader_view[index]
            np.testing.assert_array_equal(out[tactile_history.LEFT_CLIP_KEY], uint8(expected["left_tactile_image"]),
                                          err_msg=f"episode {episode} step {step} left")
            np.testing.assert_array_equal(out[tactile_history.RIGHT_CLIP_KEY], uint8(expected["right_tactile_image"]),
                                          err_msg=f"episode {episode} step {step} right")
