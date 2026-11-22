"""Plan item 2.6 checks, on tiny CPU models."""

import numpy as np
import pytest
import torch

from conftest import ACTION_DIM, HORIZON, TACTILE_FIRST_KEYS, make_batch, vision_observation
from tacguide.guided import GuidedPolicy, SeededNoisePolicy, guided_sample_actions


def test_w1_equals_strong_alone_and_w0_equals_weak_alone(strong, weak):
    observation, _, noise, _ = make_batch()
    weak_observation = vision_observation(observation)
    only_strong = strong.sample_actions("cpu", observation, noise=noise, num_steps=4)
    only_weak = weak.sample_actions("cpu", weak_observation, noise=noise, num_steps=4)
    assert not torch.allclose(only_strong, only_weak)

    def guided(w):
        return guided_sample_actions(strong, weak, "cpu", observation, weak_observation, w=w, noise=noise, num_steps=4)

    assert torch.equal(guided(1.0), only_strong)  # element for element
    assert torch.equal(guided(0.0), only_weak)
    extrapolated = guided(2.0)
    assert not torch.allclose(extrapolated, only_strong) and not torch.allclose(extrapolated, only_weak)


def _raw_observation(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/left_tactile_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/right_tactile_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(ACTION_DIM).astype(np.float32),
        "tokens": rng.integers(0, 1000, 12),
    }


def _common(data):
    return {
        "state": data["observation/state"],
        "tokenized_prompt": np.asarray(data["tokens"]),
        "tokenized_prompt_mask": np.ones(12, dtype=bool),
    }


def _strong_inputs(data):
    names = TACTILE_FIRST_KEYS
    images = (
        data["observation/left_tactile_image"],
        data["observation/right_tactile_image"],
        data["observation/image"],
        data["observation/wrist_image"],
    )
    return {
        "image": dict(zip(names, images, strict=True)),
        "image_mask": {name: np.True_ for name in names},
        **_common(data),
    }


def _weak_inputs(data):
    base = data["observation/image"]
    return {
        "image": {
            "base_0_rgb": base,
            "left_wrist_0_rgb": data["observation/wrist_image"],
            "right_wrist_0_rgb": np.zeros_like(base),
        },
        "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        **_common(data),
    }


def _policy(model, inputs):
    policy_module = pytest.importorskip("openpi.policies.policy")
    return policy_module.Policy(
        model,
        transforms=[inputs],
        output_transforms=[lambda data: {"actions": np.asarray(data["actions"][..., :8])}],
        sample_kwargs={"num_steps": 3},
        is_pytorch=True,
        pytorch_device="cpu",
    )


def test_guided_policy_matches_single_policies(strong, weak):
    """The served path: two real openpi Policy objects, raw client observation in, actions out."""
    obs = _raw_observation()
    noise = np.random.default_rng(1).standard_normal((HORIZON, ACTION_DIM)).astype(np.float32)
    only_strong = _policy(strong, _strong_inputs).infer(obs, noise=noise)["actions"]
    only_weak = _policy(weak, _weak_inputs).infer(obs, noise=noise)["actions"]
    assert only_strong.shape == (HORIZON, 8)

    def guided(w):
        policy = GuidedPolicy(_policy(strong, _strong_inputs), _policy(weak, _weak_inputs), w=w)
        return policy.create_session().infer(obs, noise=noise)

    at_one = guided(1.0)
    assert np.array_equal(at_one["actions"], only_strong)
    assert at_one["guidance"] == {"w": 1.0}
    assert np.array_equal(guided(0.0)["actions"], only_weak)
    assert not np.allclose(guided(2.0)["actions"], only_strong)


def test_seeded_noise_makes_runs_repeatable(strong, weak):
    obs = _raw_observation()
    alone = SeededNoisePolicy(_policy(strong, _strong_inputs), seed=5).create_session()
    guided = SeededNoisePolicy(
        GuidedPolicy(_policy(strong, _strong_inputs), _policy(weak, _weak_inputs), w=1.0), seed=5
    ).create_session()
    first, second = alone.infer(obs)["actions"], alone.infer(obs)["actions"]
    assert not np.array_equal(first, second)  # a new noise sample for every request
    assert np.array_equal(guided.infer(obs)["actions"], first)  # same seed, same step -> same actions
    assert np.array_equal(guided.infer(obs)["actions"], second)
    assert np.array_equal(alone.infer({**obs, "reset": True})["actions"], first)  # reset restarts the stream
