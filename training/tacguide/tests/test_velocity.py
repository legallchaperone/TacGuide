import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from conftest import TACTILE_FIRST_KEYS, make_batch, vision_observation
from tacguide import velocity


def test_combine_is_exact_at_the_endpoints():
    torch.manual_seed(0)
    v_a, v_b = torch.randn(4, 50, 32), torch.randn(4, 50, 32)
    assert torch.equal(velocity.combine_velocity(v_a, v_b, 1.0), v_a)  # w = 1 -> v_A, bit for bit
    assert torch.equal(velocity.combine_velocity(v_a, v_b, 0.0), v_b)  # w = 0 -> v_B, bit for bit


@pytest.mark.parametrize("w", [1.5, 2.0, 3.0, -0.5])
def test_combine_matches_the_plan_formula(w):
    torch.manual_seed(0)
    v_a, v_b = torch.randn(4, 50, 32), torch.randn(4, 50, 32)
    assert torch.allclose(velocity.combine_velocity(v_a, v_b, w), v_b + w * (v_a - v_b), atol=1e-5)


def test_combine_rejects_non_finite_weight():
    with pytest.raises(ValueError):
        velocity.combine_velocity(torch.zeros(1), torch.zeros(1), float("nan"))


@pytest.mark.parametrize("train", [False, True])
def test_predict_velocity_follows_the_runtime_forward(strong, train):
    """`PI0Pytorch.forward` and our velocity read-out must stay the same computation."""
    observation, actions, noise, time = make_batch()
    strong.train(train)
    try:
        torch.manual_seed(7)
        reference = strong.forward(observation, actions, noise=noise, time=time)
        torch.manual_seed(7)
        processed = velocity.preprocess_once(observation, TACTILE_FIRST_KEYS, train=True)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        v_t = velocity.predict_velocity(strong, velocity.full_view(processed, TACTILE_FIRST_KEYS), x_t, time)
    finally:
        strong.eval()
    assert torch.equal(F.mse_loss(noise - actions, v_t, reduction="none"), reference)


def test_vision_view_matches_the_vision_pipeline(weak):
    """A vision view cut out of a tactile batch == what the vision-only pipeline feeds B."""
    observation, actions, noise, time = make_batch()
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    processed = velocity.preprocess_once(observation, TACTILE_FIRST_KEYS, train=False)
    view = velocity.vision_view(processed)
    assert view.image_keys == velocity.VISION_IMAGE_KEYS
    assert not view.image_masks["right_wrist_0_rgb"].any()
    with torch.no_grad():
        from_view = velocity.predict_velocity(weak, view, x_t, time)
        direct = velocity.preprocess_once(vision_observation(observation), velocity.VISION_IMAGE_KEYS, train=False)
        reference = velocity.predict_velocity(weak, velocity.full_view(direct, velocity.VISION_IMAGE_KEYS), x_t, time)
    assert torch.equal(from_view, reference)


def test_vision_view_never_contains_tactile():
    observation, *_ = make_batch()
    processed = velocity.preprocess_once(observation, TACTILE_FIRST_KEYS, train=False)
    view = velocity.vision_view(processed)
    assert not any("tactile" in key for key in view.image_keys)
    leaking = velocity.full_view(processed, TACTILE_FIRST_KEYS)
    with pytest.raises(AssertionError):
        velocity.assert_no_tactile(leaking)


def test_parameter_hash_detects_any_change():
    layer = torch.nn.Linear(4, 4)
    before = velocity.parameter_hash(layer)
    assert velocity.parameter_hash(layer) == before
    with torch.no_grad():
        layer.weight[0, 0] += 1e-6
    assert velocity.parameter_hash(layer) != before
