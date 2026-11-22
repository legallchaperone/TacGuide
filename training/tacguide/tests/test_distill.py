"""Plan item 2.5 checks, on tiny CPU models."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from conftest import TACTILE_FIRST_KEYS, VISION_KEYS, build_tiny, make_batch
from tacguide import velocity
from tacguide.distill import DistillStudent


@pytest.fixture(scope="module")
def student(strong, weak):
    model = build_tiny(VISION_KEYS, cls=DistillStudent, seed=3)
    model.attach_teachers(strong=strong, weak=weak, w=2.0)
    return model


def manual_velocities(strong, weak, student, observation, actions, noise, time, *, seed):
    torch.manual_seed(seed)
    processed = velocity.preprocess_once(observation, TACTILE_FIRST_KEYS, train=student.training)
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    vision = velocity.vision_view(processed)
    with torch.no_grad():
        v_a = velocity.predict_velocity(strong, velocity.full_view(processed, TACTILE_FIRST_KEYS), x_t, time)
        v_b = velocity.predict_velocity(weak, vision, x_t, time)
        v_s = velocity.predict_velocity(student, vision, x_t, time)
    return v_a, v_b, v_s


@pytest.mark.parametrize("train", [False, True])
def test_target_is_v_a_at_w1_and_v_b_at_w0(strong, weak, student, train):
    observation, actions, noise, time = make_batch()
    student.train(train)
    try:
        v_a, v_b, v_s = manual_velocities(strong, weak, student, observation, actions, noise, time, seed=11)
        for w, target in ((1.0, v_a), (0.0, v_b)):
            student.attach_teachers(strong=strong, weak=weak, w=w, batch_image_keys=TACTILE_FIRST_KEYS)
            torch.manual_seed(11)
            with torch.no_grad():
                loss = student(observation, actions, noise=noise, time=time)
            assert torch.equal(loss, F.mse_loss(target, v_s, reduction="none")), f"w = {w}"
        student.attach_teachers(strong=strong, weak=weak, w=2.0)
        torch.manual_seed(11)
        with torch.no_grad():
            loss = student(observation, actions, noise=noise, time=time)
        assert torch.allclose(loss, F.mse_loss(v_b + 2.0 * (v_a - v_b), v_s, reduction="none"), atol=1e-5)
    finally:
        student.eval()
        student.attach_teachers(strong=strong, weak=weak, w=2.0)


def test_w1_skips_teacher_b_and_w0_skips_teacher_a(strong, weak, student):
    try:
        student.attach_teachers(strong=strong, weak=None, w=1.0)
        assert set(student.teachers()) == {"strong"}
        student.attach_teachers(strong=None, weak=weak, w=0.0)
        assert set(student.teachers()) == {"weak"}
        with pytest.raises(ValueError):
            student.attach_teachers(strong=strong, weak=None, w=2.0)
    finally:
        student.attach_teachers(strong=strong, weak=weak, w=2.0)


def test_student_equal_to_b_has_zero_loss_at_w0(strong, weak):
    """End to end: shared noise, shared time, shared augmentation, correct camera mapping.

    Runs in train mode, where every preprocessing call draws fresh random crops and
    colour jitter. The loss is only zero if the student and B saw identical pixels.
    """
    clone = build_tiny(VISION_KEYS, cls=DistillStudent, seed=4)
    clone.load_state_dict(weak.state_dict())
    clone.attach_teachers(strong=strong, weak=weak, w=0.0)
    clone.train()
    observation, actions, _, _ = make_batch()
    with torch.no_grad():
        loss = clone(observation, actions)
    assert float(loss.max()) == 0.0


def test_teachers_are_not_part_of_the_student(strong, weak, student):
    plain = build_tiny(VISION_KEYS, seed=5)
    assert set(student.state_dict()) == set(plain.state_dict())
    student_parameters = {id(p) for p in student.parameters()}
    for teacher in (strong, weak):
        assert not any(id(p) in student_parameters for p in teacher.parameters())
        assert not teacher.training
        assert not any(p.requires_grad for p in teacher.parameters())


def test_teachers_unchanged_and_student_updated_after_10_steps(strong, weak):
    learner = build_tiny(VISION_KEYS, cls=DistillStudent, seed=6)
    learner.attach_teachers(strong=strong, weak=weak, w=2.0)
    learner.train()
    learner.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-3)
    teacher_hashes = learner.teacher_hashes()
    assert set(teacher_hashes) == {"strong", "weak"}
    student_hash = velocity.parameter_hash(learner)
    losses = []
    for step in range(10):
        observation, actions, _, _ = make_batch(seed=100 + step)
        loss = learner(observation, actions).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert learner.teacher_hashes() == teacher_hashes
    assert velocity.parameter_hash(learner) != student_hash
    assert all(p.grad is None for teacher in (strong, weak) for p in teacher.parameters())


def test_student_input_has_no_tactile(strong, weak, student):
    """Tactile pixels can only reach the loss through teacher A."""
    observation, actions, noise, time = make_batch()
    altered, *_ = make_batch()
    for key in ("left_tactile_0_rgb", "right_tactile_0_rgb"):
        altered.images[key] = torch.zeros_like(altered.images[key])
    try:
        # w = 0: only B and the student run. Changing tactile must change nothing.
        student.attach_teachers(strong=strong, weak=weak, w=0.0)
        with torch.no_grad():
            assert torch.equal(
                student(observation, actions, noise=noise, time=time),
                student(altered, actions, noise=noise, time=time),
            )
        # w = 1: the target is v_A, which does read tactile.
        student.attach_teachers(strong=strong, weak=weak, w=1.0)
        with torch.no_grad():
            assert not torch.equal(
                student(observation, actions, noise=noise, time=time),
                student(altered, actions, noise=noise, time=time),
            )
    finally:
        student.attach_teachers(strong=strong, weak=weak, w=2.0)
