"""Distillation / extrapolation loss (plan item 2.5).

The student is a plain vision-only ``PI0Pytorch``. Each step it regresses

    target = v_B + w * (v_A - v_B)

where A (vision + tactile) and B (vision only) are frozen and evaluated on the same
noise, the same time step and the same augmented images as the student.
w = 1 is experiment C (pure distillation), w > 1 is experiment D (extrapolation).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from . import velocity as _v

logger = logging.getLogger("tacguide")


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class DistillStudent(PI0Pytorch):
    """Vision-only student whose ``forward`` returns the distillation loss.

    The teachers are deliberately *not* registered as submodules: they stay out of
    ``parameters()`` (so the optimizer cannot touch them) and out of ``state_dict()``
    (so a student checkpoint is an ordinary vision-only checkpoint that the plain
    serving path loads unchanged).
    """

    def attach_teachers(
        self,
        *,
        strong: PI0Pytorch | None,
        weak: PI0Pytorch | None,
        w: float,
        batch_image_keys: tuple[str, ...] | None = None,
    ) -> "DistillStudent":
        w = float(w)
        if w != 0.0 and strong is None:
            raise ValueError("a strong teacher (A) is required unless w == 0")
        if w != 1.0 and weak is None:
            raise ValueError("a weak teacher (B) is required unless w == 1")
        # w == 1 never reads v_B and w == 0 never reads v_A, so that forward is skipped.
        strong = strong if w != 0.0 else None
        weak = weak if w != 1.0 else None
        for teacher in (strong, weak):
            if teacher is not None:
                freeze(teacher)
        if weak is not None and tuple(weak.config.image_keys) != tuple(self.config.image_keys):
            raise ValueError("teacher B and the student must take the same image inputs")
        if batch_image_keys is None:
            batch_image_keys = tuple(strong.config.image_keys) if strong is not None else None
        # Plain dict in __dict__: nn.Module.__setattr__ would register the teachers.
        self.__dict__["_distill"] = {
            "strong": strong,
            "weak": weak,
            "w": w,
            "batch_image_keys": batch_image_keys,
        }
        return self

    @property
    def guidance_weight(self) -> float:
        return self.__dict__["_distill"]["w"]

    def teachers(self) -> dict[str, PI0Pytorch]:
        state = self.__dict__.get("_distill", {})
        return {name: state[name] for name in ("strong", "weak") if state.get(name) is not None}

    def teacher_hashes(self) -> dict[str, str]:
        return {name: _v.parameter_hash(teacher) for name, teacher in self.teachers().items()}

    @staticmethod
    def _teacher_velocity(teacher, inputs, x_t, time):
        device = next(teacher.parameters()).device
        with torch.no_grad():
            velocity = _v.predict_velocity(teacher, inputs.to(device), x_t.to(device), time.to(device))
        return velocity.to(x_t.device).detach()

    def forward(self, observation, actions, noise=None, time=None):
        state = self.__dict__.get("_distill")
        if state is None:
            raise RuntimeError("call attach_teachers() before training the student")
        strong, weak, w = state["strong"], state["weak"], state["w"]

        # One preprocessing call, hence one set of random augmentations, for all models.
        batch_keys = state["batch_image_keys"]
        if batch_keys is None:
            batch_keys = tuple(k for k in observation.images if k not in _v.TACTILE_IMAGE_KEYS)
        processed = _v.preprocess_once(observation, batch_keys, train=self.training)

        if set(self.config.image_keys) <= set(batch_keys) and not set(batch_keys) & set(_v.TACTILE_IMAGE_KEYS):
            student_inputs = _v.full_view(processed, self.config.image_keys)
            _v.assert_no_tactile(student_inputs)
        else:
            student_inputs = _v.vision_view(processed, self.config.image_keys)

        # Noise and time are drawn once per sample and shared by all three models.
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        v_strong = v_weak = None
        if strong is not None:
            v_strong = self._teacher_velocity(strong, _v.full_view(processed, strong.config.image_keys), x_t, time)
        if weak is not None:
            v_weak = self._teacher_velocity(weak, student_inputs, x_t, time)
        if v_strong is None:
            target = v_weak
        elif v_weak is None:
            target = v_strong
        else:
            target = _v.combine_velocity(v_strong, v_weak, w)

        v_student = _v.predict_velocity(self, student_inputs, x_t, time)
        return F.mse_loss(target, v_student, reduction="none")
