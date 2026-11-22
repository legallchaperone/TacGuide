"""Velocity-level building blocks shared by distillation training and guided inference.

Everything here follows the conventions of ``PI0Pytorch.forward`` in the bundled
runtime: ``x_t = t * noise + (1 - t) * actions`` and the model predicts
``u_t = noise - actions``. Nothing in this file defines a second convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import math

import torch

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

# Image names used by the two model families (see openpi.policies.sim_policy).
VISION_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TACTILE_IMAGE_KEYS = ("left_tactile_0_rgb", "right_tactile_0_rgb")
# vision-model key -> key of the same camera in a vision+tactile batch.
VISION_FROM_TACTILE_BATCH = {"base_0_rgb": "base_0_rgb", "left_wrist_0_rgb": "wrist_0_rgb"}
# SimVisionInputs feeds an all-zero uint8 image for the unused third camera. After the
# [-1, 1] conversion (and the training augmentations, which keep black black) it is -1.
MASKED_IMAGE_FILL = -1.0


def combine_velocity(v_strong: torch.Tensor, v_weak: torch.Tensor, w: float) -> torch.Tensor:
    """Return ``v_weak + w * (v_strong - v_weak)``.

    It is written as ``(1 - w) * v_weak + w * v_strong``. The two forms are the same
    algebraically, but this one is exact in floating point at the endpoints: w = 1
    returns ``v_strong`` bit for bit and w = 0 returns ``v_weak`` bit for bit, which
    is what the plan's equivalence checks ask for.
    """
    w = float(w)
    if not math.isfinite(w):
        raise ValueError(f"guidance weight must be finite, got {w}")
    return (1.0 - w) * v_weak + w * v_strong


@dataclasses.dataclass
class ModelInputs:
    """One model's view of an observation after preprocessing."""

    images: list[tuple[str, torch.Tensor]]
    image_masks: dict[str, torch.Tensor]
    lang_tokens: torch.Tensor
    lang_masks: torch.Tensor
    state: torch.Tensor

    @property
    def image_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.images)

    def to(self, device) -> "ModelInputs":
        return ModelInputs(
            images=[(k, v.to(device)) for k, v in self.images],
            image_masks={k: v.to(device) for k, v in self.image_masks.items()},
            lang_tokens=self.lang_tokens.to(device),
            lang_masks=self.lang_masks.to(device),
            state=self.state.to(device),
        )


def preprocess_once(observation, image_keys: Sequence[str], *, train: bool):
    """Run the runtime's preprocessing (and its random augmentations) exactly once.

    Teacher A, teacher B and the student must see the same augmented pixels, otherwise
    the regression target is computed for a different input than the student's.
    """
    return _preprocessing.preprocess_observation_pytorch(observation, train=train, image_keys=tuple(image_keys))


def full_view(processed, image_keys: Sequence[str]) -> ModelInputs:
    """View for a model that consumes the batch's own image names (teacher A)."""
    return ModelInputs(
        images=[(key, processed.images[key]) for key in image_keys],
        image_masks={key: processed.image_masks[key] for key in image_keys},
        lang_tokens=processed.tokenized_prompt,
        lang_masks=processed.tokenized_prompt_mask,
        state=processed.state,
    )


def vision_view(
    processed,
    image_keys: Sequence[str] = VISION_IMAGE_KEYS,
    source_keys: Mapping[str, str] | None = None,
) -> ModelInputs:
    """View for a vision-only model, built from a vision+tactile batch.

    Cameras are renamed to the vision model's names. A camera the batch does not have
    becomes a masked constant image, as in ``SimVisionInputs``. Tactile data cannot
    pass through: only keys listed in ``source_keys`` are ever read.
    """
    source_keys = dict(VISION_FROM_TACTILE_BATCH if source_keys is None else source_keys)
    images: list[tuple[str, torch.Tensor]] = []
    masks: dict[str, torch.Tensor] = {}
    reference = None
    for key in image_keys:
        source = source_keys.get(key)
        if source is not None and source in processed.images:
            reference = processed.images[source]
            break
    if reference is None:
        raise ValueError(f"none of the vision cameras {tuple(image_keys)} is present in the batch")
    for key in image_keys:
        source = source_keys.get(key)
        if source is not None and source in processed.images:
            images.append((key, processed.images[source]))
            masks[key] = processed.image_masks[source]
        else:
            images.append((key, torch.full_like(reference, MASKED_IMAGE_FILL)))
            masks[key] = torch.zeros(reference.shape[0], dtype=torch.bool, device=reference.device)
    view = ModelInputs(
        images=images,
        image_masks=masks,
        lang_tokens=processed.tokenized_prompt,
        lang_masks=processed.tokenized_prompt_mask,
        state=processed.state,
    )
    assert_no_tactile(view)
    return view


def assert_no_tactile(view: ModelInputs) -> None:
    """Fail loudly if a tactile field reached a vision-only model."""
    for key, image in view.images:
        if "tactile" in key or key in TACTILE_IMAGE_KEYS:
            raise AssertionError(f"tactile image '{key}' leaked into a vision-only view")
        if image.ndim != 4:
            raise AssertionError(f"image '{key}' has shape {tuple(image.shape)}; tactile clips are 5-D")


def _maybe_bfloat16(model, *tensors):
    weight = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight
    if weight.dtype == torch.bfloat16:
        return tuple(t.to(dtype=torch.bfloat16) for t in tensors)
    return tensors


def predict_velocity(model, inputs: ModelInputs, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """The model's velocity ``v_t`` at ``(x_t, time)``.

    This is ``PI0Pytorch.forward`` from the embedding step to ``action_out_proj``,
    with ``x_t`` and ``time`` supplied by the caller instead of sampled inside.
    ``forward`` itself only returns the squared error, so the velocity has to be
    read out here. ``tests/test_velocity.py`` pins the two against each other.
    """
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        inputs.images, inputs.image_masks, inputs.lang_tokens, inputs.lang_masks
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(inputs.state, x_t, time)
    suffix_embs, prefix_embs = _maybe_bfloat16(model, suffix_embs, prefix_embs)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)  # noqa: SLF001

    def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        (_, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out

    suffix_out = model._apply_checkpoint(  # noqa: SLF001
        forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
    )
    suffix_out = suffix_out[:, -model.config.action_horizon :].to(dtype=torch.float32)
    return model._apply_checkpoint(model.action_out_proj, suffix_out)  # noqa: SLF001


def parameter_hash(model: torch.nn.Module) -> str:
    """SHA-256 over every parameter and buffer (names, shapes, dtypes and bytes)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        data = tensor.detach().cpu().contiguous()
        if data.numel():
            digest.update(data.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def norm_stats_equal(a, b, keys: Sequence[str] = ("state", "actions")) -> tuple[bool, str]:
    """Compare two openpi norm-stats dicts on the keys that define the action space."""
    import numpy as np

    if a is None or b is None:
        return False, "missing norm stats"
    for key in keys:
        if (key in a) != (key in b):
            return False, f"'{key}' present in only one of the two"
        if key not in a:
            continue
        for field in ("mean", "std", "q01", "q99"):
            left, right = getattr(a[key], field, None), getattr(b[key], field, None)
            if (left is None) != (right is None):
                return False, f"{key}.{field} present in only one of the two"
            if left is not None and not np.array_equal(np.asarray(left), np.asarray(right)):
                diff = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
                return False, f"{key}.{field} differs (max abs diff {diff:.3g})"
    return True, ""
