"""Guided inference (plan item 2.6).

Every denoising step uses ``v = v_weak + w * (v_strong - v_weak)``. Both models
start from the same noise and are evaluated at the same ``x_t``.

  CFG(A, B):                strong = gc_A, weak = gc_B (the client sends tactile)
  zero-training extrapolation: strong = gc_C, weak = gc_B (no tactile anywhere)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

from . import velocity as _v

logger = logging.getLogger("tacguide")


def _prefix_cache(model, observation):
    """First half of ``PI0Pytorch.sample_actions``: encode images + prompt once."""
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return state, prefix_pad_masks, past_key_values


@torch.no_grad()
def guided_sample_actions(
    strong,
    weak,
    device,
    obs_strong,
    obs_weak,
    *,
    w: float,
    noise: torch.Tensor | None = None,
    num_steps: int = 10,
) -> torch.Tensor:
    """``PI0Pytorch.sample_actions`` with the velocity replaced by the guided one.

    The Euler loop, step count and time grid are copied from the runtime so that
    w = 1 reproduces ``strong.sample_actions`` and w = 0 reproduces
    ``weak.sample_actions`` element for element, given the same noise.
    """
    bsize = obs_strong.state.shape[0]
    if noise is None:
        noise = strong.sample_noise((bsize, strong.config.action_horizon, strong.config.action_dim), device)
    if (strong.config.action_horizon, strong.config.action_dim) != (weak.config.action_horizon, weak.config.action_dim):
        raise ValueError("strong and weak models must share the action horizon and dimension")

    cache_strong = _prefix_cache(strong, obs_strong)
    cache_weak = _prefix_cache(weak, obs_weak)

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_strong = strong.denoise_step(*cache_strong, x_t, expanded_time)
        v_weak = weak.denoise_step(*cache_weak, x_t, expanded_time)
        v_t = _v.combine_velocity(v_strong, v_weak, w)
        x_t = x_t + dt * v_t
        time += dt
    return x_t


def unwrap_policy(policy):
    """Follow ``_policy`` links down to the ``openpi.policies.policy.Policy`` that owns the model."""
    seen = 0
    while not hasattr(policy, "_sample_actions"):
        policy = getattr(policy, "_policy", None) or getattr(policy, "_strong", None)
        seen += 1
        if policy is None or seen > 8:
            raise TypeError("could not find the underlying openpi Policy (no `_sample_actions`)")
    return policy


class _GuidedSampler:
    """Replaces ``Policy._sample_actions`` of the strong policy."""

    def __init__(self, strong_model, weak_model, w: float):
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.w = float(w)
        self.pending_weak_observation = None

    def __call__(self, device, observation, **kwargs):
        if self.pending_weak_observation is None:
            raise RuntimeError("guided sampling was called without a weak-model observation")
        return guided_sample_actions(
            self.strong_model,
            self.weak_model,
            device,
            observation,
            self.pending_weak_observation,
            w=self.w,
            **kwargs,
        )


class GuidedPolicy:
    """Serve two trained policies as one guided policy.

    The strong policy keeps its whole serving stack (tactile-history wrappers,
    input checks, output transforms, reset handling). Only its sampling function is
    swapped for the guided one. The weak policy contributes its input transforms and
    its model. Both policies must normalise state and actions identically, since the
    two velocities are mixed in normalised action space; pass the two norm-stats
    dicts to have this verified.
    """

    def __init__(self, strong, weak, *, w: float, strong_norm_stats=None, weak_norm_stats=None, _sampler=None):
        self._strong = strong
        self._weak = weak
        self._strong_core = unwrap_policy(strong)
        self._weak_core = unwrap_policy(weak)
        if not getattr(self._strong_core, "_is_pytorch_model", False) or not getattr(
            self._weak_core, "_is_pytorch_model", False
        ):
            raise TypeError("guided inference is implemented for PyTorch checkpoints only")
        if strong_norm_stats is not None or weak_norm_stats is not None:
            same, why = _v.norm_stats_equal(strong_norm_stats, weak_norm_stats)
            if not same:
                raise ValueError(f"strong and weak policies use different normalisation: {why}")
        if _sampler is None:
            _sampler = _GuidedSampler(self._strong_core._model, self._weak_core._model, w)  # noqa: SLF001
            self._strong_core._sample_actions = _sampler  # noqa: SLF001
        self._sampler = _sampler

    @property
    def w(self) -> float:
        return self._sampler.w

    def _weak_observation(self, obs: dict):
        from openpi.models import model as _model

        core = self._weak_core
        inputs = {k: copy.copy(v) for k, v in obs.items() if k not in ("reset", "episode_reset")}
        inputs = core._input_transform(inputs)  # noqa: SLF001
        device = core._pytorch_device  # noqa: SLF001
        inputs = _tree_map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        self._sampler.pending_weak_observation = self._weak_observation(obs)
        try:
            outputs = self._strong.infer(obs, noise=noise) if noise is not None else self._strong.infer(obs)
        finally:
            self._sampler.pending_weak_observation = None
        outputs["guidance"] = {"w": self.w}
        return outputs

    def reset(self) -> None:
        for policy in (self._strong, self._weak):
            reset = getattr(policy, "reset", None)
            if reset is not None:
                reset()

    def create_session(self) -> "GuidedPolicy":
        strong = self._strong.create_session() if hasattr(self._strong, "create_session") else self._strong
        return GuidedPolicy(strong, self._weak, w=self.w, _sampler=self._sampler)

    @property
    def metadata(self) -> dict[str, Any]:
        return {**(getattr(self._strong, "metadata", None) or {}), "guidance_w": self.w}


class SeededNoisePolicy:
    """Give any policy a reproducible noise stream.

    Call k of an episode uses noise drawn from ``default_rng([seed, k])``. The counter
    restarts on a new connection and whenever the client sends a truthy ``reset`` or
    ``episode_reset``. Wrapping the plain server and the guided server with the same
    seed makes "guided, w = 1" and "strong model alone" comparable step for step.
    """

    def __init__(self, policy, *, seed: int, noise_shape: tuple[int, int] | None = None):
        self._policy = policy
        self._seed = int(seed)
        if noise_shape is None:
            config = unwrap_policy(policy)._model.config  # noqa: SLF001
            noise_shape = (config.action_horizon, config.action_dim)
        self._noise_shape = tuple(noise_shape)
        self._step = 0

    def infer(self, obs: dict) -> dict:
        if obs.get("reset") or obs.get("episode_reset"):
            self._step = 0
        rng = np.random.default_rng([self._seed, self._step])
        noise = rng.standard_normal(self._noise_shape).astype(np.float32)
        self._step += 1
        return self._policy.infer(obs, noise=noise)

    def reset(self) -> None:
        self._step = 0
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()

    def create_session(self) -> "SeededNoisePolicy":
        inner = self._policy.create_session() if hasattr(self._policy, "create_session") else self._policy
        return SeededNoisePolicy(inner, seed=self._seed, noise_shape=self._noise_shape)

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", None) or {}


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: _tree_map(fn, v) for k, v in tree.items()}
    return fn(tree)
