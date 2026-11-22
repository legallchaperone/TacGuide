"""CPU fixtures: tiny pi0.5 models built from the runtime's own classes."""

import pathlib
import sys

import pytest

TRAINING = pathlib.Path(__file__).resolve().parents[2]
RUNTIME = TRAINING / "openpi_runtime"
for path in (TRAINING, RUNTIME / "src", RUNTIME / "packages" / "openpi-client" / "src", RUNTIME / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

VISION_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
TACTILE_FIRST_KEYS = ("left_tactile_0_rgb", "right_tactile_0_rgb", "base_0_rgb", "wrist_0_rgb")
HORIZON, ACTION_DIM, TOKENS, BATCH = 6, 32, 12, 2


def build_tiny(image_keys, cls=None, seed=0):
    """A real PI0Pytorch (pi0.5 mode) shrunk far enough to run on a laptop CPU."""
    import openpi.models.pi0_config as pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(seed)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
        pytorch_compile_mode=None,
        dtype="float32",
        image_keys=tuple(image_keys),
        max_token_len=TOKENS,
        action_horizon=HORIZON,
        action_dim=ACTION_DIM,
        # Tactile images go through the ordinary image encoder here; AnyTouch2 needs
        # its pretrained checkpoint, which CPU tests do not have.
        tactile_encoder_type="siglip",
    )
    model = (cls or PI0Pytorch)(config)
    paligemma = model.paligemma_with_expert.paligemma.model
    encoder = paligemma.vision_tower.vision_model.encoder
    encoder.layers = torch.nn.ModuleList(list(encoder.layers)[:1])
    width = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    old = paligemma.multi_modal_projector.linear
    paligemma.multi_modal_projector.linear = torch.nn.Linear(old.in_features, width)
    return model


def make_batch(seed=0, batch=BATCH):
    """A batch shaped like the vision+tactile data pipeline's output."""
    from openpi.models import model as _model

    generator = torch.Generator().manual_seed(seed)
    images = {key: torch.rand(batch, 3, 224, 224, generator=generator) * 2 - 1 for key in TACTILE_FIRST_KEYS}
    observation = _model.Observation(
        images=images,
        image_masks={key: torch.ones(batch, dtype=torch.bool) for key in TACTILE_FIRST_KEYS},
        state=torch.randn(batch, ACTION_DIM, generator=generator),
        tokenized_prompt=torch.randint(0, 1000, (batch, TOKENS), generator=generator),
        tokenized_prompt_mask=torch.ones(batch, TOKENS, dtype=torch.bool),
    )
    actions = torch.randn(batch, HORIZON, ACTION_DIM, generator=generator)
    noise = torch.randn(batch, HORIZON, ACTION_DIM, generator=generator)
    time = torch.rand(batch, generator=generator) * 0.9 + 0.05
    return observation, actions, noise, time


def vision_observation(observation):
    from tacguide import velocity

    return velocity.vision_observation(observation)


@pytest.fixture(scope="session")
def strong():
    return build_tiny(TACTILE_FIRST_KEYS, seed=1).eval()


@pytest.fixture(scope="session")
def weak():
    return build_tiny(VISION_KEYS, seed=2).eval()
