from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from vessel_seg.models.vanilla_unet import VanillaUNet

# Hydra 1.4: only these packages may be instantiated from config
TRUSTED_TARGETS = ["segmentation_models_pytorch.*", "vessel_seg.*"]
MODEL_CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "model").glob("*.yaml"))


def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_model_configs_exist() -> None:
    assert MODEL_CONFIGS, "no model configs found"


@pytest.mark.parametrize("config_path", MODEL_CONFIGS, ids=lambda p: p.stem)
def test_model_config_forward_shape(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    if "encoder_weights" in cfg:
        cfg.encoder_weights = None  # no weight downloads in tests / CI
    model = instantiate(cfg, _execution_whitelist_=TRUSTED_TARGETS).eval()

    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))

    assert out.shape == (2, 1, 64, 64)


def test_vanilla_unet_handles_size_not_divisible_by_stride() -> None:
    model = VanillaUNet(depth=4).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 50, 50))
    assert out.shape == (1, 1, 50, 50)


def test_vanilla_unet_depth_controls_size() -> None:
    assert _n_params(VanillaUNet(depth=2)) < _n_params(VanillaUNet(depth=4))


def test_vanilla_unet_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        VanillaUNet(depth=0)
