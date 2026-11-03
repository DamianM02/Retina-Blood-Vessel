from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import execution_whitelist, instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2

CONFIG_DIR = Path(__file__).parents[1] / "configs"
GROUPS = sorted(p.name for p in CONFIG_DIR.iterdir() if p.is_dir() and any(p.glob("*.yaml")))
GROUP_OPTIONS = [(group, option.stem) for group in GROUPS for option in sorted((CONFIG_DIR / group).glob("*.yaml"))]
TRUSTED_TARGETS = ["vessel_seg.*", "torchvision.*", "segmentation_models_pytorch.*", "torch.*"]


def _compose(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides or [])


def test_full_config_composes_and_resolves() -> None:
    cfg = _compose()
    OmegaConf.resolve(cfg)  # catches broken interpolations like ${data.size}

    assert set(cfg) >= {"data", "model", "loss", "optimizer", "scheduler", "train", "seed"}
    assert cfg.data.image_transform.size == cfg.data.image_size


@pytest.mark.parametrize(("group", "option"), GROUP_OPTIONS, ids=lambda x: str(x))
def test_every_group_option_composes(group: str, option: str) -> None:
    cfg = _compose([f"{group}={option}"])
    OmegaConf.resolve(cfg)


@pytest.mark.parametrize(("group", "option"), GROUP_OPTIONS, ids=lambda x: str(x))
def test_every_target_in_group_option_is_trusted(group: str, option: str) -> None:
    cfg = OmegaConf.to_container(_compose([f"{group}={option}"])[group], resolve=True)
    targets = _collect_targets(cfg)
    prefixes = tuple(t.removesuffix("*") for t in TRUSTED_TARGETS)

    assert all(t.startswith(prefixes) for t in targets), targets


def _collect_targets(node: object) -> list[str]:
    if isinstance(node, dict):
        own = [node["_target_"]] if "_target_" in node else []
        return own + [t for value in node.values() for t in _collect_targets(value)]
    if isinstance(node, list):
        return [t for value in node for t in _collect_targets(value)]
    return []


def test_size_override_propagates_to_transforms() -> None:
    cfg = _compose(["data.image_size=64"])

    assert cfg.data.image_transform.size == 64
    assert cfg.data.mask_transform.size == 64
    assert cfg.data.augmentation_transform.size == 64


@pytest.mark.parametrize("key", ["image_transform", "mask_transform", "augmentation_transform"])
def test_data_transforms_instantiate(key: str) -> None:
    cfg = _compose()
    with execution_whitelist(TRUSTED_TARGETS):
        transform = instantiate(cfg.data[key])

    assert isinstance(transform, v2.Transform)


def test_schema_rejects_wrong_type() -> None:
    with pytest.raises(ConfigCompositionException):
        _compose(["data.image_size=not_a_number"])


def test_schema_rejects_unknown_data_key() -> None:
    # DataConfig is a structured config: a typo in an override must fail instead of being silently ignored
    with pytest.raises(ConfigCompositionException):
        _compose(["data.imagesize=64"])


def test_seed_override() -> None:
    assert _compose(["seed=7"]).seed == 7
