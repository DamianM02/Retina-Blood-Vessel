from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import execution_whitelist
from PIL import Image

from vessel_seg.config import DataConfig
from vessel_seg.data.datamodule import get_train_val_dataloaders

CONFIG_DIR = Path(__file__).parents[1] / "configs"
TRUSTED_TARGETS = ["vessel_seg.*", "torchvision.*"]
N_IMAGES = 10
SIZE = 32


def _make_raw_dir(root: Path, n: int = N_IMAGES, size: int = SIZE) -> Path:
    """Fake data/raw/train with RGB images and 0/255 masks."""
    train = root / "train"
    (train / "image").mkdir(parents=True)
    (train / "mask").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in range(n):
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        mask = (rng.random((size, size)) > 0.5).astype(np.uint8) * 255
        Image.fromarray(image, mode="RGB").save(train / "image" / f"{idx}.png")
        Image.fromarray(mask, mode="L").save(train / "mask" / f"{idx}.png")
    return root


def _data_cfg(root: Path, *overrides: str) -> DataConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            # val_size set explicitly so tests do not depend on the value in retina.yaml; later overrides win
            overrides=[
                f"data.root={root}",
                f"data.image_size={SIZE}",
                "data.batch_size=4",
                "data.val_size=0.2",
                *overrides,
            ],
        )
    return cfg.data


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return _make_raw_dir(tmp_path)


def _load(cfg: DataConfig, seed: int = 0) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    with execution_whitelist(TRUSTED_TARGETS):
        return get_train_val_dataloaders(cfg=cfg, seed=seed)


def test_split_sizes(raw_dir: Path) -> None:
    train_loader, val_loader = _load(_data_cfg(raw_dir, "data.val_size=0.2"))

    assert len(train_loader.dataset) == 8  # type: ignore[arg-type]
    assert len(val_loader.dataset) == 2  # type: ignore[arg-type]


def test_train_and_val_are_disjoint(raw_dir: Path) -> None:
    train_loader, val_loader = _load(_data_cfg(raw_dir))

    train_files = set(train_loader.dataset.filenames)  # type: ignore[attr-defined]
    val_files = set(val_loader.dataset.filenames)  # type: ignore[attr-defined]
    assert train_files.isdisjoint(val_files)
    assert len(train_files | val_files) == N_IMAGES


def test_split_depends_on_seed(raw_dir: Path) -> None:
    cfg = _data_cfg(raw_dir, "data.val_size=0.5")
    val_a = _load(cfg, seed=0)[1].dataset.filenames  # type: ignore[attr-defined]
    val_b = _load(cfg, seed=0)[1].dataset.filenames  # type: ignore[attr-defined]
    val_c = _load(cfg, seed=1)[1].dataset.filenames  # type: ignore[attr-defined]

    assert val_a == val_b
    assert val_a != val_c


def test_batch_shapes_and_binary_masks(raw_dir: Path) -> None:
    train_loader, val_loader = _load(_data_cfg(raw_dir))

    for loader in (train_loader, val_loader):
        images, masks = next(iter(loader))
        assert images.shape[1:] == (3, SIZE, SIZE)
        assert masks.shape[1:] == (1, SIZE, SIZE)
        assert images.dtype == masks.dtype == torch.float32
        assert set(masks.unique().tolist()) <= {0.0, 1.0}


def test_images_are_normalized(raw_dir: Path) -> None:
    # ImageNet mean/std in retina.yaml -> values leave the [0, 1] range
    _, val_loader = _load(_data_cfg(raw_dir))
    images, _ = next(iter(val_loader))

    assert images.min() < 0.0


def test_val_has_no_augmentation(raw_dir: Path) -> None:
    _, val_loader = _load(_data_cfg(raw_dir))

    assert val_loader.dataset.augmentation_transform is None  # type: ignore[attr-defined]
    first = next(iter(val_loader))[0]
    second = next(iter(val_loader))[0]
    assert torch.equal(first, second)


def test_train_has_augmentation(raw_dir: Path) -> None:
    train_loader, _ = _load(_data_cfg(raw_dir))

    assert train_loader.dataset.augmentation_transform is not None  # type: ignore[attr-defined]


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full data size is: 0"):
        _load(_data_cfg(tmp_path / "does_not_exist", "data.val_size=0.2"))
