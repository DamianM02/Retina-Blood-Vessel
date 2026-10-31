from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from vessel_seg.data.dataset import RetinaDataset
from vessel_seg.data.transforms import get_image_transform, get_mask_transform

SIZE = 32


def _make_split(root: Path, indices: list[int], size: int = SIZE) -> Path:
    """Create a fake split: root/image/{idx}.png (RGB) and root/mask/{idx}.png (L, values 0/255)."""
    (root / "image").mkdir(parents=True)
    (root / "mask").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in indices:
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[:, : size // 2] = 255
        Image.fromarray(image, mode="RGB").save(root / "image" / f"{idx}.png")
        Image.fromarray(mask, mode="L").save(root / "mask" / f"{idx}.png")
    return root


@pytest.fixture
def split(tmp_path: Path) -> Path:
    return _make_split(tmp_path / "train", [0, 1, 2, 10])


def test_len_matches_number_of_files(split: Path) -> None:
    assert len(RetinaDataset(split)) == 4


def test_filenames_sorted_numerically(split: Path) -> None:
    assert RetinaDataset(split).filenames == ["0.png", "1.png", "2.png", "10.png"]


def test_indices_select_subset(split: Path) -> None:
    dataset = RetinaDataset(split, indices=[10, 1])
    assert dataset.filenames == ["1.png", "10.png"]
    assert len(dataset) == 2


def test_missing_index_raises(split: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RetinaDataset(split, indices=[99])


def test_item_shapes_and_types(split: Path) -> None:
    image, mask = RetinaDataset(split)[0]
    assert isinstance(image, tv_tensors.Image)
    assert isinstance(mask, tv_tensors.Mask)
    assert image.shape == (3, SIZE, SIZE)
    assert mask.shape == (1, SIZE, SIZE)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_image_scaled_to_unit_range(split: Path) -> None:
    image, _ = RetinaDataset(split)[0]
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_mask_is_binary(split: Path) -> None:
    _, mask = RetinaDataset(split)[0]
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    assert mask[0, :, : SIZE // 2].eq(1).all()
    assert mask[0, :, SIZE // 2 :].eq(0).all()


def test_mask_threshold_applied(split: Path) -> None:
    # Mask pixels are 1.0 after scaling, so a threshold of 1.0 must zero everything out
    _, mask = RetinaDataset(split, mask_threshold=1.0)[0]
    assert mask.sum() == 0


def test_mask_transform_without_scaling_raises(split: Path) -> None:
    unscaled = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
    with pytest.raises(ValueError, match="mask_transform must scale"):
        RetinaDataset(split, mask_transform=unscaled)


def test_augmentation_applied_to_image_and_mask_together(split: Path) -> None:
    dataset = RetinaDataset(split, augmentation_transform=v2.RandomHorizontalFlip(p=1.0))
    image, mask = dataset[0]
    assert torch.equal(image, dataset.images[0].flip(-1))
    assert torch.equal(mask, dataset.masks[0].flip(-1))


def test_dataloader_batches(split: Path) -> None:
    images, masks = next(iter(DataLoader(RetinaDataset(split), batch_size=2)))
    assert images.shape == (2, 3, SIZE, SIZE)
    assert masks.shape == (2, 1, SIZE, SIZE)


def test_project_transforms(tmp_path: Path) -> None:
    split = _make_split(tmp_path / "train", [0], size=600)
    image, mask = RetinaDataset(split, image_transform=get_image_transform(), mask_transform=get_mask_transform())[0]
    assert image.shape[-2:] == mask.shape[-2:]
