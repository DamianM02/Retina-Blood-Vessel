from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class RetinaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    _simple_load_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __init__(
        self,
        path: str | Path,
        image_transform: v2.Transform | None = None,
        mask_transform: v2.Transform | None = None,
        augmentation_transform: v2.Transform | None = None,
        indices: list[int] | None = None,
        mask_threshold: float = 0.15,
    ) -> None:

        super().__init__()
        path = Path(path)  # Hydra passes paths from YAML as str
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation_transform = augmentation_transform
        images = []
        masks = []

        if indices is None:
            self.filenames = [file.name for file in (path / "image").glob("*.png")]

        else:
            self.filenames = [f"{idx}.png" for idx in indices]
        if not self.filenames:
            raise FileNotFoundError(f"No .png files found in {path / 'image'}")

        self.filenames = sorted(self.filenames, key=lambda name: int(Path(name).stem))

        for filename in self.filenames:
            image_path, mask_path = path / "image" / filename, path / "mask" / filename
            image = self._load_image(image_path, "RGB", self.image_transform)
            mask = self._load_image(mask_path, "L", self.mask_transform)
            if mask.max() > 1:
                raise ValueError(f"{mask_path}: mask_transform must scale to [0, 1].")
            mask = (mask > mask_threshold).float()
            images.append(image)
            masks.append(mask)

        self.images = torch.stack(images)
        self.masks = torch.stack(masks)

    @staticmethod
    def _load_image(path: Path, mode: Literal["RGB", "L"], transform: v2.Transform | None) -> torch.Tensor:
        with Image.open(path) as image_file:
            image = image_file.convert(mode)
            if transform is None:
                transform = RetinaDataset._simple_load_transform
            return transform(image)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = tv_tensors.Image(self.images[index])
        mask = tv_tensors.Mask(self.masks[index])
        if self.augmentation_transform is not None:
            return self.augmentation_transform(image, mask)
        else:
            return image, mask

    def __len__(self) -> int:
        return len(self.images)
