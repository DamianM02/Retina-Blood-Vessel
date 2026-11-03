from pathlib import Path

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from vessel_seg.config import DataConfig
from vessel_seg.data.data_split import train_val_split
from vessel_seg.data.dataset import RetinaDataset


def get_train_val_dataloaders(cfg: DataConfig, seed: int) -> tuple[DataLoader, DataLoader]:
    image_transform = instantiate(cfg.image_transform)
    mask_transform = instantiate(cfg.mask_transform)
    augmentation_transform = instantiate(cfg.augmentation_transform)

    train_idx, val_idx = train_val_split(path=Path(cfg.root) / "train", val_size=cfg.val_size, seed=seed)

    train_dataset = RetinaDataset(
        path=Path(cfg.root) / "train",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augmentation_transform=augmentation_transform,
        indices=train_idx,
        mask_threshold=cfg.mask_threshold,
    )
    val_dataset = RetinaDataset(
        path=Path(cfg.root) / "train",
        image_transform=image_transform,
        mask_transform=mask_transform,
        augmentation_transform=None,
        indices=val_idx,
        mask_threshold=cfg.mask_threshold,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    return train_dataloader, val_dataloader
