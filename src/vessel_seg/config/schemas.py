from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING


@dataclass
class DataConfig:
    root: Path
    image_size: int
    mask_threshold: float
    val_size: float | int
    batch_size: int
    num_workers: int = 0
    image_transform: Any = MISSING
    mask_transform: Any = MISSING
    augmentation_transform: Any = MISSING


@dataclass
class Config:
    data: DataConfig
    model: Any
    loss: Any
    optimizer: Any
    scheduler: Any
    train: Any
    seed: int = 44
