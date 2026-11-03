from pathlib import Path

import torch


def train_val_split(
    path: str | Path,
    val_size: float | int = 0.2,
    seed: int = 44,
) -> tuple[list[int], list[int]]:
    """Split file ids (png stems) in `path/image` into train/val for RetinaDataset(indices=...).

    `val_size` in (0, 1) is a fraction of all images, otherwise the number of validation images.
    """
    image_dir = Path(path) / "image"
    ids = sorted(int(f.stem) for f in image_dir.glob("*.png"))  # sort -> independent of glob order

    if 0.0 < val_size < 1.0:
        val_size = val_size * len(ids)
    val_size = int(val_size)
    if val_size <= 0 or val_size >= len(ids):
        raise ValueError(f"Val size is: {val_size}, full data size is: {len(ids)}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ids), generator=generator).tolist()

    val_ids = sorted(ids[i] for i in perm[:val_size])
    train_ids = sorted(ids[i] for i in perm[val_size:])
    return train_ids, val_ids
