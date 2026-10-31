import torch
from torchvision.transforms import v2


def get_image_transform(
    size: int = 512, mean: list[float] | None = None, std: list[float] | None = None
) -> v2.Transform:
    transforms: list[v2.Transform] = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((size, size)),
    ]
    # mean/std should be computed on the training split only
    if mean is not None and std is not None:
        transforms.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(transforms)


def get_mask_transform(size: int = 512) -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size)),
        ]
    )


def get_augmentation_transform(
    size: int = 512,
    degrees: float = 180,
    crop_scale: tuple[float, float] = (0.625, 1.0),
    crop_p: float = 1.0,
) -> v2.Transform:
    # Geometric only, so v2 applies the same params to Image and Mask (Mask uses nearest interpolation)
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=degrees),
            # Crop after rotation partially removes the black corners
            v2.RandomApply([v2.RandomResizedCrop(size, scale=crop_scale, ratio=(1, 1))], p=crop_p),
        ]
    )
