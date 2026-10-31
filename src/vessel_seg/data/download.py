"""Download the Retina Blood Vessel dataset from Kaggle and validate its folder layout.

Expected layout of the raw dataset::

    data/raw/
    ├── train/
    │   ├── image/  0.png, 1.png, ...
    │   └── mask/   0.png, 1.png, ...
    └── test/
        ├── image/
        └── mask/
"""

import shutil
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "abdallahwagih/retina-blood-vessel"
"""Kaggle dataset handle in the ``owner/dataset-name`` format."""

DEFAULT_RAW_DIR_PATH = Path("data") / "raw"
"""Default directory (relative to the project root) where the raw dataset is stored."""


def validate_layout(raw_dir: Path | str = DEFAULT_RAW_DIR_PATH) -> None:
    """Check that the raw dataset has the expected folder structure.

    For every split (``train`` and ``test``) the function checks that the
    ``image/`` and ``mask/`` folders exist, are not empty, and that every image
    has a mask with the same file name (and vice versa).

    Args:
        raw_dir: Root directory of the raw dataset, containing the ``train/``
            and ``test/`` folders.

    Raises:
        FileNotFoundError: If an ``image/`` or ``mask/`` folder is missing or empty.
        ValueError: If some images have no matching mask, or some masks have
            no matching image.
    """
    raw_dir = Path(raw_dir)
    for split in ("train", "test"):
        image_dir = raw_dir / split / "image"
        mask_dir = raw_dir / split / "mask"
        for folder in (image_dir, mask_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected directory not found: {folder}")

        if not (images_names_set := {image_path.name for image_path in image_dir.iterdir()}):
            raise FileNotFoundError(f"No images in: {raw_dir / split}")

        if not (masks_names_set := {mask_path.name for mask_path in mask_dir.iterdir()}):
            raise FileNotFoundError(f"No masks in: {raw_dir / split}")

        if missing_masks := images_names_set - masks_names_set:
            raise ValueError(f"Images without masks in {raw_dir / split}: {sorted(missing_masks)}")
        if missing_images := masks_names_set - images_names_set:
            raise ValueError(f"Masks without images in {raw_dir / split}: {sorted(missing_images)}")

    print("Validation complete. Structure of dataset is correct.")


def download_raw_dataset(
    kaggle_dataset_path: str = KAGGLE_DATASET, raw_dir_path: Path | str = DEFAULT_RAW_DIR_PATH, force: bool = False
) -> Path:
    """Download the dataset with kagglehub and copy it into ``raw_dir_path``.

    kagglehub keeps its own cache (``~/.cache/kagglehub``), so the files are
    fetched from the internet only once; later calls copy them from the cache.
    If ``raw_dir_path`` already contains data (hidden files such as
    ``.DS_Store`` are ignored), nothing is downloaded unless ``force`` is set.

    Args:
        kaggle_dataset_path: Kaggle dataset handle in the ``owner/dataset-name`` format.
        raw_dir_path: Target directory for the raw dataset.
        force: If ``True``, delete the existing ``raw_dir_path`` and copy
            the dataset again.

    Returns:
        Path to the directory with the raw dataset.
    """
    raw_dir_path = Path(raw_dir_path)
    if (
        raw_dir_path.is_dir()
        and any(children for children in raw_dir_path.iterdir() if not children.name.startswith("."))
        and not force
    ):
        print(f"Dataset already exists in {raw_dir_path}.")
        try:
            validate_layout(raw_dir=raw_dir_path)
        except (FileNotFoundError, ValueError) as err:
            raise ValueError(
                f"Dataset in {raw_dir_path} has incorrect structure, use arg force=True to download right version."
            ) from err

        return raw_dir_path

    print("Downloading dataset...")
    cache_dir = Path(kagglehub.dataset_download(kaggle_dataset_path)) / "Data"
    if raw_dir_path.exists():
        shutil.rmtree(raw_dir_path)
    shutil.copytree(src=cache_dir, dst=raw_dir_path, dirs_exist_ok=False)
    print(f"Dataset saved to {raw_dir_path}.")
    validate_layout(raw_dir=raw_dir_path)
    return raw_dir_path


if __name__ == "__main__":
    download_raw_dataset(force=False)
