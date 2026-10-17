import shutil
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "abdallahwagih/retina-blood-vessel"
DEFAULT_RAW_DIR_PATH = Path("data") / "raw"


# DEFAULT_RAW_DIR_PATH = "data/raw"


def validate_layout(raw_dir: Path | str = DEFAULT_RAW_DIR_PATH) -> None:
    raw_dir = Path(raw_dir)
    for split in ("train", "test"):
        image_dir = raw_dir / split / "image"
        mask_dir = raw_dir / split / "mask"
        for folder in (image_dir, mask_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected directory not found: {folder}")

        images_names_set = {image_path.name for image_path in image_dir.iterdir()}
        masks_names_set = {mask_path.name for mask_path in mask_dir.iterdir()}

        if missing_masks := images_names_set - masks_names_set:
            raise ValueError(f"Images without masks in {raw_dir / split}: {sorted(missing_masks)}")
        if missing_images := masks_names_set - images_names_set:
            raise ValueError(f"Masks without images in {raw_dir / split}: {sorted(missing_images)}")

    print("Validation complete. Structure of dataset is correct.")


def download_raw_dataset(
    kaggle_dataset_path: str = KAGGLE_DATASET, raw_dir_path: Path | str = DEFAULT_RAW_DIR_PATH, force: bool = False
) -> Path:
    raw_dir_path = Path(raw_dir_path)
    if (
        raw_dir_path.is_dir()
        and any(children for children in raw_dir_path.iterdir() if not children.name.startswith("."))
        and not force
    ):
        print(f"Dataset already exists in {raw_dir_path}.")
        return raw_dir_path

    print("Downloading dataset...")
    if raw_dir_path.exists():
        shutil.rmtree(raw_dir_path)
    cache_dir = Path(kagglehub.dataset_download(kaggle_dataset_path)) / "Data"
    shutil.copytree(src=cache_dir, dst=raw_dir_path, dirs_exist_ok=False)
    print(f"Dataset saved to {raw_dir_path}.")
    return raw_dir_path


if __name__ == "__main__":
    download_raw_dataset(force=False)
    validate_layout()
