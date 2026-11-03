from pathlib import Path

import pytest

from vessel_seg.data.data_split import train_val_split


def _make_split_dir(root: Path, n: int) -> Path:
    image_dir = root / "image"
    image_dir.mkdir(parents=True)
    for i in range(n):
        (image_dir / f"{i}.png").touch()  # content irrelevant: split only globs filenames
    return root


def test_split_is_disjoint_and_complete(tmp_path: Path) -> None:
    path = _make_split_dir(tmp_path, 20)
    train_ids, val_ids = train_val_split(path, val_size=0.2, seed=0)

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(20))


@pytest.mark.parametrize(("val_size", "expected_val"), [(0.2, 4), (0.5, 10), (1, 1), (3, 3), (7, 7)])
def test_val_size_from_fraction_or_count(tmp_path: Path, val_size: float | int, expected_val: int) -> None:
    path = _make_split_dir(tmp_path, 20)
    train_ids, val_ids = train_val_split(path, val_size=val_size)

    assert len(val_ids) == expected_val
    assert len(train_ids) == 20 - expected_val


def test_ids_are_file_stems_not_positions(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    stems = [3, 10, 42, 100, 7]
    for stem in stems:
        (image_dir / f"{stem}.png").touch()

    train_ids, val_ids = train_val_split(tmp_path, val_size=2)

    assert sorted(train_ids + val_ids) == sorted(stems)
    assert train_ids == sorted(train_ids)
    assert val_ids == sorted(val_ids)


def test_split_is_deterministic_per_seed(tmp_path: Path) -> None:
    path = _make_split_dir(tmp_path, 50)

    assert train_val_split(path, 0.2, seed=42) == train_val_split(path, 0.2, seed=42)
    assert train_val_split(path, 0.2, seed=42) != train_val_split(path, 0.2, seed=1)


@pytest.mark.parametrize("val_size", [0, 0.0, -0.1, 10, 15])
def test_degenerate_val_size_raises(tmp_path: Path, val_size: float | int) -> None:
    path = _make_split_dir(tmp_path, 10)
    with pytest.raises(ValueError, match="Val size"):
        train_val_split(path, val_size=val_size)


def test_empty_dir_raises(tmp_path: Path) -> None:
    path = _make_split_dir(tmp_path, 0)
    with pytest.raises(ValueError, match="full data size is: 0"):
        train_val_split(path)
