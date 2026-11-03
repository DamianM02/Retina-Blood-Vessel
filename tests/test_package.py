import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["vessel_seg", "vessel_seg.config", "vessel_seg.data.datamodule", "vessel_seg.models.vanilla_unet"],
)
def test_modules_import(module: str) -> None:
    importlib.import_module(module)
