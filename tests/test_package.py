import vessel_seg


def test_package_imports() -> None:
    assert hasattr(vessel_seg, "main")
