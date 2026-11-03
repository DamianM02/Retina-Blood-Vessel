from vessel_seg.config import register_configs

# Register structured config schemas once per test session (re-registering warns in Hydra 1.4)
register_configs()
