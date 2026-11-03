from hydra.core.config_store import ConfigStore

from vessel_seg.config.schemas import Config, DataConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="data_schema", group="data", node=DataConfig, replace=True)
    cs.store(name="config_schema", group=None, node=Config, replace=True)
