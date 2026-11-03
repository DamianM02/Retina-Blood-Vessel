"""Training entry point: `uv run python -m vessel_seg.scripts.train [overrides]`."""

import hydra

from vessel_seg.config import Config, register_configs
from vessel_seg.config.execution_whitelist import EXECUTION_WHITELIST
from vessel_seg.data.datamodule import get_train_val_dataloaders

# TODO: thin wrapper: seed, start MLflow run, call training.trainer.fit(...), return metric (for sweepers)


register_configs()


@hydra.main(config_path="../../../configs", config_name="config", execution_whitelist=EXECUTION_WHITELIST)
def train(cfg: Config) -> None:
    train_dataloader, val_dataloader = get_train_val_dataloaders(cfg=cfg.data, seed=cfg.seed)
    type(train_dataloader), type(val_dataloader)


if __name__ == "__main__":
    train()
