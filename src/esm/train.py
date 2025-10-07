import torch
import wandb
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from simplefold.utils.pylogger import RankedLogger
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler
from pathlib import Path
from .datasets.dataset import AlignBio_DataModule
from .model.RCfold import PL_ESM_Regressor
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
# log = RankedLogger(__name__, rank_zero_only=True)

def train(cfg):
    pl.seed_everything(42, workers=True)

    print(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    profiler = PyTorchProfiler(dirpath="lightning_logs", filename="ESM_regressor_0.txt")
    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch, 
        profiler=profiler,
        logger=wandb_logger
    )

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    train_dm: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    trainer.fit(model=model, train_dataloaders=train_dm)

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    train(cfg)
    return

if __name__ == "__main__":
    submit_run()
