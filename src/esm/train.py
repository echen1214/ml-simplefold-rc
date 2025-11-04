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
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

torch.set_float32_matmul_precision("medium")
# log = RankedLogger(__name__, rank_zero_only=True)

def train(cfg):
    pl.seed_everything(42, workers=True)

    print(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.profiler.init:
        profiler = PyTorchProfiler(dirpath=cfg.profiler.dir, filename=f"profile_{cfg.profiler.filename}")
    else:
        profiler = None

    if cfg.wandb.init:
        wandb.init(
            dir=cfg.wandb.dir,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            group=cfg.wandb.get("group", None),
            tags=cfg.wandb.get("tags", []),
            notes=cfg.wandb.get("notes", None),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger = WandbLogger()
    else:
        logger = CSVLogger(save_dir=cfg.trainer.default_root_dir, name=cfg.job_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/spearman",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="{epoch:02d}-{valid_spearman:.4f}",
        auto_insert_metric_name=False,
        every_n_epochs=5,
    )

    earlystopping_callback = EarlyStopping(
        monitor="valid/spearman",
        mode="max",
        patience=10,
        check_on_train_epoch_end=True,
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epoch, 
        profiler=profiler,
        logger=logger,
        callbacks=[checkpoint_callback, earlystopping_callback]
    )

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    train_dm: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    trainer.fit(model=model, train_dataloaders=train_dm, ckpt_path=cfg.get("restore_ckpt"))

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    train(cfg)
    return

if __name__ == "__main__":
    submit_run()
