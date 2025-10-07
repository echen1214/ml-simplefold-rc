import torch
import wandb
import pytorch_lightning as pl
from simplefold.utils.pylogger import RankedLogger
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler
from pathlib import Path
from src.esm.datasets.dataset import AlignBio_DataModule
from src.esm.model.RCfold import PL_ESM_Regressor
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")
log = RankedLogger(__name__, rank_zero_only=True)

if __name__ == "__main__":
    data_dir = Path("/scratch/eac709/overlays/the-protein-engineering-tournament-2023/in_silico_supervised/input/Alpha-Amylase (In Silico_ Supervised)")
    esm_cache_dir = data_dir / Path("esm")
    train_csv = Path("train.csv")

    pl.seed_everything(42, workers=True)

    model = PL_ESM_Regressor(
        input_dim=425
    )

    run = wandb.init(
        entity="eac709-nyu",
        project="RCfold",
        name="ESM_regressor_0"
    )

    profiler = PyTorchProfiler(dirpath="lightning_logs", filename="ESM_regressor_0.txt")
    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=250,
        profiler=profiler,
        logger=wandb_logger
    )
    train_dm = AlignBio_DataModule(
        data_dir,
        train_csv,
        esm_cache_dir,
        "expression",
        batch_size=64,
    )

    trainer.fit(model=model, train_dataloaders=train_dm)
