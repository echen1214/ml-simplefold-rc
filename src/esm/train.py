import torch
import pytorch_lightning as pl
from simplefold.utils.pylogger import RankedLogger
from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler
from pathlib import Path
from src.esm.model.RCfold import PL_ESM_Regressor, AlignBio_DataModule

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

    profiler = PyTorchProfiler(dirpath=".", filename="perf_logs.txt")
    trainer = pl.Trainer(
        max_epochs=1,
        profiler=profiler
        # logger=log
    )
    train_dm = AlignBio_DataModule(
        data_dir,
        train_csv,
        esm_cache_dir,
        "expression",
        batch_size=64,
    )

    trainer.fit(model=model, train_dataloaders=train_dm)
