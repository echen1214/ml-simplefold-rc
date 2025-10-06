from pathlib import Path
import pytorch_lightning as pl

from src.esm.model.RCfold import RCFold, AlignBio_DataModule, ESM_Regressor

if __name__ == "__main__":
    model = RCFold()
    trainer = pl.Trainer(max_epochs=10)
    # load data into dataloader object
    
    data_dir = Path("/scratch/eac709/overlays/the-protein-engineering-tournament-2023/in_silico_supervised/input/Alpha-Amylase (In Silico_ Supervised)")
    esm_cache_dir = Path("/scratch/eac709/overlays/the-protein-engineering-tournament-2023/in_silico_supervised/input/Alpha-Amylase (In Silico_ Supervised)/esm")

    csv = Path("train.csv")
    train_datamodule = AlignBio_DataModule(data_dir, csv, esm_cache_dir)

    model = ESM_Regressor()