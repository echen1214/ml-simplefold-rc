import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl

# Preprocess_Dataset
# i believe in esm/extract.py
# the FastaBatchedDataset object loads as many sequences into
# the batch as possible... this is more memory efficient than
# using fixed number sequences 
# see get_batch_indices() ... this passes a list of slices 
# to batch the dataset
# in our case lets just pass in X sequences in a batch it directly
class Preprocess_Dataset(Dataset):
    def __init__(self, df_alignbio: pd.DataFrame):
        super().__init__()
        self.df_alignbio = df_alignbio

    def __len__(self) -> int:
        return len(self.df_alignbio)

    def __getitem__(self, idx: int):
        return (
            self.df_alignbio["mutant"].iloc[idx],
            self.df_alignbio["mutated_sequence"].iloc[idx]
        )

# AlignBio_Dataset
# define a custom dataset
# load in any csv file (train/test), also specify the column that
# we will be making predictions to 
# columns of csv file are:
# mutant,dataset,mutated_sequence,expression,thermostability,specific activity
class AlignBio_Dataset(Dataset):
    def __init__(
        self, 
        csv: Path = None, 
        label_col: str = "expression", 
        cache: Path = None, 
        aggregate_replicates: bool = False,
        use_all_zeros = False,
        use_bos: bool = False,
    ):
        super().__init__()

        self.df_alignbio = pd.read_csv(csv)
        self.label_col = label_col
        if self.label_col is not None and self.label_col not in self.df_alignbio.columns:
            raise ValueError(
                f"Label column '{self.label_col}' not found in CSV columns: {list(self.df_alignbio.columns)}"
            )
        # Only drop NaN values if label_col is specified (for training/inference)
        if self.label_col is not None:
            self.df_alignbio = self.df_alignbio.dropna(subset=[self.label_col]).reset_index(drop=True)
        self.cache = cache
        self.use_all_zeros = use_all_zeros
        self.use_bos = use_bos

    def __len__(self) -> int:
        return len(self.df_alignbio)

    # tokenization of the sequence here
    def preprocessing(self, esm_cache_dir: Path, truncation_seq_length: int = 4096, batch_size: int = 4):
        self.truncation_seq_length = truncation_seq_length
        self.batch_size = batch_size
        load_fn = torch.hub.load
        # later allow for flexible interchange of ESM directory
        # load in esm model from cache 
        self.esm_model, self.esm_dict = load_fn("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.esm_model.eval()
        if torch.cuda.is_available():
            print("Transferred model to GPU")
            self.esm_model.cuda()

        pp_data = Preprocess_Dataset(self.df_alignbio)
        batch_converter = self.esm_dict.get_batch_converter()
        data_loader = DataLoader(pp_data, collate_fn=batch_converter, batch_size=self.batch_size)

        # following https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
        # but with a more simple collate_fn
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(f"Processing {batch_idx + 1} of {len(data_loader)} batches: ({toks.size(0)} sequences per batch)")
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.esm_model(toks, repr_layers=[33], return_contacts=False)
                logits = out["logits"].to(device="cpu")
                representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
                for i, label in enumerate(labels):
                    output_file = esm_cache_dir / f"{label}.pt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    result = {"label": label}
                    truncate_len = min(self.truncation_seq_length, len(strs[i]))
                    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
                    # Save BOS/CLS token embedding at index 0
                    result["bos"] = {
                        layer: t[i, 0].clone()
                        for layer, t in representations.items()
                    }
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                    torch.save(result, output_file)

    # return batch item 
    # for memory heavy datasets load in csv find the file and load it in
    # this makes sense for the esm cache ... so the model doesn't load
    # the entire cache
    # do we transform the labels here?
    def __getitem__(self, idx: int) -> dict:
        # convert AA sequence to tokens
        sample = {
            "name": self.df_alignbio["mutant"].iloc[idx],
            "seq": self.df_alignbio["mutated_sequence"].iloc[idx],
            "label": torch.tensor(float(self.df_alignbio[self.label_col].iloc[idx]), dtype=torch.float32)
        }
        obj = torch.load(self.cache / Path(f"{sample['name']}.pt"))
        reps = obj["representations"][33]
        if self.use_bos:
            bos = obj["bos"][33].unsqueeze(0)
            embed = torch.cat([bos, reps], dim=0)
        else:
            embed = reps
        if self.use_all_zeros:
            embed = torch.zeros_like(embed)
        sample["embed"] = embed
        return sample

# pytorch lightning wrapper to prepare the datasets
# user should pass in the csv path name corresponding to train&val/test/predict set
# refactor so that AlignBio_dataset gets initialized outside?
# TODO:
# -[ ] add optionality to split by dataset column
class AlignBio_DataModule(pl.LightningDataModule):
    # target_csv dataset
    def __init__(
        self, 
        data_dir: str = None,
        csv: str = None,
        esm_cache_dir: str = None,
        label: str = "expression",
        batch_size: int = 32,
        preprocess: bool = False,
        data_root: str = None,
        use_all_zeros: bool = False,
        use_bos: bool = False,
    ):
        super().__init__()    
        # assert label in ["expression", "thermostability", "specific activity"]
        self.label = label
        self.use_all_zeros = use_all_zeros
        self.use_bos = use_bos
        if self.use_all_zeros:
            print("WARNING: dataset will be set to all zeroes for baseline checking!")

        # Resolve and validate data paths
        data_dir_path = Path(data_dir) if data_dir is not None else None
        if data_dir_path is None or not data_dir_path.exists():
            raise FileNotFoundError(
                f"data_dir does not exist: {data_dir_path}. Check your data_root and experiment.protein_dir_name."
            )
        self.csv: Path = data_dir_path / Path(csv)
        if not self.csv.exists():
            raise FileNotFoundError(f"CSV not found at resolved path: {self.csv}")
        self.esm_cache_dir = Path(esm_cache_dir)
        # Ensure the ESM cache directory exists early
        self.esm_cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.save_hyperparameters()

        # TODO: check if all files exist (like in DiffDock)
        if (
            self.esm_cache_dir is not None
            and (not self.esm_cache_dir.exists() or not any(self.esm_cache_dir.iterdir()))
        ) or self.preprocess:
            self.esm_cache_dir.mkdir(parents=True, exist_ok=True)
            AlignBio_Dataset(self.csv).preprocessing(self.esm_cache_dir, batch_size=batch_size)

    # setup
    # can also try doing different splitting strategies
    # split by dataset column?
    def setup(self, stage: str) -> None:
        if stage == "fit":
            data = AlignBio_Dataset(
                self.csv,
                self.label,
                self.esm_cache_dir,
                use_all_zeros=self.use_all_zeros,
                use_bos=self.use_bos,
            )
            self.train, self.val = random_split(
                data,
                [0.8, 0.2],
                torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test = AlignBio_Dataset(
                self.csv,
                self.label,
                self.esm_cache_dir,
                use_all_zeros=self.use_all_zeros,
                use_bos=self.use_bos,
            )
        if stage == "predict":
            self.predict = AlignBio_Dataset(
                self.csv,
                self.label,
                self.esm_cache_dir,
                use_all_zeros=self.use_all_zeros,
                use_bos=self.use_bos,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            batch_size=self.batch_size,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            batch_size=self.batch_size,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            batch_size=self.batch_size,
        )
    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            batch_size=self.batch_size,
        )
