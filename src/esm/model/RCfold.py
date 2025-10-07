import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from simplefold.utils.esm_utils import _af2_to_esm, esm_model_dict
# from utils.esm_utils import _af2_to_esm

class ESM_Regressor(nn.Module):
    # initialize architecture
    # simple feed forward network
    def __init__(
        self,
        input_dim,
        hidden_size1 = 64,
        # hidden_size2 = 256,
        # hidden_size3 = 128,
        esm_embed_dim = 1280,
        # device = "cuda:0",
    ):
    # extract per-residue reprsentation
    # s representation
        super().__init__()
        # self.device = device
        # simplefold seems to do a different encoding than what 
        # is done in ESM

        # B: batchsize
        # N: number of residues in sequence (for amylase: 425)
        # E: embedding layer 33 (output) (1280)
        # B x N x 1280 
        # how do we pool the embeddings to pass into feed-forward
        # all of the sequences are quite similar ... 
        # so perhaps the pooling may not be so useful
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.LayerNorm(hidden_size1),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(hidden_size1, 1)

    def forward(self, input):
        # TODO: add optionality for pooling choices
        #  mean pooling...
        avg = torch.mean(input, dim=2)
        x = self.feed_forward(avg)
        out = self.output_layer(x)
        return out

class PL_ESM_Regressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 425, 
        # esm_model: str = "esm2_3B",
        loss_fn = nn.MSELoss(), 
        lr: float = 0.01
    ):
    # add config file that describes the architecture
        super().__init__()
        self.model = ESM_Regressor(
            input_dim=input_dim
        )
        self.loss_fn = loss_fn
        self.lr = lr
        
    def forward(self, inputs):
        return self.model(inputs)
    
    # define training_step
    def training_step(self, batch, batch_idx):
        x, y = batch['embed'], batch['label']
        y_hat = self.model(x).reshape(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    # validation_step
    def validation_step(self, batch, batch_idx):
        x, y = batch['embed'], batch['label']
        y_hat = self.model(x).reshape(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # configure_optimizers
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # test_step

    # predict_step
    # run model beginning from sequence->embedding->push through model


class Preprocess_Dataset(Dataset):
    def __init__(
        self,
        df_alignbio: pd.DataFrame,
    ):
        super().__init__()
        self.df_alignbio = df_alignbio

    def __len__(self) -> int :
        return len(self.df_alignbio)

    def __getitem__(self, idx: int) -> dict:
        return ( 
            self.df_alignbio["mutant"].iloc[idx],
            self.df_alignbio["mutated_sequence"].iloc[idx]
        )
    # i believe in esm/extract.py
    # the FastaBatchedDataset object loads as many sequences into
    # the batch as possible... this is more emory efficient than
    # using fixed number sequences 
    # see get_batch_indices() ... this passes a list of slices 
    # to batch the dataset
    # in our case lets just pass in X sequences in a batch it directly

    

# define a custom dataset
# load in any csv file (train/test), also specify the column that
# we will be making predictions to 
# columns of csv file are:
# mutant,dataset,mutated_sequence,expression,thermostability,specific activity
class AlignBio_Dataset(Dataset):
    def __init__(
        self, 
        csv: Path = None, 
        label_col: str = None, # specify the column that describes the training data
        cache: Path = None, 
    ):
        super().__init__()
        self.df_alignbio = pd.read_csv(csv)
        self.label_col = label_col
        self.df_alignbio = self.df_alignbio.dropna(subset=[self.label_col]).reset_index(drop=True)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.df_alignbio)
    
    # tokenization of the sequence here
    def preprocessing(self, esm_cache_outdir: Path, truncation_seq_length: int = 4096, batch_size: int = 4):
        self.truncation_seq_length = truncation_seq_length
        self.batch_size = batch_size
        load_fn = torch.hub.load
        ## later allow for flexible interchange of ESM directory
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
                print(
                    f"Processing {batch_idx + 1} of {len(data_loader)} batches: ({toks.size(0)} sequences per batch)"
                )
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.esm_model(toks, repr_layers=[33], return_contacts=False)
                logits = out["logits"].to(device="cpu")
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                # "per_tok" representation
                for i, label in enumerate(labels):
                    output_file = esm_cache_outdir / f"{label}.pt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    result = {"label": label}
                    truncate_len = min(self.truncation_seq_length, len(strs[i]))
                    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                    torch.save(
                        result,
                        output_file,
                    )


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
        input = torch.load(self.cache / Path(f"{sample['name']}.pt"))['representations'][33]
        sample["embed"] = input
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
        data_dir: Path = None, 
        csv: Path = None,
        esm_cache_outdir: Path = None, 
        label: str = "expression",
        batch_size: int = 32,
        preprocess: bool = False,
    ):
        super().__init__()    
        assert label in ["expression", "thermostability", "specific activity"]
        self.label = label
        self.csv: Path = data_dir / csv
        self.esm_cache_outdir = esm_cache_outdir
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.save_hyperparameters()

        # TODO: check if all files exist (like in DiffDock)
        if (
            self.esm_cache_outdir is not None
            and (not self.esm_cache_outdir.exists() 
                 or not any(self.esm_cache_outdir.iterdir()))
        ) or self.preprocess:
            self.esm_cache_outdir.mkdir(parents=True, exist_ok=True)
            AlignBio_Dataset(self.csv).preprocessing(self.esm_cache_outdir, batch_size=batch_size)

    # setup
    # can also try doing different splitting strategies
    # split by dataset column?
    def setup(self, stage: str) -> None:
        if stage == "fit":
            data = AlignBio_Dataset(self.csv, self.label, self.esm_cache_outdir)
            self.train, self.val = random_split(
                data, [0.8, 0.2], torch.Generator().manual_seed(42)
            )
        if stage == "test":
            self.test = AlignBio_Dataset(self.csv, self.label, self.esm_cache_outdir)

        if stage == "predict":
            self.predict = AlignBio_Dataset(self.csv, self.label, self.esm_cache_outdir)
        
    # return train/val/test/predict_dataloader -> Dataloader
    # dataloader also has various features like 
    # num_workers/pin_memory/shuffle/collate_fn
    def train_dataloader(self):
        return DataLoader(self.train, pin_memory=True, num_workers=2, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.val, pin_memory=True, num_workers=2, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


    # trainer.fit(model, dummy_data)
# load data into pytorch dataloader (add options for distributed data parallel)

# 

    