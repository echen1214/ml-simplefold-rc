import torch
import pytorch_lightning as pl
import wandb
from torch import nn
from torchmetrics.functional import spearman_corrcoef
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

class ESM_Regressor(nn.Module):
    # initialize architecture
    # simple feed forward network
    def __init__(
        self,
        input_dim,
        input_mean_axis = 2,
        hidden_size1 = 64,
        # hidden_size2 = 256,
        # hidden_size3 = 128,
        esm_embed_dim = 1280,
        # device = "cuda:0",
        dropout_rate: float = 0.1,
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
        self.input_mean_axis = input_mean_axis
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.LayerNorm(hidden_size1),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.output_layer = nn.Linear(hidden_size1, 1)

    def forward(self, input):
        # TODO: add optionality for pooling choices
        #  mean pooling over sequence length to get [B, 1280]
        avg = torch.mean(input, dim=self.input_mean_axis)
        x = self.feed_forward(avg)
        out = self.output_layer(x)
        return out

class PL_ESM_Regressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 425, 
        input_mean_axis: int = 2,
        hidden_size1: int = 64, 
        # esm_model: str = "esm2_3B",
        loss_fn = nn.MSELoss(), 
        lr: float = 0.01,
        dropout_rate: float = 0.1,
    ):
    # add config file that describes the architecture
        super().__init__()
        self.model = ESM_Regressor(
            input_dim=input_dim,
            input_mean_axis=input_mean_axis,
            hidden_size1=hidden_size1,
            esm_embed_dim=1280,
            dropout_rate=dropout_rate
        )
        self.loss_fn = loss_fn
        self.lr = lr
        # Save key hyperparameters into the Lightning checkpoint automatically
        # (exclude non-serializable objects like loss_fn)
        self.save_hyperparameters(ignore=["loss_fn"])
        self.test_step_outputs = []
        self.test_step_y = []
        
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

    def on_save_checkpoint(self, checkpoint):
        dm = self.trainer.datamodule
        data = f"{dm.csv.parent.name}/{dm.csv.name}"
        checkpoint["data"] = data
        checkpoint["label"] = dm.label
        # Save the W&B run URL for later reference
        if isinstance(self.logger, WandbLogger):
            run = getattr(self.logger, "experiment", None)
            run_url = getattr(run, "url", None)
            if run_url:
                checkpoint["wandb_run_url"] = run_url

    def on_load_checkpoint(self, checkpoint):
        # Load the W&B run URL from the checkpoint
        self.train_run_url = checkpoint.get("wandb_run_url")

    def test_step(self, batch, batch_idx):
        x, y = batch['embed'], batch['label']
        y_hat = self.model(x).reshape(-1)
        self.test_step_outputs.extend(y_hat)
        self.test_step_y.extend(y)
        return {'y_hat': y_hat.detach().cpu(), 'y': y.detach().cpu()}

    def on_test_epoch_end(self):
        y_hat = torch.tensor(self.test_step_outputs)
        y = torch.tensor(self.test_step_y)
        pearson_r = torch.corrcoef(torch.stack([y_hat, y]))[0, 1].item()
        rho = spearman_corrcoef(y_hat, y).item()
        ckpt = getattr(self, "ckpt_path", None)
        dm = self.trainer.datamodule
        test_data = f"{dm.csv.parent.name}/{dm.csv.name}"
        train_run_url = getattr(self.logger._checkpoint, "url", None)
        columns = ["train_run_url", "test_pearson_r", "test_spearman_rho", "test_data", "test_label", "ckpt"]
        values = [str(train_run_url), pearson_r, rho, str(test_data), str(dm.label), str(ckpt)]
        if isinstance(self.logger, WandbLogger):
            self.logger.log_table(
                key="Test Table",
                columns=columns,
                data=[values]
            )
        else:
            metrics_dict = dict(zip(columns, values))
            self.logger.log_metrics(metrics_dict)

    # predict_step
    # run model beginning from sequence->embedding->push through model





    