import torch
import pytorch_lightning as pl
import wandb
from torch import nn
from torchmetrics.functional import spearman_corrcoef

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
        self.logger.log_table(key="Test Table", 
                              columns=["test_pearson_r", "test_spearman_rho"], 
                              data=[[pearson_r, rho]] )


    # predict_step
    # run model beginning from sequence->embedding->push through model





    