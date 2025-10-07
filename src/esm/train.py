import torch
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from src.esm.model.RCfold import RCFold, AlignBio_DataModule, ESM_Regressor, AlignBio_Dataset

if __name__ == "__main__":
    model = ESM_Regressor(
        input_dim=425
    )
    trainer = pl.Trainer(max_epochs=10)
    # load data into dataloader object
    
    data_dir = Path("/scratch/eac709/overlays/the-protein-engineering-tournament-2023/in_silico_supervised/input/Alpha-Amylase (In Silico_ Supervised)")
    esm_cache_dir = data_dir / Path("esm")

    csv = Path("train.csv")
    dataset = AlignBio_Dataset(data_dir / csv, "expression", esm_cache_dir)
    dataloader = DataLoader(dataset, batch_size=4)

    print(len(dataset))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for i, data in enumerate(dataloader):
        # TODO: develop some sample model architectures
        # later: implement the pytorch lightning training loop
        inputs = data["embed"]
        labels = data["label"]

        # zero gradient
        optimizer.zero_grad()
        # forward pass
        output = model(inputs)

        # calculate loss and backprop gradients
        # Ensure output and labels are float tensors and have matching shapes
        output = output.float().view(-1)
        labels = labels.float().view(-1)
        loss = loss_fn(output, labels)
        loss.backward()

        # update
        optimizer.step()

        print(loss.item())
        pass

    # train_datamodule = AlignBio_DataModule(data_dir, csv, esm_cache_dir)

    # model = ESM_Regressor()
    # trainer.fit(model, train_datamodule)