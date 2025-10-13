import torch
import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

def test(cfg):
    ckpt_path = cfg.get("ckpt_path", None)
    assert ckpt_path is not None, "ckpt_path must be provided in config."

    print(f"Instantiating model {cfg.model._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(
        checkpoint["state_dict"], strict=True
    )
    model.eval()

    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    if cfg.wandb.init:
        wandb.init(
            job_type="evaluation",
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
        logger = None


    trainer = pl.Trainer(logger=logger, enable_checkpointing=False)

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    test_dm: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    print("Starting evaluation!")
    trainer.test(model=model, datamodule=test_dm)

@hydra.main(version_base="1.3", config_path="configs", config_name="test.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    test(cfg)
    return

if __name__ == "__main__":
    submit_run()
