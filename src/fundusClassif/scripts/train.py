import argparse
import os

import torch
from nntools.utils import Config
from pl_bolts.callbacks import ORTCallback
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from fundusClassif.data.data_factory import get_datamodule_from_config
from fundusClassif.my_lightning_module import TrainerModule

torch.set_float32_matmul_precision("medium")


def train(arch: str):
    config = Config("configs/config.yaml")
    config["model"]["architecture"] = arch
    projet_name = config["logger"]["project"]
    if os.environ.get("LOCAL_RANK", None) is None:
        api = wandb.Api()
        try:
            runs = api.runs(f"liv4d-polytechnique/{projet_name}")
            for r in runs:
                if r.config["model/architecture"] == architecture & (r.state == "finished" | r.state == "running"):
                    return
        except ValueError:
            print("Project not existing, starting run")

    seed_everything(1234, workers=True)
    wandb_logger = WandbLogger(**config["logger"], config=config.tracked_params)
    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = wandb_logger.experiment.name

    datamodule = get_datamodule_from_config(config["datasets"], config["data"])
    
    test_dataloader = datamodule.test_dataloader()
    test_datasets_ids = [d.dataset.id for i, d in enumerate(test_dataloader)]
    model = TrainerModule(config["model"], config["training"], test_datasets_ids)

    checkpoint_callback = ModelCheckpoint(
        monitor="Quadratic Kappa",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", os.environ["WANDB_RUN_NAME"]),
    )

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="Quadratic Kappa", patience=25, mode="max"),
            LearningRateMonitor(),
            ORTCallback()
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    architecture = args.model

    train(architecture)
