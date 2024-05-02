from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.data.mixup import Mixup

from fundusClassif.models.model_factory import create_model


class TrainerModule(pl.LightningModule):
    def __init__(self, network_config: dict, training_config: dict, test_datasets_ids: list) -> None:
        super().__init__()

        self.as_regression = training_config.get("as_regression", False)
        self.n_classes = network_config["num_classes"]

        if self.as_regression:
            network_config["num_classes"] = 1

        self.network_config = network_config
        self.training_config = training_config
        print(network_config)

        self.model = create_model(**network_config)

        if self.as_regression:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.metrics = torchmetrics.MetricCollection(
            {
                "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes),
                "Quadratic Kappa": torchmetrics.CohenKappa(
                    num_classes=self.n_classes,
                    task="multiclass",
                    weights="quadratic",
                ),
            }
        )
        test_metrics = []
        for d_id in test_datasets_ids:
            test_metrics.append(
                torchmetrics.MetricCollection(
                    {
                        "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes),
                        "Recall": torchmetrics.Recall(task="multiclass", num_classes=self.n_classes),
                        "Specificity": torchmetrics.Specificity(task="multiclass", num_classes=self.n_classes),
                        "Precision": torchmetrics.Precision(task="multiclass", num_classes=self.n_classes),
                        "Quadratic Kappa": torchmetrics.CohenKappa(
                            num_classes=self.n_classes,
                            task="multiclass",
                            weights="quadratic",
                        ),
                    },
                    postfix=f"_{d_id}",
                )
            )
        self.test_metrics = nn.ModuleList(test_metrics)
        
        mixup_config = training_config.get("mixup", None)
        if mixup_config is not None and any(
            [
                mixup_config["mixup_alpha"] > 0,
                mixup_config["cutmix_alpha"] > 0,
                mixup_config["cutmix_minmax"] is not None,
            ]
        ):
            self.mixup = Mixup(**mixup_config)
        else:
            self.mixup = None

    def training_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        if self.mixup is not None:
            image, gt = self.mixup(image, gt)

        logits = self.model(image)
        loss = self.get_loss(logits, gt)
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def get_pred(self, logits):
        if self.as_regression:
            return self.regression_to_prediction(logits)
        else:
            return torch.argmax(logits, dim=1)

    def regression_to_prediction(self, logits):
        logits = torch.round(logits).clamp(0, self.n_classes - 1)
        logits = logits.long().view(-1)
        return logits

    def get_loss(self, logits, gt):
        if self.as_regression:
            return self.loss(logits.flatten(), gt.float())
        else:
            return self.loss(logits, gt.long())

    def validation_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        logits = self.model(image)
        loss = self.get_loss(logits, gt)
        pred = self.get_pred(logits)
        self.metrics.update(pred, gt)
        self.log_dict(self.metrics, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return {"pred": pred, "gt": gt}

    def test_step(self, data, batch_index, dataloader_idx=0) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        logits = self.model(image)
        pred = self.get_pred(logits)
        test_metrics = self.test_metrics[dataloader_idx]
        test_metrics.update(pred, gt)
        self.log_dict(test_metrics, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.training_config["lr"], **self.training_config["optimizer"]
        )
        return [optimizer], [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode="min"),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
