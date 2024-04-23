from typing import Any

import pytorch_lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.data.mixup import Mixup

from fundusClassif.models.model_factory import create_model


class TrainerModule(pytorch_lightning.LightningModule):
    def __init__(self, network_config: dict, training_config: dict) -> None:
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

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "Test accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes),
                "Test recall": torchmetrics.Recall(task="multiclass", num_classes=self.n_classes),
                "Test specificity": torchmetrics.Specificity(task="multiclass", num_classes=self.n_classes),
                "Test precision": torchmetrics.Precision(task="multiclass", num_classes=self.n_classes),
                "Test Quadratic Kappa": torchmetrics.CohenKappa(
                    num_classes=self.n_classes,
                    task="multiclass",
                    weights="quadratic",
                ),
            }
        )
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

    def test_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        logits = self.model(image)
        pred = self.get_pred(logits)
        self.test_metrics.update(pred, gt)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False, sync_dist=True)

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
