from typing import Dict

from torchmetrics import Accuracy, CohenKappa, Metric, Precision, Recall, Specificity

from fundusClassif.metrics.binary import (
    CustomBinaryMetric,
)


def get_metric(num_classes:int=5) -> Dict[str, Metric]:
    return {
        "Accuracy": Accuracy(task="multiclass", num_classes=num_classes),
        "Recall": Recall(task="multiclass", num_classes=num_classes),
        "Specificity": Specificity(task="multiclass", num_classes=num_classes),
        "Precision": Precision(task="multiclass", num_classes=num_classes),
        "Quadratic Kappa": CohenKappa(num_classes=num_classes, task="multiclass", weights="quadratic"),
        "Binary Accuracy": CustomBinaryMetric(metric=Accuracy, index_class=1),
        "Binary Recall": CustomBinaryMetric(metric=Recall, index_class=1),
        "Binary Specificity": CustomBinaryMetric(metric=Specificity, index_class=1),
        "Binary Precision": CustomBinaryMetric(metric=Precision, index_class=1),
        "Binary Kappa": CustomBinaryMetric(metric=CohenKappa, index_class=1)
    }
    