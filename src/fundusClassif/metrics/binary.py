from typing import Any

from torch import Tensor
from torchmetrics import Metric


class CustomBinaryMetric(Metric):
    def __init__(self, metric, index_class=1):
        super().__init__()
        self.metric = metric(task='binary')
        self.index_class = index_class
    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = (preds > self.index_class).to(preds)
        target = (target > self.index_class).to(target)
        self.metric.update(preds, target)
    def compute(self) -> Any:
        return self.metric.compute()
