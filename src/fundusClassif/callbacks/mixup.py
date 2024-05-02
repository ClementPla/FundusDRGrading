"""
We only use Mixup/CutMix for classification tasks. For regression, we could try to implement:
C-Mixup: Improving Generalization in Regression
https://arxiv.org/pdf/2210.05775

Repository:
https://github.com/huaxiuyao/C-Mixup
"""
from typing import Any

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from timm.data.mixup import Mixup

from fundusClassif.my_lightning_module import TrainerModule


class MixupCallback(Callback):
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        
        
        self.mixup = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=cutmix_minmax, prob=prob,
                           switch_prob=switch_prob, mode=mode, correct_lam=correct_lam, label_smoothing=label_smoothing, 
                           num_classes=num_classes)
    
        self._activated = True

    def on_fit_start(self, trainer: Trainer, pl_module: TrainerModule) -> None:
        self._activated = not pl_module.as_regression
        if not self._activated:
            rank_zero_warn("Mixup is not activated for regression tasks")
    
    def on_train_batch_start(self, trainer: Trainer, pl_module: TrainerModule, batch: Any, batch_idx: int) -> None:
        if not self._activated:
            pass
        else:
            image = batch["image"]
            gt = batch["label"]
            
            image, gt = self.mixup(image, gt)
            batch["image"] = image
            batch["label"] = gt
            
            return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)