from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from timm.data.mixup import Mixup


class MixupCallback(Callback):
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        
        
        self.mixup = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmixminmax=cutmix_minmax, prob=prob,
                           switch_prob=switch_prob, mode=mode, correct_lam=correct_lam, label_smoothing=label_smoothing)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        image = batch["image"]
        gt = batch["label"]
        
        image, gt = self.mixup(image, gt)
        batch["image"] = image
        batch["label"] = gt
        
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)