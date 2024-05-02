"""
Taken from https://github.com/benihime91/gale/blob/master/gale/collections/callbacks/ema.py and very slghtly modified
"""

import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV3
from torch.optim import Optimizer

_logger = logging.getLogger(__name__)


class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = False,
        warmup_gamma: float = 1.0,
        warmup_power: float = 2 / 3,
        foreach: bool = True,
        exclude_buffers: bool = False,
        infer_with_exponential_moving_average: bool = False,
    ):
        self.ema = None
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.exclude_buffers = exclude_buffers
        self.infer_with_exponential_moving_average = infer_with_exponential_moving_average

    def on_fit_start(self, trainer, pl_module):
        self.ema = ModelEmaV3(
            pl_module,
            decay=self.decay,
            min_decay=self.min_decay,
            device=pl_module.device,
            use_warmup=self.use_warmup,
            warmup_gamma=self.warmup_gamma,
            warmup_power=self.warmup_power,
            foreach=self.foreach,
            exclude_buffers=self.exclude_buffers,
        )

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        "do validation using the stored parameters"
        # save original parameters before replacing with EMA version
        self.store(pl_module.parameters())

        # update the LightningModule with the EMA weights
        # ~ Copy EMA parameters to LightningModule
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        # update the LightningModule with the EMA weights
        if self.infer_with_exponential_moving_average:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            msg = "Model weights replaced with the EMA version."
            self.log_msg(msg)

    @rank_zero_only
    def log_msg(self, msg):
        _logger.log(logging.INFO, msg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}

    def on_load_checkpoint(self, callback_state):
        if self.ema is not None:
            self.ema.module.load_state_dict(callback_state["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
