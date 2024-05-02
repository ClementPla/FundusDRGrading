from pytorch_lightning.callbacks import StochasticWeightAveraging

from fundusClassif.callbacks.ema import EMACallback
from fundusClassif.callbacks.mixup import MixupCallback


def get_callbacks(training_config):
    callbacks = []
    
    if "ema" in training_config:
        callbacks.append(EMACallback(**training_config["ema"]))
    if "mixup" in training_config:
        callbacks.append(MixupCallback(**training_config["mixup"]))
    if "swa" in training_config:
        callbacks.append(StochasticWeightAveraging(**training_config["swa"]))
    return callbacks