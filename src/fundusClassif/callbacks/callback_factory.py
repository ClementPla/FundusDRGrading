from fundusClassif.callbacks.ema import EMACallback
from fundusClassif.callbacks.mixup import MixupCallback


def get_callbacks(training_config):
    callbacks = []
    
    if "ema" in training_config:
        callbacks.append(EMACallback(**training_config["ema"]))
    if "mixup" in training_config:
        callbacks.append(MixupCallback(**training_config["mixup"]))
        
    return callbacks