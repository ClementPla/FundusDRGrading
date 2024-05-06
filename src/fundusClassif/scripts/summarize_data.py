from nntools.utils import Config

from fundusClassif.data.data_factory import get_datamodule_from_config


def main():
    config = Config("configs/config.yaml")
    datamodules = get_datamodule_from_config(config["datasets"], config["data"])
    
    print('TRAINING')
    train = datamodules.train
    for d in train.datasets:
        print(f"Dataset: {d.id} has {len(d)} samples")
    
    print('VALIDATION')
    val = datamodules.val        
    for d in val.datasets:
        print(f"Dataset: {d.id} has {len(d)} samples")
    
    print('TEST')    
    test = datamodules.test
    for d in test:
        print(f"Dataset: {d.id} has {len(d)} samples")
    
if __name__=='__main__':
    main()