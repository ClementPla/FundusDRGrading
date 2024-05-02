from nntools.utils import Config

from fundusClassif.data.data_factory import precache_datamodule


def main():
    config = Config("configs/config.yaml")
    
    precache_datamodule(config["datasets"], config["data"])

if __name__=='__main__':
    main()