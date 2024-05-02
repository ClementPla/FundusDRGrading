from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2
from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.datamodules import CLASSIF_PATHS, Task, register_paths
from fundus_data_toolkit.datamodules.classification import (
    AptosDataModule,
    DDRDataModule,
    EyePACSDataModule,
    IDRiDDataModule,
)
from fundus_data_toolkit.datamodules.utils import merge_existing_datamodules
from tqdm import tqdm


class FundusDataset(Enum):
    IDRID: str = "IDRID"
    EYEPACS: str = "EYEPACS"
    APTOS: str = "APTOS"
    DDR: str = "DDR"


def setup_data_toolkit(paths: Dict[str, Path]):
    register_paths(paths, Task.CLASSIFICATION)


def setup_data_from_config(datasets: Dict[str, str]):
    paths = {}
    for dataset in datasets:
        paths[dataset] = Path(datasets[dataset])
        assert paths[dataset].exists(), f"Path {paths[dataset]} does not exist"
    setup_data_toolkit(paths)


def get_datamodule(datasets: List[str], dataset_args):
    all_datamodules = []
    for d in datasets:
        match FundusDataset(d):
            case FundusDataset.IDRID:
                all_datamodules.append(
                    IDRiDDataModule(
                        CLASSIF_PATHS.IDRID, precise_autocrop=True, 
                        data_augmentation_type=DAType.HEAVY,
                        flag=cv2.IMREAD_COLOR, **dataset_args
                    ).setup_all()
                )
            case FundusDataset.EYEPACS:
                all_datamodules.append(
                    EyePACSDataModule(
                        CLASSIF_PATHS.EYEPACS, precise_autocrop=True, flag=cv2.IMREAD_COLOR, 
                        data_augmentation_type=DAType.HEAVY,
                        **dataset_args
                    ).setup_all()
                )
            case FundusDataset.APTOS:
                all_datamodules.append(
                    AptosDataModule(
                        CLASSIF_PATHS.APTOS, precise_autocrop=True, flag=cv2.IMREAD_COLOR, 
                        data_augmentation_type=DAType.HEAVY,
                        **dataset_args
                    ).setup_all()
                )
            case FundusDataset.DDR:
                all_datamodules.append(
                    DDRDataModule(
                        CLASSIF_PATHS.DDR, precise_autocrop=True, flag=cv2.IMREAD_COLOR, 
                        data_augmentation_type=DAType.HEAVY,
                        **dataset_args
                    ).setup_all()
                )
    return merge_existing_datamodules(all_datamodules)


def get_datamodule_from_config(config: Dict[str, str], dataset_args):
    setup_data_from_config(config)
    datasets = [FundusDataset(d.upper()) for d in config]
    return get_datamodule(datasets, dataset_args)


def precache_datamodule(config: Dict[str, str], dataset_args):
    datamodule = get_datamodule_from_config(config, dataset_args)
    train_dataloader = datamodule.train_dataloader()
    for _ in tqdm(train_dataloader, total=len(train_dataloader)):
        pass
    if datamodule.val:
        val_dataloader = datamodule.val_dataloader()
        for _ in tqdm(val_dataloader, total=len(val_dataloader)):
            pass
    if datamodule.test:
        test_dataloader = datamodule.test_dataloader()
        if not isinstance(test_dataloader, list):
            test_dataloader = [test_dataloader]

        for dl in test_dataloader:
            for _ in tqdm(dl, total=len(dl)):
                pass
