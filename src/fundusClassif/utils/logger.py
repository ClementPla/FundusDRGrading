import os
from typing import Optional

from pytorch_lightning.loggers import WandbLogger

import wandb


def get_wandb_logger(
    project_name,
    tracked_params,
    item_check_if_run_exists: Optional[tuple[str, str]] = None,
):
    name_item, value_item = item_check_if_run_exists
    if item_check_if_run_exists:
        if os.environ.get("LOCAL_RANK", None) is None:
            api = wandb.Api()
            try:
                runs = api.runs(f"liv4d-polytechnique/{project_name}")
                for r in runs:
                    if name_item in r.config.keys():
                        if r.config[name_item] == value_item:
                            exit("Run already exists, exiting")
            except ValueError:
                print("Project not existing, starting run")

    wandb_logger = WandbLogger(project=project_name, config=tracked_params)

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = wandb_logger.experiment.name

    return wandb_logger
