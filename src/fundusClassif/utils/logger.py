import os
from typing import Optional

from pytorch_lightning.loggers import WandbLogger

import wandb


def get_wandb_logger(project_name, tracked_params, item_check_if_run_exists: Optional[tuple[str, str]] = None):
    if item_check_if_run_exists:
        if os.environ.get("LOCAL_RANK", None) is None:
            api = wandb.Api()
            try:
                runs = api.runs(f"liv4d-polytechnique/{project_name}")
                for r in runs:
                    if r.config[item_check_if_run_exists[0]] == item_check_if_run_exists[1] & (
                        r.state == "finished" | r.state == "running"
                    ):
                        exit("Run already exists, exiting")
            except ValueError:
                print("Project not existing, starting run")

    wandb_logger = WandbLogger(project=project_name, config=tracked_params)

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = wandb_logger.experiment.name

    return wandb_logger
