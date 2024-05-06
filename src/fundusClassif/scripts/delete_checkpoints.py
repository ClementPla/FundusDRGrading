import os
import shutil
from pathlib import Path

import wandb
from nntools.utils import Config


def clean_all():
    config = Config("configs/config.yaml")
    project_name = config["logger"]["project"]
    
    api = wandb.Api()
    
    runs = api.runs(f"liv4d-polytechnique/{project_name}")
    for r in runs:
        pathdir = Path('checkpoints') / r.name
        if pathdir.exists():
            shutil.rmtree(pathdir, ignore_errors=True)
    

def clean_from_id(run_id: int):    
    list_ckpts = os.listdir('checkpoints')
    for ckpt in list_ckpts:
        id_ckpt = int(ckpt.split('-')[-1])
        if id_ckpt >= run_id:
            pathdir = Path('checkpoints') / ckpt
            if pathdir.exists():
                shutil.rmtree(pathdir, ignore_errors=True)
                
if __name__=='__main__':
    # clean_from_id(29)
    pass