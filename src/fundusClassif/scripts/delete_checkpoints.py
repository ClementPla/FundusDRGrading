import os
import shutil
from pathlib import Path

from nntools.utils import Config

import wandb


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

def clean_all_except_from(from_project):
    api = wandb.Api()
    runs = api.runs(f"liv4d-polytechnique/{from_project}")
    run_ids = [r.name for r in runs]
    list_runs = os.listdir('checkpoints')
    for run in list_runs:
        if run not in run_ids:
            pathdir = Path('checkpoints') / run
            if pathdir.exists():
                shutil.rmtree(pathdir, ignore_errors=True)
                
if __name__=='__main__':
    # clean_from_id(29)
    # clean_all_except_from('Grading-DiabeticRetinopathy-Comparisons-V2')
    pass