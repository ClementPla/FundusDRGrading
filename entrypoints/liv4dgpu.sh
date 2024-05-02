#!/bin/bash 
#
#SBATCH --job-name=FundusDRClassification
#SBATCH --output=logs/log_liv4dgpu.txt
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --partition=liv4d


entrypoints/generic.sh