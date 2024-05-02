#!/bin/bash 
#
#SBATCH --job-name=FundusDRClassification
#SBATCH --output=logs/log_a6000.txt
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu



#Load le module anaconda 
source /etc/profile.d/modules.sh 
module load anaconda3

source activate ~/.conda/envs/torch18
conda activate torch18

entrypoints/generic.sh