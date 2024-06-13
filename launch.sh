#!/bin/bash

python src/fundusClassif/scripts/generate_sh_script.py

sbatch entrypoints/a6000.sh
sleep 10
sbatch entrypoints/a6000.sh
sleep 10
sbatch entrypoints/liv4dgpu.sh
sleep 10
sbatch entrypoints/3090.sh
