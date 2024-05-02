#!/bin/bash

sbatch entrypoints/a6000.sh
sleep 5
sbatch entrypoints/a6000.sh
sleep 5
sbatch entrypoints/liv4dgpu.sh
sleep 5
sbatch entrypoints/3090.sh
