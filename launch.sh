#!/bin/bash

python src/fundusClassif/scripts/train.py --model vit_base_patch14_dinov2
python src/fundusClassif/scripts/train.py --model vit_large_patch14_dinov2
python src/fundusClassif/scripts/train.py --model vit_small_patch14_dinov2
