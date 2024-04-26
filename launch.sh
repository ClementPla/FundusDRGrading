#!/bin/bash

python src/fundusClassif/scripts/train.py --model vit_base_patch14_reg4_dinov2.lvd142m
python src/fundusClassif/scripts/train.py --model vit_large_patch14_reg4_dinov2.lvd142m
