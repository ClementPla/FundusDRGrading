#!/bin/bash
eval "$(conda shell.bash hook)"
#Load le module anaconda
source /etc/profile.d/modules.sh
module load anaconda3
source activate ~/.conda/envs/torch18
conda activate torch18

python src/fundusClassif/scripts/train.py --model mobilenetv3_small_100 
python src/fundusClassif/scripts/train.py --model efficientnet_b0 
python src/fundusClassif/scripts/train.py --model efficientnet_b2 
python src/fundusClassif/scripts/train.py --model mobilevit_s 
python src/fundusClassif/scripts/train.py --model resnet18 
python src/fundusClassif/scripts/train.py --model tf_efficientnet_b5 
python src/fundusClassif/scripts/train.py --model densenet121.tv_in1k 
python src/fundusClassif/scripts/train.py --model inception_v3.gluon_in1k 
python src/fundusClassif/scripts/train.py --model resnet50 
python src/fundusClassif/scripts/train.py --model seresnet50 
python src/fundusClassif/scripts/train.py --model resnext50_32x4d.tv2_in1k 
python src/fundusClassif/scripts/train.py --model seresnext50_32x4d 
python src/fundusClassif/scripts/train.py --model vit_small_patch14_dinov2 
python src/fundusClassif/scripts/train.py --model convnext_small 
python src/fundusClassif/scripts/train.py --model swinv2_base_window16_256 
python src/fundusClassif/scripts/train.py --model convnext_base 
python src/fundusClassif/scripts/train.py --model vit_base_patch16_384 
python src/fundusClassif/scripts/train.py --model vit_base_patch14_dinov2 
python src/fundusClassif/scripts/train.py --model vit_base_patch14_reg4_dinov2.lvd142m 
python src/fundusClassif/scripts/train.py --model vgg19_bn 
python src/fundusClassif/scripts/train.py --model swinv2_large_window12to16_192to256.ms_in22k_ft_in1k 
python src/fundusClassif/scripts/train.py --model convnext_large 
python src/fundusClassif/scripts/train.py --model vit_large_patch16_384 
python src/fundusClassif/scripts/train.py --model vit_large_patch14_dinov2 
python src/fundusClassif/scripts/train.py --model vit_large_patch14_reg4_dinov2.lvd142m 
