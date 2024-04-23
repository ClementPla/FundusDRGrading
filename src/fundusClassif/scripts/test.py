import logging

import pandas as pd
import timm
import torch
from thop import clever_format, profile

logging.basicConfig(level=logging.CRITICAL)

if __name__ == "__main__":
    dummy_data = torch.randn(1, 3, 512, 512).cuda()
    list_models = [
        "efficientnet_b0",
        "efficientnet_b2",
        "mobilenetv3_small_100",
        "mobilevit_s",
        "resnet18",
        "tf_efficientnet_b5",
        "efficientnet_b5",
        "resnet50",
        "seresnet50",
        "seresnext50_32x4d",
        "convnext_small",
        "swinv2_base_window16_256",
        "convnext_base",
        "vit_base_patch16_384",
        "vgg19_bn",
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        "convnext_large",
        "vit_large_patch16_384",
        "vit_base_patch14_dinov2",
        "vit_large_patch14_dinov2",
        "vit_small_patch14_dinov2"
    ]
    results = {"Model":[], "MACs":[], "Params":[]}
    for m in list_models:
        with torch.autocast('cuda'):
            print(f"Running model {m}")
            try:
                model = timm.create_model(m, pretrained=True, num_classes=5, img_size=512)
            except TypeError:
                model = timm.create_model(m, pretrained=True, num_classes=5)
            model = model.cuda()
            try:
                macs, params = profile(model, inputs=(dummy_data,), verbose=False)
                out = model(dummy_data)                
                results["Model"].append(m)
                results["MACs"].append(macs)
                results["Params"].append(params)
                macs, params = clever_format([macs, params], "%.3f")

                print(f"Model {m} has {macs} MACs and {params} parameters")
            except Exception as e:
                print(f"Model {m} failed with error {e}")
    
    df = pd.DataFrame(results)
    df.to_csv("models_macs_params.csv", index=False)