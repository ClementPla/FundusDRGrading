import logging

import pandas as pd
import timm
import torch
from nntools.utils import Config
from thop import clever_format, profile

import wandb

logging.basicConfig(level=logging.CRITICAL)

if __name__ == "__main__":
    config = Config("configs/config.yaml")
    dummy_data = torch.randn(1, 3, 512, 512).cuda()
    api = wandb.Api()
    project_name = config["logger"]["project"]
    runs = api.runs(f"liv4d-polytechnique/{project_name}")
    list_models = [r.config["model/architecture"] for r in list(runs)]
    
    results = {"Model": [], "MACs": [], "Params": []}
    for m in list_models:
        with torch.autocast("cuda"):
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
