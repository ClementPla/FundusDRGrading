import os
import textwrap
from pathlib import Path

import pandas as pd


def main():
    df = pd.read_csv("models_macs_params.csv")
    df.sort_values(by="MACs", inplace=True)
    filepath = Path("entrypoints/generic.sh")
    if filepath.exists():
        os.remove(filepath)

    with open(filepath, "w") as f:
        head = ("""\
            #!/bin/bash
            eval "$(conda shell.bash hook)"
            #Load le module anaconda
            source /etc/profile.d/modules.sh
            module load anaconda3
            source activate ~/.conda/envs/torch18
            conda activate torch18
            
            """)
        
        f.write(textwrap.dedent(head))
        for i, row in df.iterrows():
            f.write(f"python src/fundusClassif/scripts/train.py --model {row['Model']} \n")

    os.chmod(filepath, 0o755)
    
if __name__ == "__main__":
    main()
