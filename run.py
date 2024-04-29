import sys
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" #set here before import torch

from utils.config import SHINEConfig
from utils.mapper import Mapper

def run():

    config = SHINEConfig()

    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_incre.py xxx/xxx_config.yaml"
        )

    mapper = Mapper(config)
    mapper.mapping()

if __name__ == "__main__":
    run()