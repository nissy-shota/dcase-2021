# Standard library imports.
import datetime
import os
import sys
from itertools import chain

# Related third party imports.
import joblib
import numpy
import scipy.stats
import torch
import torch.utils.data
from scipy.special import softmax
from torch import optim
from torch.utils.data.dataset import Subset
from torchinfo import summary

# Local application/library specific imports.
import util
from models import MobileNetV2

# Load configuration from YAML file.
CONFIG = util.load_yaml("./config.yaml")

# String constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")








def main():
    
    mode = util.command_line_chk()  # constant: True or False
    if mode is None:
        sys.exit(-1)

    os.makedirs(CONFIG["model_directory"], exist_ok=True)

if __name__ == "__main__":
    main()