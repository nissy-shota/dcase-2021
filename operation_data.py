from itertools import chain

import torch
import torch.utils.data
import numpy

import util

CONFIG = util.load_yaml("./config.yaml")
class DCASE2021_dataset(torch.utils.data.Dataset):
    '''
    data set class
    '''
    def __init__(self, unique_section_names, target_dir, mode):
        super().__init__()




