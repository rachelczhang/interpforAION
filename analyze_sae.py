import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sparse_autoencoder import SparseAutoencoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import sys
from save_activations import get_data
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add both directories to sys.path
sys.path.append(os.path.join(current_dir, "AION"))
sys.path.append(os.path.join(current_dir, "4Mclone"))

from aion import AION

# Load model with trained parameters
print("\n=== Loading Model ===")
model = AION.from_pretrained('/mnt/ceph/users/polymathic/aion/dec24/base')
print(f"Model loaded from pretrained weights")
model.freeze_encoder()
model.freeze_decoder()
model = model.cuda().eval()  
print(f"Model moved to CUDA and set to eval mode")
print(f"Model architecture:\n{model}")

# Load tokenized training data
print("\n=== Setting Up Data ===")
class Args:
    """Simple class to hold arguments needed for setup functions."""
    def __init__(self):
        self.data_config = "4Mclone/cfgs/default/mmoma/data/mmu/rusty_legacysurvey_desi_sdss_hsc_eval.yaml"
        self.text_tokenizer_path = "4Mclone/fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json"
        self.input_size = 224
        self.patch_size = 16
        self.num_input_tokens = 128
        self.num_target_tokens = 128
        self.min_input_tokens = None
        self.min_target_tokens = None
        self.batch_size = 128
        self.epoch_size = 100
        self.num_workers = 1
        self.num_tasks = 1
        self.pin_mem = True
        self.dist_eval = False
        self.fixed_eval = False
        self.fixed_eval_input_tokens = None
        self.fixed_eval_target_tokens = None
        self.in_domains = None
        self.out_domains = None
        self.all_domains = None

args = Args()
data_loader_train = get_data(args)

