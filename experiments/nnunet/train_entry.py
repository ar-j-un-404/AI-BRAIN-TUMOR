import random
import numpy as np
import torch
from nnunetv2.run.run_training import run_training_entry
if __name__=='__main__':
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    run_training_entry()
