import gc
import random

import numpy as np
import torch


def check_model_size(logger, model):
    model_num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Current model size: {model_num_params:,}")
    return model_num_params


def cleanup_mem():
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

