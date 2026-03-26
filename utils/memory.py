import psutil
import os
import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def estimate_dataset_memory(dataset):
    """Estimate memory usage of dataset in MB"""
    if dataset.train_data is None:
        return 0
    
    memory = 0
    for data in [dataset.train_data, dataset.val_data, dataset.test_data]:
        if data:
            X, y = data
            memory += X.nbytes + y.nbytes
    
    return memory / (1024 * 1024)

def estimate_model_memory(model):
    """Estimate model memory usage in MB"""
    # Rough estimation: count parameters and multiply by 4 bytes (float32)
    param_count = sum([np.prod(var.shape) for var in model.trainable_variables])
    memory = param_count * 4 / (1024 * 1024)
    return memory
