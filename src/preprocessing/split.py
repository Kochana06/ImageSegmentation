import torch
from torch.utils.data import random_split


# Splitting Dataset into Train, Validation, and Test

def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    return train_dataset, valid_dataset, test_dataset