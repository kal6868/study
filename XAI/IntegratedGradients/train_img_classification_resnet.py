import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU_IDS'
from pathlib import Path

import torch
import torch.nn as nn
import pickle as pkl
from torchvision import models
from torch.utils.data import TensorDataset, Subset, DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.train_utils import Train

def main():
    BASE_PATH = Path(__file__).parents[0]
    FILE_NAME = Path(__file__).name
    save_path = BASE_PATH / 'outputs' / f'{FILE_NAME.split(".")[0]}'
    save_path.mkdir(parents=True, exist_ok=True)
    
    log_file = open(Path(save_path / 'log.log'), "w")
    sys.stdout = log_file

    with open(f'/{BASE_PATH}/OxfordIIITPet_preprocessed.pkl', 'rb') as f:
        train_x, train_y, valid_x, valid_y, test_x, test_y = pkl.load(f)
    print(f'Train_x:{train_x.shape}, Train_y: {train_y.shape}')
    print(f'Valid_x:{valid_x.shape}, Valid_y: {valid_y.shape}')
    print(f'Test_x:{test_x.shape}, Test_y: {test_y.shape}')

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    batch_size = 1024
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        # num_workers=4, pin_memory=True, persistent_workers=True
        )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        # num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        # num_workers=num_workers, pin_memory=True, persistent_workers=True
        )

if __name__ == '__main__':
    main()
