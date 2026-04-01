import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU_IDS'
from pathlib import Path

import torch
import pickle as pkl
from transformers import AutoModelForSequenceClassification
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
    
    with open(f'/{BASE_PATH}/imdb_preprocessed.pkl', 'rb') as f:
        train_x, train_y, valid_x, valid_y, test_x, test_y = pkl.load(f)
    print(f'Train_x:{train_x.shape}, Train_y: {train_y.shape}')
    print(f'Valid_x:{valid_x.shape}, Valid_y: {valid_y.shape}')
    print(f'Test_x:{test_x.shape}, Test_y: {test_y.shape}')

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_dataset = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    batch_size = 64
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    print(model)
