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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 37)
    print(model)

    epochs = 200
    patience = 20
    gpu_parallel = True
    lr = {
        "max_lr": 1e-2,
        "momentum": 9e-1,
        "weight_decay": 1e-4,
        "T_max": 50
    }

    model.to(device)
    if gpu_parallel:
        model = torch.nn.DataParallel(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr['max_lr'],
    momentum=lr['momentum'],
    weight_decay=lr['weight_decay']
    )   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lr["T_max"]
    )
    att_type = ['Abyssinian', 'American Bulldog','American Pit Bull Terrier', 'Basset Hound',
        'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair','Chihuahua',
        'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired',
        'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger',
        'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian',
        'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier',
        'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier',
        'Yorkshire Terrier']
    
    Train(
        epochs = epochs, save_path = save_path, patience = patience,
        model = model, loss_fn = loss_fn, optimizer = optimizer, scheduler = scheduler,
        gpu_parallel = gpu_parallel, device = device, att_type = att_type,
        train_dataloader = train_loader, valid_dataloader = valid_loader, test_dataloader = test_loader
        )

if __name__ == '__main__':
    main()
