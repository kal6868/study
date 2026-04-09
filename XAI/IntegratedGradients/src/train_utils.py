import os
import subprocess

import time
import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def Test(save_path: "Pathlib.Path",
        test_dataloader: torch.utils.data.DataLoader, 
        device: torch.device,
        gpu_parallel:bool,
        att_type: list,
        ):
    model = torch.load(save_path / "best_model.pt", map_location=device)
    if gpu_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    start_time = time.time()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(
                device,
                dtype=torch.float32 if x.is_floating_point() else torch.long,
                non_blocking = False)
            y = y.to(device, dtype=torch.long,non_blocking = False)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pb = model(x)
            else:
                pb = model(x)

            y_pred.extend(torch.argmax(pb if isinstance(pb, torch.Tensor) else pb.logits, dim=1).detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    test_f1score = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    print("test_f1 = {:10.8f}  time = {:4.2f}".format(test_f1score, time.time() - start_time), flush = True)
    print(f"accuracy_score: {round(float(accuracy_score(y_true, y_pred)) * 100, 2)}", flush = True)
    print(f"precision_score: {round(float(precision_score(y_true, y_pred, average='micro', zero_division=0)) * 100, 2)}", flush = True)
    print(f"recall_score: {round(float(recall_score(y_true, y_pred, average='micro', zero_division=0)) * 100, 2)}", flush = True)
    print(classification_report(y_true, y_pred, zero_division=0), flush = True)

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, xticklabels=att_type, yticklabels=att_type, annot=True, fmt='d')
    plt.title('Test Confusion Matrix F1_Score: {:0.8f}'.format(round(test_f1score, 5)))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    plt.savefig(save_path / "best_model.jpg")

    gc.collect()
    torch.cuda.empty_cache()

    return 0


def Train(
    epochs: int,
    save_path: "Pathlib.Path",
    patience: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    gpu_parallel:bool,
    att_type: list,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    ):

    save_path.mkdir(parents=True, exist_ok=True)
    train_start_time = time.time()
    print("Starting Training!\n", flush = True)
    patient, before_loss = 0, np.inf
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, valid_loss = 0, 0
        train_pred, train_real, valid_pred, valid_real = [], [], [], []

        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad(set_to_none = True)

            x = x.to(
                device,
                dtype=torch.float32 if x.is_floating_point() else torch.long,
                non_blocking = False)
            y = y.to(device, dtype=torch.long, non_blocking = False)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pb = model(x)
                    loss = loss_fn(pb if isinstance(pb, torch.Tensor) else pb.logits, y)
            else:
                pb = model(x)
                loss = loss_fn(pb if isinstance(pb, torch.Tensor) else pb.logits, y)

            train_loss += loss.item()
            if scaler is not None:
                scaler.scale(loss).backward()
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_pred.extend(torch.argmax(pb if isinstance(pb, torch.Tensor) else pb.logits, dim=1).detach().cpu().tolist())
            train_real.extend(y.detach().cpu().tolist())

        model.eval()
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(
                    device,
                    dtype=torch.float32 if x.is_floating_point() else torch.long,
                    non_blocking = False)
                y = y.to(device, dtype=torch.long, non_blocking = False)

                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        pb = model(x)
                        loss = loss_fn(pb if isinstance(pb, torch.Tensor) else pb.logits, y)
                else:
                    pb = model(x)
                    loss = loss_fn(pb if isinstance(pb, torch.Tensor) else pb.logits, y)

                valid_loss += loss.item()
                valid_pred.extend(torch.argmax(pb if isinstance(pb, torch.Tensor) else pb.logits, dim=1).detach().cpu().tolist())
                valid_real.extend(y.detach().cpu().tolist())

        train_loss /= len(train_real)    # to average loss
        valid_loss /= len(valid_real)    # to average loss

        train_f1score = f1_score(train_real, train_pred, average="micro", zero_division=0)
        valid_f1score = f1_score(valid_real, valid_pred, average="micro", zero_division=0)

        if scheduler is not None:
            curr_lr = scheduler.get_lr()[0]
        else:
            curr_lr = optimizer.param_groups[0]['lr']

        print(f"epoch = {epoch:4d}", flush = True)
        print(
             f"    train_f1 = {train_f1score:10.8f}  valid_f1 = {valid_f1score:10.8f}  train_loss = {train_loss:10.8f}  valid_loss = {valid_loss:10.8f}  lr = {curr_lr:10.8f}  req_time = {(time.time() - start_time):4.2f}sec",
             flush = True)
        print(f"    {' | '.join(get_gpu_usage_per_epoch())}\n", flush = True)

        if not save_path.exists():
            save_path.mkdir(parents = True, exist_ok = True)

        if valid_loss < before_loss:    # early stopping
            torch.save(model.module if hasattr(model, "module") else model, save_path / "best_model.pt")
            before_loss = valid_loss
            patient = 0
        else:
            patient += 1
            if patient >= patience:
                print("Train Early Stopped!", flush = True)
                break

        torch.save(model.module if hasattr(model, "module") else model, save_path / "last_model.pt")

        if (scheduler is not None) and (scaler.get_scale() >= prev_scale):
            scheduler.step()

        if epoch % 20 == 0:
            Test(
                save_path = save_path, test_dataloader = test_dataloader,
                device = device, gpu_parallel = gpu_parallel, att_type = att_type,
                )
    Test(
        save_path = save_path, test_dataloader = test_dataloader,
        device = device, gpu_parallel = gpu_parallel, att_type = att_type,
        )
    print(f"Training Done! {round((time.time() - train_start_time) / 60, 2)} min required.\n", flush = True)

    gc.collect()
    torch.cuda.empty_cache()

    return 0


class Timer:
    def __init__(self, message=None):
        self.start_time = time.time()
        self.message = message

    def end(self):
        end_time = time.time()
        elapsed = end_time - self.start_time  
        
        if self.message is not None:
            final_message = f'{self.message} took {int(elapsed//60)}min {round(elapsed%60, 2)}sec'
            return final_message

        else:
            return int(elapsed//60), round(elapsed%60, 2)


def set_logger(save_path: Path, print_log: bool = False):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )

    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path), exist_ok = True)

    parents_dir = save_path.parent
    if not parents_dir.exists():
        parents_dir.mkdir(parents = True, exist_ok = True)

    if not logger.handlers:
        file_handler = logging.FileHandler(save_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if print_log:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def get_gpu_usage_per_epoch():
    current_gpus = set(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    current_gpus.discard("")

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    gpu_usage = []
    for line in result.stdout.strip().splitlines():
        gpu_idx, mem_used, mem_total = [x.strip() for x in line.split(",")]

        if gpu_idx in current_gpus:
            gpu_usage.append(f"GPU {gpu_idx}: {mem_used} MiB / {mem_total} MiB")

    return gpu_usage
