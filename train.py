from torcheeg.models import EEGNet
import torch
import torch.nn as nn
from torcheeg.models import CCNN
from torcheeg.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torcheeg.model_selection import KFoldPerSubject
import os
import time
import logging
import random
import numpy as np

os.makedirs('./examples_vanilla_torch/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./examples_vanilla_torch/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

k_fold = KFoldPerSubject(n_splits=10,
                         split_path='./examples_vanilla_torch/split',
                         shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# training process
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            # logger.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss

# validation process
def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    # logger.info(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

    return correct, loss

loss_fn = nn.CrossEntropyLoss()
batch_size = 16 # Hyperparam?

test_accs = []
test_losses = []

for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    # initialize model
    model = EEGNet(chunk_size=32,
               num_electrodes=32,
               dropout=0.25, # Hyperparam? -> Consider changing it based on results
               kernel_1=8, # Hyperparam 8, 16, 32
               kernel_2=2, # Hyperparam 2, 4
               F1=8, # Default 8, Hyperparam?
               F2=16, # Default 16, Hyperparam?
               D=2, # Default 2, Hyperparam?
               num_classes=4).to(device)
    
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)  # official: weight_decay=5e-1
    
    # split train and val
    train_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        split_path=f'./examples_vanilla_torch/split{i}',
        shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    epochs = 50
    best_val_acc = 0.0
    for t in range(epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(val_loader, model, loss_fn)
        # save the best model based on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f'./examples_vanilla_torch/model{i}.pt')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the best model to test on test set
    model.load_state_dict(torch.load(f'./examples_vanilla_torch/model{i}.pt'))
    test_acc, test_loss = valid(test_loader, model, loss_fn)

    # log the test result
    # logger.info(f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}")

    test_accs.append(test_acc)
    test_losses.append(test_loss)

# log the average test result on cross-validation datasets
# logger.info(f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}")

