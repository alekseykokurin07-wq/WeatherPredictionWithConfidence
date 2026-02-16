import os
import sys

import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class MovingMNIST(Dataset):
    def __init__(self, path):
        data = np.load(path)
        data = torch.from_numpy(data).float() / 255
        self.videos = data.permute(1, 0, 2, 3).unsqueeze(2)
        self.all_cadr = self.videos.shape[1]
        self.cnt_one = self.all_cadr - 3

    def __len__(self):
        return self.videos.shape[0] * self.cnt_one

    def __getitem__(self, pos):
        vid = pos // self.cnt_one
        ind = pos % self.cnt_one

        video = self.videos[vid]
        x = video[ind:ind+3]
        y = video[ind+3]
        return x, y


def train_val_loaders(path, batch_size=64, val_ratio=0.1, num_workers=0, seed=42):
    ds = MovingMNIST(path)

    n_val = int(len(ds) * val_ratio)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        x = x.squeeze(2)
        return self.net(x)


def training_epoch(model, loader, criterion, device, lr):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        #optimizer.zero_grad(set_to_none=True)

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad
        #optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(1, n_batches)}


def validation_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / max(1, n_batches)}


device = torch.device("cpu")

npy_path = os.path.join("./data", "mnist_test_seq.npy")
train_loader, val_loader = train_val_loaders(npy_path, batch_size=64, val_ratio=0.1, num_workers=0)

model = Model().to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.MSELoss()

for epoch in range(1, 6):
    tr = training_epoch(model, train_loader, criterion, device, 1e-2)
    va = validation_epoch(model, val_loader, criterion, device)
    print(f"epoch {epoch}: train {tr} val {va}")
