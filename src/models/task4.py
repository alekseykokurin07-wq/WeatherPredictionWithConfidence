import torch
import torch.nn as nn
import torch.optim as optim
from random import randint, uniform, gauss
import math

C_train = 60000
alpha = 0.37
error_all = 0.5

X_train = []
y_train = []
for i in range(C_train):
    xi = uniform(0, 3)
    sigmai = uniform(0.1, 2)
    error_now = abs(gauss(0, error_all))

    xi1 = math.exp(alpha * xi)
    sigmai1 = ((alpha * xi1) ** 2 * sigmai ** 2 + error_now ** 2) ** 0.5
    X_train.append([xi, sigmai ** 2])
    y_train.append([xi1, sigmai1 ** 2])

X_train = torch.tensor(X_train).view(-1, 2)
y_train = torch.tensor(y_train).view(-1, 2)

backbone = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())

head_x = nn.Linear(32, 1)
head_sigma = nn.Linear(32, 1)
softplus = nn.Softplus()

params = list(backbone.parameters()) + list(head_x.parameters()) + list(head_sigma.parameters())
opt = optim.Adam(params, lr=10**(-3))

batch = 1024
epochs = 50

for k in range(epochs):
    pos = torch.randperm(X_train.size(0))
    ans = 0
    cnt = 0
    for i in range(0, X_train.size(0), batch):
        ind = pos[i:i + batch]
        x_now = X_train[ind]
        y_now = y_train[ind]

        res = backbone(x_now)
        x = head_x(res)

        sigma = softplus(head_sigma(res))
        x_ok = y_now[:, [0]]
        sigma_ok = y_now[:, [1]]
        loss = ((x - x_ok) ** 2).mean() + ((sigma - sigma_ok) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        ans += loss.item()
        cnt += 1
    print("iter ", k + 1, ", loss = ", ans / cnt, sep='')

C_test = 10000
X_test = []
y_test = []
for i in range(C_test):
    xi = uniform(0, 3)
    sigmai = uniform(0.1, 2)
    error_now = abs(gauss(0, error_all))

    xi1 = math.exp(alpha * xi)
    sigmai1 = ((alpha * xi1) ** 2 * sigmai ** 2 + error_now ** 2) ** 0.5
    X_test.append([xi, sigmai ** 2])
    y_test.append([xi1, sigmai1 ** 2])

X_test = torch.tensor(X_test).view(-1, 2)
y_test = torch.tensor(y_test).view(-1, 2)
mse_error = 0
mse_x = 0
for i in range(C_test):
    h = backbone(X_test[i])
    xi1 = head_x(h)
    sigmai1 = softplus(head_sigma(h))
    mse_error += (y_test[i][1] - sigmai1) ** 2
    mse_x += (y_test[i][0] - xi1) ** 2

mse_error /= C_test
mse_x /= C_test
print("MSE sigma =", mse_error.item())
print("MSE x = ", mse_x.item())

import numpy as np
import matplotlib.pyplot as plt

C_iter = 30
col_test = 1000

need_x = [0.0] * (C_iter + 1)
need_sigma = [0.0] * (C_iter + 1)
res_x  = [0.0] * (C_iter + 1)
res_sigma = [0.0] * (C_iter + 1)

sum_error_x = [0] * (C_iter + 1)
sum_error_sigma = [0] * (C_iter + 1)

for iter1 in range(col_test):
    x0 = uniform(0, 1.5)
    sigma0 = uniform(0.2, 1)

    x = torch.tensor([[x0]], dtype=torch.float32)
    sigma = torch.tensor([[sigma0 ** 2]], dtype=torch.float32)
    x_true = torch.tensor([[x0]], dtype=torch.float32)
    sigma_true = torch.tensor([[sigma0 ** 2]], dtype=torch.float32)

    need_x[0] += x0
    need_sigma[0] += sigma0**2
    res_x[0] += x0
    res_sigma[0] += sigma0**2

    for i in range(C_iter):
        error_now = error_all ** 2
        xii1 = torch.exp(alpha * x_true)
        sigmai1 = ((alpha * xii1)**2) * sigma_true + error_now
        need_x[i + 1] += xii1.item()
        need_sigma[i + 1] += sigmai1.item()
        x_true = xii1
        sigma_true = sigmai1

        now = torch.cat([x, sigma], dim=1)
        h = backbone(now)
        x = softplus(head_x(h))
        sigma = softplus(head_sigma(h))
        res_x[i + 1] += x.item()
        res_sigma[i + 1] += sigma.item()

        sum_error_x[i + 1] += (x_true.item() - x.item()) ** 2
        sum_error_sigma[i + 1] += (sigma_true.item() - sigma.item()) ** 2

for i in range(C_iter + 1):
    need_x[i] /= col_test
    need_sigma[i] /= col_test
    res_x[i] /= col_test
    res_sigma[i] /= col_test

    sum_error_x[i] /= col_test
    sum_error_sigma[i] /= col_test

need_x = np.array(need_x)
need_sigma = np.sqrt(np.array(need_sigma))
res_x = np.array(res_x)
res_sigma = np.sqrt(np.array(res_sigma))

sum_error_x = np.array(sum_error_x)
sum_error_sigma = np.array(sum_error_sigma)
print(sum_error_sigma)