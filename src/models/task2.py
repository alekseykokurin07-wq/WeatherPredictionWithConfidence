from sklearn.linear_model import LinearRegression
from random import randint, uniform, gauss
C_train = 1000000
Inf = 10**2

res1 = []
steps = []
alpha = uniform(1, Inf)
G = uniform(1, Inf)

for sigma in range(1, 1001, 100):
    X_train = []
    y_train = []
    
    for i in range(C_train):
        x0 = uniform(1, Inf)
        e = gauss(0, sigma)
        
        X_train.append([x0])
        y_train.append(x0 * alpha + e)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    C_test = 1000
    
    X_test = []
    y_test = []
    for i in range(C_test):
        x0 = uniform(1, Inf)
        e = gauss(0, sigma)
        
        X_test.append([x0])
        y_test.append(x0 * alpha + e)
    y_pred = model.predict(X_test)
    
    score = model.score(X_test, y_test)
    print(alpha, " alpha")
    
    res = []
    for i in range(C_test):
        res.append(y_pred[i] / X_test[i][0])
        y_pred = model.predict(X_test)
    alpha_pred = sum(res) / len(res)
    
    sigma_pred = 0
    for i in range(C_test):
        sigma_pred += (y_test[i] - y_pred[i]) ** 2
    sigma_pred = ((1 / C_test) * sigma_pred) ** 0.5

    print(alpha_pred, " alpha_pred\n\n")
    
    print(sigma, " sigma")
    print(sigma_pred, " sigma_pred\n")
    
    print(score * 100, " score")
    res1.append(score * 100)
    steps.append(sigma)

import torch

Train = 10000
alpha_ok = uniform(1, 5)
sigma_ok = uniform(0, 1)
x_was = []
y_was = []

for i in range(Train):
    x = uniform(-10, 10)
    e = gauss(0, sigma_ok)
    y = alpha_ok * x + e
    
    x_was.append(x)
    y_was.append(y)
x = torch.tensor(x_was).view(-1, 1)
y = torch.tensor(y_was).view(-1, 1)
alpha = torch.tensor(1.0, requires_grad=True)
sigma_log = torch.tensor(0.0, requires_grad=True)

optimizer = torch.optim.Adam([alpha, sigma_log], lr=0.001)

CntIter = 5000
for iter1 in range(CntIter):
    optimizer.zero_grad()
    sigma = torch.exp(sigma_log)
    loss = ((y - (x * alpha))**2 / (2 * sigma**2) + torch.log(sigma)).mean()
    loss.backward()
    optimizer.step()
sigma = torch.exp(sigma_log)

print("alpha_true ", alpha_ok)
print("alpha_pred ", alpha.item(), "\n")
print("sigma_true ", sigma_ok)
print("sigma_pred ", sigma.item())

alpha1 = alpha.detach().item()
sigma1 = torch.exp(sigma_log).detach().item()

C_test = 2000
x_test_was = []
y_test_was = []

for _ in range(C_test):
    x = uniform(-10, 10)
    eps = gauss(0, sigma_ok)
    y = alpha_ok * x + eps
    x_test_was.append(x)
    y_test_was.append(y)

x_test_now = np.array(x_test_was)
y_test_now = np.array(y_test_was)