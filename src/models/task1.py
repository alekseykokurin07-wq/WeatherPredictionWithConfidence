from sklearn.linear_model import LinearRegression
from random import randint, uniform
C_train = 1000
Inf = 10**8

alpha = uniform(1, Inf)
G = uniform(1, Inf)
X_train = []
y_train = []

def NextX(x):
    return x * alpha

for i in range(C_train):
    x0 = uniform(1, Inf)
    X_train.append([x0])
    y_train.append(NextX(x0))
model = LinearRegression()
model.fit(X_train, y_train)

C_test = 200

X_test = []
y_test = []
for i in range(C_test):
    x0 = uniform(1, Inf)
    X_test.append([x0])
    y_test.append(NextX(x0))

y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
print(alpha, " alpha")

res = []
for i in range(C_test):
    res.append(y_pred[i] / X_test[i][0])
alpha_pred = sum(res) / len(res)
print(alpha_pred, " alpha_pred")

print(score * 100, " score")

import matplotlib.pyplot as plt
res = [x[0] for x in X_test]
plt.figure()
plt.scatter(res, y_test, label="Правильный ответы", alpha=0.5)
plt.scatter(res, y_pred, label="Предсказания модели", alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()