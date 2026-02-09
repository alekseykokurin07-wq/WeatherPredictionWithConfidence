from sklearn.linear_model import LinearRegression
from random import randint, uniform, gauss

C_train = 1000000

alpha = uniform(1.1, 1.2)
X_train = []
y_train = []

for i in range(C_train):
    xi = randint(1, 100)
    sigmai = uniform(0.1, 3)
    new_error = uniform(0.1, 0.3)
    xi1 = gauss(xi, sigmai) * alpha + gauss(0, new_error)
    sigmai1 = ((sigmai ** 2) * (alpha ** 2) + (new_error ** 2)) ** 0.5
    X_train.append([xi, sigmai**2])
    y_train.append([xi1, sigmai1**2])


model = LinearRegression()
model.fit(X_train, y_train)

C_test = 1000

X_test = []
y_test = []

for i in range(C_test):
    xi = randint(1, 100)
    sigmai = uniform(0.1, 3)
    new_error = uniform(0.1, 0.3)
    xi1 = gauss(xi, sigmai) * alpha + gauss(0, new_error)
    sigmai1 = ((sigmai ** 2) * (alpha ** 2) + (new_error ** 2)) ** 0.5
    X_test.append([xi, sigmai**2])
    y_test.append([xi1, sigmai1**2])


y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
print(score * 100, " score\n")
print(alpha, " alpha")
print(model.coef_[0], " alpha_pred")
