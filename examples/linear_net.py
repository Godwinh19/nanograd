import pandas as pd
from nanograd.nn import MLP
import matplotlib.pyplot as plt

data = pd.read_csv("data/iris.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

for i in range(X.shape[1]):
    pass  # X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))

net = MLP(4, [8, 4, 4, 2, 2, 1])
lr = 0.0001
losses = []
for e in range(100):
    # forward pass
    yhat = [net(x) for x in X]
    loss = sum((yout - ytrue) ** 2 for yout, ytrue in zip(yhat, y))
    losses.append(loss.data)

    # backward pass
    # zero grad
    net.zero_grad()
    loss.backward()

    # update parameters
    for p in net.parameters():
        p.data += - lr * p.grad

    if e % 5 == 0: print("Epoch ", e, loss.data)

plt.plot(range(len(losses)), losses)
plt.show()
print(losses)
