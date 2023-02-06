import numpy as np
import torch
import sklearn.model_selection
from sklearn import datasets
from matplotlib import pyplot

import scipy.stats
import matplotlib.pyplot as plt


BATCH_SIZE = 18
LEARNING_RATE = 0.001

housing = datasets.fetch_california_housing()
print(housing.feature_names)

x = housing.data
y = housing.target


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
mu = 10
sigma = 2.

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=8, out_features=12),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=12, out_features=16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=16, out_features=12),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=12, out_features=8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=4, out_features=1),
            torch.nn.Softmax()
        )

        self.nn_layers = torch.nn.ModuleList()

    def forward(self, x):
        # out = x
        # for layer in self.layers:
        #     out = layer.forward(out)
        # return out
        y_prim = self.layers.forward(x)
        return y_prim

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()



model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
criteria = torch.nn.MSELoss()

loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    losses = []
    for x in (x_train):
        #forwards
        y_prim = model.forward(x)
        loss = criteria(y_prim, y)

        #backwards
        loss.backward()

        #update
        optimizer.step()
        optimizer.zero_grad()

    if (epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#plot
#predicted = model(x).detach().numpy()
#plt.plot(x)