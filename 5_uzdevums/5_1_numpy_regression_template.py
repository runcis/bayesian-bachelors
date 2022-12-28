import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-2
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.75

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            data = pickle.load(fp)
            self.X, self.Y, self.labels = data

        self.X = np.array(self.X)
        X_max = np.max(self.X, axis=0)
        X_min = np.min(self.X, axis=0)
        self.X = (self.X - (X_max + X_min) * 0.5) /(X_max - X_min) * 0.5

        self.Y = np.array(self.Y)
        Y_max = np.max(self.Y)
        Y_min = np.min(self.Y)
        self.Y = (self.Y - (Y_max + Y_min) * 0.5) /(Y_max - Y_min) * 0.5
        self.Y = self.Y[:,0] # dadu kopā par vienu parametru par daudz

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), self.Y[idx]

class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
    
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch > len(self):
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size
        x, y = self.dataset[idx_start:idx_end]
        y = y[:, np.newaxis] # tas pats kas: y = np.expand_dims(y, axis=-1)
        self.idx_batch += 1
        return x, y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W:  Variable = Variable(
            value=np.random.uniform(low=-1, size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b: Variable = Variable(
            value=np.zeros(shape=(out_features,)),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
    
        self.output = Variable(
            np.squeeze(self.W.value.T @ np.expand_dims(x.value, axis=-1), axis=-1) + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += 1 * self.output.grad
        self.W.grad += self.x.value[:, :, np.newaxis] @ self.output.grad[:, np.newaxis, :]
        tempGrad = self.W.value @ self.output.grad[:, :, np.newaxis]
        self.x.grad += tempGrad[:, :, 0]

class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(1.0 / (1.0 + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1.0 - self.output.value) * self.output.grad

class LayerRelu():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        temp = self.x.value
        temp[temp<0]=0
        self.output = Variable( temp )
        return self.output

    def backward(self):
        temp = self.output.value
        temp[temp<0]=0
        temp[temp>0]=1
        self.x.grad += temp * self.output.grad

class LayerSwish():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(x.value / (1.0 + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value + np.std(x) * (1.0 - self.output.value) 

class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.sum((y.value - y_prim.value)**2))
        return loss

    def backward(self):
        self.y_prim.grad += -2*(self.y.value - self.y_prim.value)


class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.abs(y.value - y_prim.value))
        return loss

    def backward(self):
        self.y_prim.grad += -(self.y.value - self.y_prim.value) / (np.abs(self.y.value - self.y_prim.value) + 1e-8)

class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=6, out_features=8), #izmainiju uz 6 in features nevis
            LayerSwish(),
            LayerLinear(in_features=8, out_features=12),
            LayerSwish(),
            LayerLinear(in_features=12, out_features=7),
            LayerSwish(),
            LayerLinear(in_features=7, out_features=1),
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

def calculateNRMSE(y, y_prim):
    rmse = np.sqrt(np.mean(np.sum((y_prim - y)**2)))
    result = rmse/np.std(y)
    return result


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossMAE()


loss_plot_train = []
loss_plot_test = []
nrmse_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        nrmse = []
        for x, y in dataloader:

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(Variable(value=y), y_prim)
            nrmse_value = calculateNRMSE(y, y_prim.value)

            losses.append(loss)
            nrmse.append(nrmse_value)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))
            nrmse_plot_test.append(np.mean(nrmse))

        

    #print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 150 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax3 = ax2.twinx()
        ax3.plot(nrmse_plot_test, 'b-', label='nrmse')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax3.legend(loc='lower left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()