import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import torch
import pymc3 as pm
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import csv
data_path = "../data/Concrete_Data.csv"

class DatasetConcrete(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/Concrete_Data.csv'

        concrete_data = np.loadtxt(data_path, dtype=np.float32, delimiter=",", skiprows=1,
                                   usecols=(0, 1, 2, 3, 4, 5, 6, 7))
        concrete_strength = np.loadtxt(data_path, dtype=np.float32, delimiter=",", skiprows=1, usecols=(8))
        labels = next(csv.reader(open(data_path, encoding='utf-8-sig'), delimiter=","))

        X = torch.from_numpy(concrete_data).float()
        Y = torch.from_numpy(concrete_strength).float()

        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        # self.applyNoise(x)

        return x, y


dataset_full = DatasetConcrete()
train_test_split = int(len(dataset_full) * 0.99)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=128,
    shuffle=True,
    drop_last=(len(dataset_train) % 128 == 1)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=128,
    shuffle=True,
    drop_last=(len(dataset_test) % 128 == 1)
)

X = dataset_full.X
Y = dataset_full.Y


# Define Bayesian neural network architecture
class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define loss function and optimizer
net = BayesianNet()
kl_loss = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define training loop
num_epochs = 500
batch_size = 32
for epoch in range(num_epochs):
    permutation = torch.randperm(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        optimizer.zero_grad()
        y_pred = net(X[indices])
        noise = torch.randn_like(y_pred)
        y_sample = y_pred + 0.1 * noise
        log_prior = 0
        log_posterior = 0
        for param in net.parameters():
            log_prior += torch.sum(torch.distributions.Normal(0, 1).log_prob(param))
            log_posterior += torch.sum(torch.distributions.Normal(param, 0.1).log_prob(y_sample))
        loss = kl_loss(log_posterior - log_prior, torch.zeros_like(log_prior))
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Define testing loop
with torch.no_grad():
    y_pred = net(X)
    noise = torch.randn_like(y_pred)
    y_sample = y_pred + 0.1 * noise
    log_prior = 0
    log_posterior = 0
    for param in net.parameters():
        log_prior += torch.sum(torch.distributions.Normal(0, 1).log_prob(param))
        log_posterior += torch.sum(torch.distributions.Normal(param, 0.1).log_prob(y_sample))
    loss = kl_loss(log_posterior - log_prior, torch.zeros_like(log_prior))
    print(f"Test Loss: {loss.item()}")