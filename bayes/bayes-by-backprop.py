import torchbnn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import csv
data_path = "../data/Concrete_Data.csv"

LEARNING_RATE = 0.001
BATCH_SIZE = 100
TEST_BATCH_SIZE = 5
EPOCHS = 500
TEST_TRAIN_SPLIT = 0.7
PRIOR_TYPE = 'gaussian' # 'gaussian' or 'gsm'
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
train_test_split = int(len(dataset_full) * TEST_TRAIN_SPLIT)
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

data = dataset_full.X
target = dataset_full.Y
data_tensor=data.float()
target_tensor=target.long()

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=50),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=50, out_features=50),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=50, out_features=1),
)

cross_entropy_loss = nn.CrossEntropyLoss()
klloss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
klweight = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses_train = []
losses_test = []
#train
for epoch in range(EPOCHS):
    losses = []

    for x, y in dataloader_train:
        y_prim = model(x)
        cross_entropy = cross_entropy_loss(y_prim[:,-1], y)
        kl = klloss(model)
        total_cost = cross_entropy + klweight * kl

        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        losses.append(total_cost.item())

    losses_train.append(torch.mean(torch.tensor(losses)))

    losses = []
    for x, y in dataloader_test:
        y_prim = model(x)
        test_loss = torch.sqrt(torch.mean((y_prim[:,-1] - y) ** 2))
        losses.append(test_loss)
    losses_test.append(torch.mean(torch.tensor(losses)))

    if epoch % 30 == 0:
        _, axes = plt.subplots(nrows=2, ncols=1)
        ax1 = axes[0]

        ax1.plot(losses_train, 'r-', label='train')
        ax1.legend()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss training")

        ax1 = axes[1]
        ax1.plot(losses_test, 'b-', label='test')
        ax1.legend()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss testing ")
        plt.show()

