import torch
import pymc3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import torchbnn as bnn
data_path = "../data/Concrete_Data.csv"

LEARNING_RATE = 0.001
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
train_test_split = int(len(dataset_full) * 0.8)
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

class BayesianNet(torch.nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=30, prior_sigma=1,
                                    in_features=8, out_features=16)
        self.oupt = bnn.BayesLinear(prior_mu=30, prior_sigma=1,
                                    in_features=16, out_features=1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()
        return z

model = BayesianNet()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Define the Loss functions
mse_loss = torch.nn.MSELoss()   # applies softmax()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.1

for epoch in range(1, 1000):
    losses = []

    for x, y in dataloader_train:

        y_prim = model(x)

        cel = mse_loss(y_prim, y)
        kll = kl_loss(model)
        loss = cel + kll # kll* .1 - why should we reduce?

        losses.append(loss.item())# accumulate

        loss.backward()  # update wt distribs
        optimizer.step()
        optimizer.zero_grad()

# correct = (predicted == y).sum()
# print('- Accuracy: %f %%' % (100 * float(correct) / total))
# print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))

for x, y in dataloader_test:
    plt.scatter(y, range(len(y)), color='b')

    models_result = np.array([model(x).data.numpy() for k in range(100)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])
    plt.scatter(y.data.numpy(), mean_values, color='g', lw=3, label='Predicted Mean Model')
    plt.show()




