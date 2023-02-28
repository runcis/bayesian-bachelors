import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
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
train_test_split = int(len(dataset_full) * 0.9)
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

class MonteCarloNet(torch.nn.Module):
    def __init__(self):
        super(MonteCarloNet, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear( in_features=8, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear( in_features=16, out_features=8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear( in_features=8, out_features=1),
        )

    def forward(self, x):
        z = self.network(x)
        return z


model = MonteCarloNet()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

mse_loss = torch.nn.MSELoss()
loss_plot_train = []

for epoch in range(1, 2000):

    for x, y in dataloader_train:

        y_prim = model(x)

        loss = torch.mean((y - y_prim) ** 2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_plot_train.append(np.mean(loss.item()))

fig, ax1 = plt.subplots()
ax1.plot(loss_plot_train, 'r-', label='train')
ax1.legend()
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
plt.show()

for x, y in dataloader_test:
    plt.scatter(y, range(len(y)), color='b')

    models_result = np.array([model(x).data.numpy() for k in range(500)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])

    plt.scatter(y.data.numpy(), mean_values, color='g', lw=1, label='Predicted Mean Model')
    plt.errorbar(y.data.numpy(), mean_values, yerr=std_values, fmt="o")
    plt.show()

