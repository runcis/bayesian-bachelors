import numpy as np
import torch
from sklearn import datasets
import torchbnn as bnn
import matplotlib.pyplot as plt

BATCH_SIZE = 128
LEARNING_RATE = 0.001
VAE_BETA = 0.001
TRAIN_TEST_SPLIT = 0.7
NUMBER_OF_FEATURES = 8

housing = datasets.fetch_california_housing()
print(housing.feature_names)

x = housing.data
y = housing.target


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        X = housing.data
        self.X = torch.from_numpy(x.astype(np.float32))
        Y = housing.target
        Y = torch.LongTensor(Y.astype(np.float32))
        self.Y = Y.unsqueeze(dim=-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        y = self.Y[idx]
        y = y.float()

        return x, y


dataset_housing = Dataset()
train_test_split = int(len(dataset_housing) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_housing,
    [train_test_split, len(dataset_housing) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


def linear(W, b, x):
    prod_W = np.squeeze(W.T @ np.expand_dims(x, axis=-1), axis=-1)
    return prod_W + b


mu = 0
sigma = 1

class BayesLinear():

    def __init__(self, in_features: int, out_features: int, prior_mu, prior_sigma):
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.w_mu = torch.nn.Linear(
            in_features=self.in_features,
            out_features=1
        )
        self.w_sigma = torch.nn.Linear(
            in_features=self.in_features,
            out_features=1
        )

    def forward(self, x):
        eps = torch.normal(mean=0.0, std=1.0, size=self.w_mu.size())
        z = self.w_mu + self.w_sigma * eps
        return z


class BayesianModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.hid1 = BayesLinear(prior_mu=mu, prior_sigma=sigma,
                                    in_features=NUMBER_OF_FEATURES, out_features=16)
        self.out = BayesLinear(prior_mu=mu, prior_sigma=sigma,
                                    in_features=16, out_features=1)

    def forward(self, x):
        z = self.hid1(x)
        z = torch.relu(z)
        z = self.out(z)

        return z #, z_mu, z_sigma for kl loss

    #TODO Jautājums: Te vajag backward function?

class LossCrossEntropy():
    def __init__(self):
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim
        return np.mean(-y.value * np.log(y_prim.value + 1e-8))

    def backward(self):
         self.y_prim.grad = -self.y.value / (self.y_prim.value + 1e-8)

class BKLLoss():

    def __init__(self):
        self.y_prim = None

    def forward(self, z_sigma, z_mu):
        torch.mean(VAE_BETA * torch.mean(
            (-0.5 * (2.0 * torch.log(z_sigma + 1e-8) - z_sigma ** 2 - z_mu ** 2 + 1))
        ), dim=0)

    def backward(self):
        # TODO
        print("Backward")


model = BayesianModel()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

ce_loss = torch.nn.CrossEntropyLoss()   # applies softmax()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []

        stage = 'train'
        if dataloader == dataloader_test:
            stage = 'test'

        for x, y in dataloader:

            y_prim = model(x)

            cel = ce_loss(y_prim, y)
            kll = kl_loss(model)
            loss = cel + kll # kll* .1 - why should we reduce?

            losses.append(loss.item())# accumulate

            if dataloader == dataloader_train:
                loss.backward()  # update wt distribs
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 10 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()
