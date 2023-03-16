import torch
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import csv
data_path = "../data/Concrete_Data.csv"

LEARNING_RATE = 0.001
TEST_TRAIN_SPLIT = 0.9
INPUT_COUNT = 8
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

dataset_full = DatasetConcrete()
train_test_split = int(len(dataset_full) * TEST_TRAIN_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

train_x, train_y = zip(*dataset_train)
train_x=torch.stack(list(train_x), dim=0)
train_y=torch.stack(list(train_y), dim=0)
test_x, test_y = zip(*dataset_test)
test_x=torch.stack(list(test_x), dim=0)
test_y=torch.stack(list(test_y), dim=0)

X_full = dataset_full.X
Y_full = dataset_full.Y

# Define the model using PyMC
# 3
with pm.Model() as model:
    X = pm.Data('X', train_x)

    # Define the priors
    w = pm.Normal('w', mu=0, sd=1, shape=INPUT_COUNT)

    # Define prior for the bias term
    b = pm.Normal('b', mu=0, sd=1)

    # Define the likelihood function
    mu = pm.math.dot(X, w) + b
    sigma = pm.HalfNormal('sigma', 1, shape=1)
    y_pred = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=train_y)

    # Run the inference algorithm
    approx = pm.fit(20000, method='advi')

    # Draw samples from the posterior distribution
    trace = approx.sample(draws=1000)


with model:
    pm.set_data({'X': test_x})
    ppc = pm.sample_posterior_predictive(trace)
    y_pred = ppc['y_obs'].mean(axis=0)
    output_std = torch.mean(ppc['y_obs'].std(axis=0))
    test_loss = torch.sqrt(torch.mean((test_y - y_pred) ** 2))


print("Test msr error: ", test_loss.item())

print("average STD: ", output_std)



