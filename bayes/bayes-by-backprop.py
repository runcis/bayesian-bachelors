import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
BATCH_SIZE = 100
TEST_BATCH_SIZE = 5
EPOCHS = 1000
TEST_TRAIN_SPLIT = 0.7
PRIOR_TYPE = 'gaussian' # 'gaussian' or 'gsm'
PERCENT_OF_MIXED_LABELS = 0.3
KL_WEIGHT = 0.1

# Load the mushroom dataset
data = pd.read_csv("../data/mushrooms.csv")

# Set column names
col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
             'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
             'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
             'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
data.columns = col_names

# Convert categorical features to numeric
for col in data.columns:
    data[col] = pd.Categorical(data[col]).codes

train_size = int(len(data) * TEST_TRAIN_SPLIT)
train_data = data.iloc[:train_size, :]
test_data = data.iloc[train_size:, :]

# mix class label randomly
num_train = len(train_data)
num_to_mix = int(num_train*PERCENT_OF_MIXED_LABELS)
mix_indices = np.random.choice(num_train, num_to_mix, replace=False)
for idx in mix_indices:
    train_data.at[idx, 'class'] = 1 - train_data.at[idx, 'class']

X_train = train_data.drop('class', axis=1)
X_train = np.array(X_train)
Y_train = train_data['class'].to_numpy()
X_train, Y_train = torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()


X_test = test_data.drop('class', axis=1)
X_test = np.array(X_test)
Y_test = test_data['class'].to_numpy()
X_test, Y_test = torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=22, out_features=100),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=1),
    nn.Softmax(),
)


ce_loss = nn.CrossEntropyLoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for step in range(EPOCHS):
    pre = model(X_train)
    ce = ce_loss(pre[:,0], Y_train)
    kl = kl_loss(model)
    cost = ce + kl_weight * kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

_, predicted = torch.max(pre.data, 1)
total = Y_train.size(0)
correct = (predicted == Y_train).sum()
print('- Training: Accuracy: %f %%' % (100 * float(correct) / total))
print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))


total = Y_test.size(0)
test_result = model(X_test)

correct = (test_result[:,0] == Y_test).sum()
print('- Test Accuracy: %f %%' % (100 * float(correct) / total))