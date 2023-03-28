import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

PERCENT_OF_MIXED_LABELS = 0.1
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.7
LEARNING_RATE = 0.01

# Load the mushroom dataset
data_path = "../data/mushrooms.csv"

class MushroomDataset(torch.utils.data.Dataset):
    def __init__(self, train=False):

        data = pd.read_csv(data_path, header=None)

        # Encode y values
        labelValues = pd.Categorical(data[0][1:]).codes
        self.y = torch.tensor(pd.get_dummies(labelValues).values, dtype=torch.float32)



        # Encode x values
        data = data.drop(columns=[0])
        self.X = []
        self.inputSize = 0
        for column in data:
            values = pd.Categorical(data[column]).codes
            tensor = torch.tensor(pd.get_dummies(values).values, dtype=torch.float32)
            self.X.append(tensor)
            self.inputSize += len(data[column].unique())

        if train:
            self.X = [x[:round(len(x) * TRAIN_TEST_SPLIT)] for x in self.X]
            self.y = self.y[:round(len(self.y) * TRAIN_TEST_SPLIT)]

            # create a mask of indices to flip
            mask = np.random.choice(len(self.y), int(len(self.y) * PERCENT_OF_MIXED_LABELS), replace=False)
            # flip the values at the selected indices
            self.y[mask] = 1 - self.y[mask]
            self.num_samples = len(self.y)

        else:
            self.X = [x[round(len(x) * TRAIN_TEST_SPLIT):] for x in self.X]
            self.y = self.y[round(len(self.y) * TRAIN_TEST_SPLIT):]
            self.num_samples = len(self.y)

    def __getitem__(self, index):
        # return the feature and label at the given index
        return [x[index] for x in self.X], self.y[index]

    def __len__(self):
        # return the total number of samples in the dataset
        return self.num_samples


mushroom_data = pd.read_csv(data_path)

train_dataset = MushroomDataset(train=True)
test_dataset = MushroomDataset(train=False)

dataloader_train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class BNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, criterion, X_test, y_test):
    model.eval()
    output = model(X_test)
    loss = criterion(output, y_test)
    pred_max = torch.argmax(output, dim=1)
    actual_max = torch.argmax(y_test, dim=1)
    correct_count = (pred_max == actual_max).sum().item()
    acc = correct_count / y_test.shape[0]
    return loss.item(), acc


# Set the model hyperparameters
input_dim = train_dataset.inputSize
hidden_dim = 128
output_dim = 2

# Initialize the model and optimizer
model = BNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Train the model with Monte Carlo Dropout

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
for step in range(200):
    losses = []
    accs = []
    for x, y in dataloader_train:
        train(model, optimizer, criterion, x, y)

    for x, y in dataloader_test:
        loss, acc = evaluate(model, criterion, x, y)

        losses.append(loss)
        accs.append(acc)

    loss_plot_test.append(np.mean(losses))
    acc_plot_test.append(np.mean(accs))


    if step % 10 == 0:
        _, axes = plt.subplots(nrows=2, ncols=1)
        ax1 = axes[0]
        ax1.set_title("Loss")
        ax1.plot(loss_plot_test, 'r-', label='test')
        ax1.legend()
        ax1.set_ylabel("Loss")

        ax1 = axes[1]
        ax1.set_title("Acc")
        ax1.plot(acc_plot_test, 'r-', label='test')
        ax1.legend()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Acc.")
        plt.show()
        print('Step: ', step, 'got accuracy: ', acc_plot_test[-1])



print('DROPOUT: mixed labels: ',PERCENT_OF_MIXED_LABELS, 'got accuracy: ', acc_plot_test[-1])