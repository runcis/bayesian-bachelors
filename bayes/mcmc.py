import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the mushroom dataset
mushroom_data = pd.read_csv("../data/mushrooms.csv")

# Convert categorical features to embeddings
for col in mushroom_data.columns:
    mushroom_data[col] = pd.Categorical(mushroom_data[col])
    mushroom_data[col] = mushroom_data[col].cat.codes

# Split the data into training and test sets
train_data = mushroom_data.sample(frac=0.8, random_state=42)
test_data = mushroom_data.drop(train_data.index)


# The model
class MushroomModel(nn.Module):
    def __init__(self):
        super(MushroomModel, self).__init__()
        self.fc1 = nn.Linear(22, 50)
        self.fc2 = nn.Linear(50, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Define the likelihood function
def likelihood(model, x, y):
    logits = model(x)
    log_likelihood = torch.nn.functional.cross_entropy(logits, y)
    return log_likelihood

# Define the prior distribution
def prior(model):
    log_prior = 0
    for name, param in model.named_parameters():
        log_prior += torch.distributions.Normal(0, 1).log_prob(param).sum()
    return log_prior

# Define the posterior distribution
def posterior(model, x, y):
    log_likelihood = likelihood(model, x, y)
    log_prior = prior(model)
    log_posterior = log_likelihood + log_prior
    return log_posterior

# Define the MCMC algorithm
def mcmc(model, train_loader, num_samples, step_size):
    samples = []
    current_log_posterior = posterior(model, *next(iter(train_loader)))
    current_params = model.state_dict()

    for i in range(num_samples):
        proposal_params = {}
        for name, param in model.named_parameters():
            proposal_params[name] = param + torch.randn_like(param) * step_size
        model.load_state_dict(proposal_params)
        proposal_log_posterior = posterior(model, *next(iter(train_loader)))
        log_accept_ratio = proposal_log_posterior - current_log_posterior

        if torch.log(torch.rand(1)) < log_accept_ratio:
            current_log_posterior = proposal_log_posterior
            current_params = model.state_dict()

        samples.append(current_params)

    return samples


# Generate samples
num_samples = 1000
step_size = 0.1
model = MushroomModel(input_size=len(train_df.columns) - 1, hidden_size=32, output_size=2)
samples = mcmc(model, train_loader, num_samples, step_size)

# Evaluate the posterior predictive distribution
test_inputs = torch.tensor([x for x, y in test_dataset])
test_targets = torch.tensor([y for x, y in test_dataset])
logits_list = []

for sample in samples:
    model.load_state_dict(sample)
    logits = model(test_inputs.float())
    logits_list.append(logits.unsqueeze(0))

logits_tensor = torch.cat(logits_list)
mean_logits = logits_tensor.mean(dim=0)
probs = softmax(mean_logits, dim=-1)

# Compute the uncertainty estimates
variance = ((probs - probs.mean(dim=0)) ** 2).mean(dim=0)
std_dev = torch.sqrt(variance)