import argparse
import copy
import json
import math
import os
import pathlib
import random

import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy import spatial

import scipy
import torch
import numpy as np
import matplotlib
import torchvision
import torch.utils.data
import torch.distributions

import matplotlib.pyplot as plt
from torchvision import transforms


BATCH_SIZE = 18
TRAIN_TEST_SPLIT = 0.8
VAE_BETA = 0.001
LEARNING_RATE = 0.001

dataset = torchvision.datasets.EMNIST(
    root='../data',
    split='bymerge',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# examine samples to select idxes you would like to use


idx = 0
for x, y_idx in data_loader:
    plt.rcParams["figure.figsize"] = (int(BATCH_SIZE/4), int(BATCH_SIZE/4))
    plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
    for i in range(BATCH_SIZE):
        plt.subplot(plt_rows, plt_rows, i + 1)
        plt.imshow(x[i][0].T, cmap=plt.get_cmap('Greys'))
        plt.title(f"idx: {idx}")
        idx += 1
        plt.tight_layout(pad=0.5)
    plt.show()

    break
    if input('inspect more samples? (y/n)') == 'n':
        break


class VAE2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=4, out_channels=8, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=8, out_channels=16, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=16, num_groups=8),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=16, out_channels=32, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=32, num_groups=8)
        )

        self.encoder_mu = torch.nn.Linear(in_features=32, out_features=32)
        self.encoder_sigma = torch.nn.Linear(in_features=32, out_features=32)

        self.decoder = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=32, out_channels=16, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=16, num_groups=8),


            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=16, out_channels=8, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),

            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=8, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),

            torch.nn.AdaptiveAvgPool2d(output_size=(28, 28)),
            torch.nn.Conv2d(in_channels=4, out_channels=1, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(normalized_shape=[1, 28, 28]),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder.forward(x)

        out_flat = out.view(x.size(0), -1)

        z_sigma = self.encoder_sigma.forward(out_flat)
        z_mu = self.encoder_mu.forward(out_flat)

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size())
        z = z_mu + eps * z_sigma

        z_2d = z.view(x.size(0), -1, 1, 1)
        y_prim = self.decoder(z_2d)
        return y_prim, z, z_sigma, z_mu

    def encode_z(self, x):
        out = self.encoder(x)

        out_flat = out.view(x.size(0), -1)

        z_sigma = self.encoder_sigma.forward(out_flat)
        z_mu = self.encoder_mu.forward(out_flat)

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size())
        z = z_mu + eps * z_sigma

        return z

    def decode_z(self, z):
        z_2d = z.view(z.size(0), -1, 1, 1)
        y_prim = self.decoder(z_2d)
        return y_prim


model = VAE2()
model.load_state_dict(torch.load('./pretrained_models/mnist_mnist_bce_low_beta_fixed_log_epsilon-17-run-25.pt', map_location='cpu'))
model.eval()
torch.set_grad_enabled(False)

INDEXES_TO_ENCODE_ONES = [
   24, 25, 7, 221, 40, 57, 67, 74, 83, 81, 97, 111, 116, 134, 136,
    138, 163, 186, 170, 188, 223, 248, 254, 258, 275, 294, 303,
    347, 350, 368, 376, 415    # all these are 1
]

INDEXES_TO_ENCODE_ZERO = [
   0, 33, 34, 35, 44, 45, 47, 48, 59, 71, 89, 91, 104, 106, 112, 118,
    120, 123]
#  124, 132, 149, 152, 157, 161, 183, 198, 210, 220, 231,243, 244, 272 # all these are similar to0


INDEXES_TO_ENCODE_THREE = [
   6, 12, 344, 22, 85, 101, 144, 156, 167, 173, 189, 199, 241, 251,
   306, 332, 323, 337] # , 365, 404   # all these are similar to 3


INDEXES_TO_ENCODE_PLUS = [
     19, 32, 68, 76, 93, 491, 465, 141, 153, 146, 147, 168, 179, 195,
    197, 201, 215, 343    # all these are similar to + (f or small t or 4)
]


# ENCODING 1:
x_to_encode = []
for idx in INDEXES_TO_ENCODE_THREE:
    x_to_encode.append(dataset[idx][0])

plt_rows = int(np.ceil(np.sqrt(len(x_to_encode))))
for i in range(len(x_to_encode)):
    plt.subplot(plt_rows, plt_rows, i + 1)
    x = x_to_encode[i]
    plt.imshow(x[0].T, cmap=plt.get_cmap('Greys'))
    plt.title(f"idx: {INDEXES_TO_ENCODE_THREE[i]}")
    plt.tight_layout(pad=0.5)
plt.show()

x_tensor = torch.stack(x_to_encode)
zs = model.encode_z(x_tensor)


# ENCODING 0:
x2_to_encode = []
for idx in INDEXES_TO_ENCODE_ZERO:
    x2_to_encode.append(dataset[idx][0])

plt_rows = int(np.ceil(np.sqrt(len(x2_to_encode))))
for i in range(len(x2_to_encode)):
    plt.subplot(plt_rows, plt_rows, i + 1)
    x = x2_to_encode[i]
    plt.imshow(x[0].T, cmap=plt.get_cmap('Greys'))
    plt.title(f"idx: {INDEXES_TO_ENCODE_ZERO[i]}")
    plt.tight_layout(pad=0.5)
plt.show()

x2_tensor = torch.stack(x2_to_encode)
zs2 = model.encode_z(x2_tensor)

z_comb = torch.add(torch.mean(zs, dim=0), torch.mean(zs2, dim=0))

z_mu = torch.mean(z_comb, dim=0)
z_sigma = torch.std(z_comb, dim=0)

# sample new letters
z_generated = []
dist = torch.distributions.Normal(z_mu, z_sigma)
for i in range(BATCH_SIZE):
    if i == 0:
        z = z_mu
    else:
        z = dist.sample()
    z_generated.append(z)
z = torch.stack(z_generated)
x_generated = model.decode_z(z)

plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
for i in range(BATCH_SIZE):
    plt.subplot(plt_rows, plt_rows, i + 1)
    plt.imshow(x_generated[i][0].T, cmap=plt.get_cmap('Greys'))
    if i == 0:
        plt.title(f"mean")
    else:
        plt.title(f"generated")
    plt.tight_layout(pad=0.5)
plt.show()