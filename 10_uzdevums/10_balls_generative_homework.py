import argparse
import copy
import json
import math
import os
import pathlib
import random
import time

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


BATCH_SIZE = 24

class DatasetBalls(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.load('../data/10_balls_dataset.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pil_y = self.data[idx]
        pil_y = pil_y.astype(float) / 255.0 # 0..1
        np_y = np.array(pil_y)
        return torch.FloatTensor(np_y)

dataset = DatasetBalls()
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# examine samples to select idxes you would like to use


idx = 0
for x in data_loader:
    plt.rcParams["figure.figsize"] = (int(BATCH_SIZE/4), int(BATCH_SIZE/4))
    plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
    for i in range(BATCH_SIZE):
        plt.subplot(plt_rows, plt_rows, i + 1)
        plt.imshow(x[i])
        plt.title(f"idx: {idx}")
        idx += 1
        plt.tight_layout(pad=0.5)
    plt.show()

    break
    time.sleep(1)
    if input('inspect more samples? (y/n)') == 'n':
        break


class VAE2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
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
            torch.nn.Conv2d(in_channels=4, out_channels=3, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(normalized_shape=[3, 28, 28]),
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
model.load_state_dict(torch.load('./pretrained_models/balls_mse_low-12-run.pt', map_location='cpu'))
model.eval()
torch.set_grad_enabled(False)


INDEXES_TO_ENCODE_BIG_RED = [
  5, 10, 25, 33, 42, 51, 54, 76, 81, 90, 104, 120, 121, 123, 125, 130, 169, 186, 192, 205,
  242, 240, 252, 272, 274, 282, 285, 328, 339, 344, 345, 348
]#351

INDEXES_TO_ENCODE_SMALL_GREEN = [
  7, 11, 16, 18, 24, 35, 37, 39, 41, 50, 56, 61, 68, 79, 82, 84, 93, 118, 131, 143,
  144, 146, 148, 159, 164, 187, 193, 196, 215, 216, 219, 228
]

INDEXES_TO_ENCODE_BIG_CORNER = [
    5, 6, 8, 9, 10, 19, 26, 65, 66, 70, 76, 81, 137, 243, 249, 280, 285, 272, 293, 309, 312,
    344, 356, 395, 400, 401, 461, 477, 490, 520, 574, 580
]

INDEXES_TO_ENCODE_CORNER = [
    4,  7, 10, 19, 26, 31, 37, 39,  47, 65, 66, 123, 146, 164, 174, 180, 195, 200,
    224, 225, 237, 243, 249, 252, 253, 259, 284, 285, 293, 330, 341, 344,
]

# ENCODING big red:
x_to_encode = []
for idx in INDEXES_TO_ENCODE_BIG_RED:
    x_to_encode.append(dataset[idx])

x_tensor = torch.stack(x_to_encode)
zs = model.encode_z(x_tensor.permute(0, 3, 2, 1))

# ENCODING green small:
x2_to_encode = []
for idx in INDEXES_TO_ENCODE_SMALL_GREEN:
    x2_to_encode.append(dataset[idx])

x2_tensor = torch.stack(x2_to_encode)
zs2 = model.encode_z(x2_tensor.permute(0, 3, 2, 1))

z_comb = (zs + zs2) * 0.5

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

x_generated = model.decode_z(z).permute(0, 2, 3, 1)

plt_rows = int(np.ceil(np.sqrt(BATCH_SIZE)))
for i in range(BATCH_SIZE):
    plt.subplot(plt_rows, plt_rows, i + 1)
    plt.imshow(x_generated[i])
    if i == 0:
        plt.title(f"mean")
    else:
        plt.title(f"generated")
    plt.tight_layout(pad=0.5)
plt.show()