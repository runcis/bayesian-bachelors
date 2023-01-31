import os
import pickle

import torch
import numpy as np
from torch.hub import download_url_to_file
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import scipy.ndimage
import sklearn.decomposition

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TRAIN_TEST_SPLIT = 0.8
DEVICE = 'cuda'
MAX_LEN = 0

if not torch.cuda.is_available():
    MAX_LEN = 0 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
    DEVICE = 'cpu'
    BATCH_SIZE = 64



class DatasetApples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/apples_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../../../Downloads/data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/apples_dataset_9_2.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        X = torch.from_numpy(np.array(X))
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = Y.unsqueeze(dim=-1)

        self.labels.append('Cats')

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y_label = self.Y[idx]

        #self.applyNoise(x)
        y_target = x

        return x, y_target, y_label

    def applyNoise(self, x):
        noise = torch.zeros(x.size())
        noise = torch.poisson(noise)
        x[noise < 0.5] = 0

class DatasetAnomalies(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        img1 = np.array([])
        img2 = np.array([])
        img3 = np.array([])
        img4 = np.array([])
        img5 = np.array([])
        with Image.open("../data/9_5_anomalies/cat_1.jpg") as im:
            img1= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_2.jpg") as im:
            img2= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_3.jpg") as im:
            img3= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_4.jpg") as im:
            img4= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_5.jpg") as im:
            img5= np.array(im)

        arrayOfImages = np.concatenate((img1[np.newaxis,:], img2[np.newaxis,:], img3[np.newaxis,:], img4[np.newaxis,:], img5[np.newaxis,:]),0)

        X = torch.from_numpy(np.array(arrayOfImages))
        self.X = X.permute(0, 3, 1, 2)
        Y = torch.LongTensor([5, 5, 5, 5, 5])
        self.Y = Y.unsqueeze(dim=-1)
        self.labels = ['orange']
        self.input_size = 100

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y_label = self.Y[idx]

        y_target = x

        return x, y_target, y_label


dataset_apples = DatasetApples()
dataset_anomalies = DatasetAnomalies()

train_test_split = int(len(dataset_apples) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_apples,
    [train_test_split, len(dataset_apples) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataset_test = torch.utils.data.ConcatDataset([dataset_test, dataset_anomalies])

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1)
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1)
)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=60),

            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=8, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=30),

            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=16, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=10),

            torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=24, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=3),

            torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=32, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=1),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=32, num_groups=2),
            torch.nn.Tanh()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=24, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=3),

            torch.nn.Conv2d(in_channels=24, out_channels=18, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=18, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=18, out_channels=12, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=12, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=10),

            torch.nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=8, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=60),

            torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=100),

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=3, num_groups=1),
            torch.nn.Tanh()
        )


    def forward(self, x):
        z = self.encoder.forward(x)
        z = z.view(-1, 32)
        y_prim = self.decoder.forward(z.view(-1, 32, 1, 1))
        return y_prim, z


model = AutoEncoder()

dummy = torch.randn((BATCH_SIZE, 3, 100, 100))
y_target = model.forward(dummy)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'z',
        'labels'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]:
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y_target, y_label in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_target = y_target.to(DEVICE)
            y_label = y_label.squeeze().to(DEVICE)

            y_prim, z = model.forward(x)
            loss = torch.mean((x - y_prim)**2)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_z = z.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y_target = y_target.cpu().data.numpy()
            np_y_label = y_label.cpu().data.numpy()

            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_labels'] += np_y_label.tolist()

    metrics_strs = []
    for key in metrics_epoch.keys():
        if '_z' not in key and '_labels' not in key:
            value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18]):
        plt.subplot(8, 6, j) # row col idx
        plt.title(f"class: {dataset_apples.labels[np_y_label[i]]}")
        plt.imshow(np.transpose(np_x[i], (1, 2, 0)))

        plt.subplot(8, 6, j+6) # row col idx
        plt.imshow(np.transpose(np_y_prim[i], (1, 2, 0)))

    plt.subplot(223) # row col idx

    pca = sklearn.decomposition.KernelPCA(n_components=2, gamma=0.1)

    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    np_z_label = np.array(metrics_epoch[f'train_labels'])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=dataset_apples.labels)

    plt.subplot(224) # row col idx

    plt.title('test_z')
    np_z = np.array(metrics_epoch[f'test_z'])
    np_z = pca.fit_transform(np_z)
    np_z_label = np.array(metrics_epoch[f'test_labels'])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=dataset_apples.labels)

    plt.tight_layout(pad=0.5)
    plt.show()
