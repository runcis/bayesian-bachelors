import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
from tqdm import tqdm
import torch.nn.functional as F

plt.rcParams["figure.figsize"] = (15,5)
plt.style.use('dark_background')

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
MAX_LEN = 200
INPUT_SIZE = 28
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0
    
class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data',
            train=is_train,
            download=True
        )
 
        a=0
        self.labels = self.data.classes

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)

        np_x = np.expand_dims(np_x, axis=0)

        x = torch.FloatTensor(np_x)

        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0

        y = torch.FloatTensor(np_y)
        return x, y

data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

def get_out_size(in_size, padding, kernel_size, stride):
    return int((in_size + 2 * padding - kernel_size) / stride + 1)

class Conv2d(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = torch.nn.Parameter(
            torch.FloatTensor(kernel_size, kernel_size, in_channels, out_channels)
        )
        torch.nn.init.kaiming_uniform_(self.K)

    def forward(self, x):
        batch_size = x.size(0)
        in_size = x.size(-1)
        out_size = get_out_size(in_size, self.padding, self.kernel_size, self.stride)

        out = torch.zeros(batch_size, self.out_channels, out_size, out_size)
        
        x_padded_size = in_size + self.padding*2
        if self.padding:
            x_padded = torch.zeros(batch_size, self.in_channels, x_padded_size, x_padded_size)
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        else:
            x_padded = x.to(DEVICE)

        K = self.K.view(-1, self.out_channels) # kernel_size8kernel_size*in_channels

        i_out = 0
        for i in range(0, x_padded_size-self.kernel_size+1, self.stride):
            j_out = 0
            for j in range(0, x_padded_size-self.kernel_size+1, self.stride):
                x_part = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                x_part = x_part.reshape(batch_size, -1)

                out_part = x_part @ K # (batch, out_channels)
                out[:, :, i_out, j_out] = out_part

                j_out += 1
            i_out += 1

        return out

class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size, 
        stride,
        padding
    ):

        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        in_size = x.size(-1)
        out_size = get_out_size(in_size, self.padding, self.kernel_size, self.stride)

        out = torch.zeros(batch_size, channels, out_size, out_size).to(DEVICE)

        x_padded_size = in_size + self.padding * 2
        if self.padding:
            x_padded = torch.zeros(batch_size, channels, x_padded_size, x_padded_size).to(DEVICE)
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
        else:
            x_padded = x.to(DEVICE)

        i_out = 0
        for i in range(0, x_padded_size-self.kernel_size+1, self.stride):
            j_out = 0
            for j in range(0, x_padded_size-self.kernel_size+1, self.stride):
                x_part = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                x_part = x_part.reshape(batch_size, channels, -1)

                out_part = torch.max(x_part, dim=-1).values # x_part @ K # (batch, out_channels)
                out[:, :, i_out, j_out] = out_part
                
                j_out += 1
            i_out += 1

        return out

class BatchNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features, 
        momentum
    ):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.gamma = torch.nn.Parameter(torch.ones(1, self.num_features, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, self.num_features, 1, 1))

        self.moving_mean = torch.zeros(1, self.num_features, 1, 1)
        self.moving_var = torch.ones(1, self.num_features, 1, 1)

    def forward(self, x):
        if x.size(1) != self.num_features:
            raise Exception('Wrong channel count in batchnorm')    

        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        if self.training:
            mean = x.mean(dim=(0,2,3), keepdims=True)
            var = ((x - mean)**2).mean(dim=(0,2,3), keepdims=True)

            self.moving = mean * self.momentum + self.moving_mean * (1 - self.momentum)
            self.moving_var = var * self.momentum + self.moving_var * (1 - self.momentum)
            out_norm = (x - mean) / torch.sqrt(var + 1e-5)
        else:
            out_norm = (x - self.moving_mean) / torch.sqrt(self.moving_var + 1e-5)
        out = self.gamma * out_norm + self.beta
        return out

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        out_channels = 16
        self.encoder = torch.nn.Sequential(
            Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(),
            BatchNorm2d(num_features=5, momentum=0.1),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv2d(in_channels=5, out_channels=9, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv2d(in_channels=9, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        o_1 = get_out_size(28, kernel_size=5, stride=1, padding=1)
        o_1 = get_out_size(o_1, kernel_size=3, stride=2, padding=1)
        o_2 = get_out_size(o_1, kernel_size=3, stride=1, padding=1)
        o_2 = get_out_size(o_2, kernel_size=3, stride=2, padding=1)
        o_3 = get_out_size(o_2, kernel_size=3, stride=1, padding=1)

        self.fc = torch.nn.Linear(
            in_features= out_channels * o_3 * o_3,
            out_features= 10 # 10 kategorijas ar apgerbu
        )

    def forward(self, x):
        batch_size = x.size(0) # x.size() > (B, C_in, W_in, H_in) > (64, 3, 28, 28)
        out = self.encoder.forward(x) # out.size() > (B, C_out, W_out, H_out) > (B, out_channels, o_3, o_3)
        out_flat = out.view(batch_size, -1) # out_flat.size() > (B, F)
        logits = self.fc.forward(out_flat) # logics.size(B, 10)
        y_prim = torch.softmax(logits, dim=1)

        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in tqdm(data_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE) # .cpu() / .cuda()

            y_prim = model.forward(x)
            loss = torch.mean(-y * torch.log(y_prim + 1e-88))

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()
            x = x.cpu()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')


    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()