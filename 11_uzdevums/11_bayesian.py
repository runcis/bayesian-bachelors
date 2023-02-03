import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

BATCH_SIZE = 64
LEARNING_RATE=1e-3

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=T)
test_data = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=T)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

def create_lenet():
    model = nn.Sequential(

        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),

        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

def testModel(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)

        value, prediction = torch.max(x, 1)
        prediction = prediction.data.cpu()
        total += x.size(0)

        correct += torch.sum(prediction == labels)

    return correct *100./total

def trainModel(epochs):
    accuracies = []
    cnn = create_lenet().to(DEVICE)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    #max_accuracy

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE),
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(testModel(cnn, test_dataloader))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        print('Epoch:', epoch + 1, "Accuracy :", accuracy, '%')
    plt.plot(accuracies)
    return best_model

lenet = trainModel(40)