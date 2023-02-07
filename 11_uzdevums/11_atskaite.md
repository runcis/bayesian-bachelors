### 11 bayesian regresison task 

Sākumā mēģināju uztaisīt parastu modeli, kas peredzēs māju cenu no dotajiem atribūtiem.

Daudz cīnijos ar datuseta importēšanu - nebiju lietojis sklearn:
~~~
housing = datasets.fetch_california_housing()
print(housing.feature_names)

x = housing.data
y = housing.target


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
~~~

Kad dati sadalīti šādi - īsti nezinu kā konstruēt epoch ciklus

Es implementēju dataset un dataloader lai sanāktu izveidot vismaz strādājosu modeli:

~~~
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(

            torch.nn.Linear(in_features=NUMBER_OF_FEATURES, out_features=8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=4, out_features=1),
            torch.nn.Sigmoid(),
            torch.nn.Tanh()
        )

        self.nn_layers = torch.nn.ModuleList()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()
~~~
+ MSELoss un Adam optimizer:
~~~
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
criteria = torch.nn.MSELoss()
~~~
Rezultāts izskatās šādi:
![11_linear_model.PNG](..%2Fmedia%2F11_linear_model.PNG)


Atradu online bibliotēku torchbnn un izmanotju lai uztaisītu modeli:
~~~
class BayesianNet(torch.nn.Module):
    def __init__(self):  # 4-100-3
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                    in_features=8, out_features=16)
        self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                    in_features=16, out_features=1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()
        return z
~~~
Loss:
~~~
ce_loss = torch.nn.CrossEntropyLoss()   # applies softmax()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
~~~
Epochs:
~~~
    for dataloader in [dataloader_train, dataloader_test]:
        losses = []

        stage = 'train'
        if dataloader == dataloader_test:
            stage = 'test'

        for x, y in dataloader:

            y_prim = model(x)

            cel = ce_loss(y_prim, y)
            kll = kl_loss(model)
            loss = cel + (0.10 * kll)

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
~~~
Graph:
![11_using-torchbnn.PNG](..%2Fmedia%2F11_using-torchbnn.PNG)
*ļoti ātri aizgaja līdz 0.