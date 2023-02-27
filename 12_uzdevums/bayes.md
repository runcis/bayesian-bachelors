Variational inference vs Montecarlo dropout

Mēģināju izveidot balstoties uz šo paraugu: https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Regression.ipynb
Bet sastapos ar problemu ka izmantoju viarāku mainīgo datusetu.

Modelis BVI:
~~~
class BayesianNet(torch.nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=30, prior_sigma=1,
                                    in_features=8, out_features=16)
        self.oupt = bnn.BayesLinear(prior_mu=30, prior_sigma=1,
                                    in_features=16, out_features=1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()
        return z

model = BayesianNet()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
~~~

Modelis MCD:
~~~
class MonteCarloNet(torch.nn.Module):
    def __init__(self):
        super(MonteCarloNet, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear( in_features=8, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear( in_features=16, out_features=1),
        )

    def forward(self, x):
        z = self.network(x)
        return z
~~~

Trenēšana BVI:
~~~
mse_loss = torch.nn.MSELoss()   # applies softmax()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.1

for epoch in range(1, 300):
    losses = []

    for x, y in dataloader_train:

        y_prim = model(x)

        cel = mse_loss(y_prim, y)
        kll = kl_loss(model)
        loss = cel + kll # kll* .1 - why should we reduce?

        losses.append(loss.item())# accumulate

        loss.backward()  # update wt distribs
        optimizer.step()
        optimizer.zero_grad()

~~~

Trenēšana MCD:
~~~
mse_loss = torch.nn.MSELoss()
loss_plot_train = []

for epoch in range(1, 1000):

    for x, y in dataloader_train:

        y_prim = model(x)

        loss = torch.mean((y - y_prim) ** 2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_plot_train.append(np.mean(loss.item()))
~~~

Testēšana BVI:
~~~
for x, y in dataloader_test:
    plt.scatter(y, range(len(y)), color='b')

    models_result = np.array([model(x).data.numpy() for k in range(100)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])
    plt.scatter(y.data.numpy(), mean_values, color='g', lw=3, label='Predicted Mean Model')
    plt.show()

~~~

![concrete-bayes-inf.PNG](..%2Fmedia%2Fconcrete-bayes-inf.PNG)

Testēšana MCD:
~~~
for x, y in dataloader_test:
    plt.scatter(y, range(len(y)), color='b')

    models_result = np.array([model(x).data.numpy() for k in range(100)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])
    plt.scatter(y.data.numpy(), mean_values, color='g', lw=3, label='Predicted Mean Model')
    plt.show()
~~~

Atvērtie jautajumi:
Vai ir izvelets labs datuset? *visos piemeros bvi redzu tikai 1 parametra funkcijas.
Kā analizēt un novērtēt vērtību sadalījumu start dažādām pieejam? 



