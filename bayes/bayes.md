Variational inference vs Montecarlo dropout

Mēģināju izveidot balstoties uz šo paraugu: https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Regression.ipynb
Bet sastapos ar problemu ka izmantoju viarāku mainīgo datusetu.

### Modelis BVI:
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

### Modelis MCD:
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

### Trenēšana BVI:
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

### Trenēšana MCD:
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

### Testēšana BVI:
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

~~~
x0 = dataloader_test.dataset[0][0]
y0 = dataloader_test.dataset[0][1]

x0_result = np.array([model(x0).data.numpy() for k in range(500)])
x0_result = x0_result[:,0]

sns.displot(x=x0_result, kind="kde", color='green', label="Predicted range")
plt.axvline(x=y0.data.numpy(), color='red')
plt.title("True data vs predicted distribution")
plt.show()
~~~
![true_vs_predicted.PNG](..%2Fmedia%2Ftrue_vs_predicted.PNG)

### Testēšana MCD:
~~~
for x, y in dataloader_test:
    plt.scatter(y, range(len(y)), color='b')

    models_result = np.array([model(x).data.numpy() for k in range(500)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])

    plt.scatter(y.data.numpy(), mean_values, color='g', lw=1, label='Predicted Mean Model')
    plt.errorbar(y.data.numpy(), mean_values, yerr=std_values, fmt="o")
    plt.show()
~~~
![concrete-mcd.PNG](..%2Fmedia%2Fconcrete-mcd.PNG)
Pievienojot std grafikam:
~~~
for x, y in dataloader_test:
    plt.scatter(y, range(len(y)), color='b')

    models_result = np.array([model(x).data.numpy() for k in range(500)])
    models_result = models_result[:, :, 0]
    models_result = models_result.T

    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])

    plt.scatter(y.data.numpy(), mean_values, color='g', lw=1, label='Predicted Mean Model')
    plt.errorbar(y.data.numpy(), mean_values, yerr=std_values, fmt="o")
    plt.title("True data vs predicted with std")
    plt.show()
~~~
![mean_w_std_vs_actual.PNG](..%2Fmedia%2Fmean_w_std_vs_actual.PNG)


##### Atvērtie jautajumi par BVI:

Vai ir izvelets labs datuset? *visos piemeros bvi redzu tikai 1 parametra funkcijas un pārsvarā klasifikācija uzdevumi nevis regresija.
Nesaprotu kā strādā tas vairāku mainīgo BNN - mēs hardkodējam mu un prior - vai tas tiek pievienots katram inputam?
Šis liekas ka nav pareizi, jo KL nesamazinas.

Nesaprotu kā pievienot mu un sigma kā apmācāmus parametrus.

##### Atvertie jautajumi par MCD:
Vai ideju esmu sapratis pareizi - Tiek veidots tīkls, kuram ir dropout slāņi un varbūtība tiek iegūta atkārtojot mērijumu ar tīklu un izvelkot mean un std.


##### Atvertie jautājumi par Bayes by Backprop:
TODO: Man vel vajag izmeklēt, Esmu apjucis - kā tas atšķiras no BVI.


##### Papildu jautājumi:
Kā analizēt un novērtēt vērtību sadalījumu starp dažādām pieejam? 
    idejas: 
    <ol>
        <li>cik liela daļa no rezultātiem iekļaujas ~98%</li>
        <li>cik resursu pieprasošs ir risinājums</li>
    </ol>


