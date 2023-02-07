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