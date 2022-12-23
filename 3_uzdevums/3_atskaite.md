# Task 3: Regression for comparing mae vs mse (4.6)

Vienkārši un saprotami priekš viendimensionālas datu kopas.

### List of implemented functions

1. Sigmoid function

~~~
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
~~~

2. Loss mae

~~~
def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim - y))
~~~

2. Loss mse

~~~
def loss_mse(y_prim, y):
    return np.mean(np.sum((y_prim - y)**2))
~~~

2. Model

~~~
def model(x, W_1, b_1, W_2, b_2):
    layer_1 = linear(W_1, b_1, x)
    layer_2 = sigmoid(layer_1)
    layer_3 = linear(W_2, b_2, layer_2)
    return layer_3
~~~
