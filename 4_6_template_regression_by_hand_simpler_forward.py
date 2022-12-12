import matplotlib.pyplot as plt
import numpy as np

X = np.array([0.0, 2.0, 5.0, 11.0]) # + 2000
Y = np.array([2.1, 4.0, 5.5, 8.9]) # * 1000

W_1 = 0
b_1 = 0
W_2 = 0
b_2 = 0

def linear(W, b, x):
    return W * x + b

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def model(x, W_1, b_1, W_2, b_2):
    layer_1 = linear(W_1, b_1, x)
    layer_2 = sigmoid(layer_1)
    layer_3 = linear(W_2, b_2, layer_2)
    return layer_3

def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim - y))

def loss_mse(y_prim, y):
    return np.mean(np.sum((y_prim - y)**2))

Y_prim = model(X, W_1, b_1, W_2, b_2)
loss_mae = loss_mae(Y_prim, Y)
loss_mse = loss_mse(Y_prim, Y)

print(f'Y_prim: {Y_prim}')
print(f'Y: {Y}')
print(f'loss mae: {loss_mae}  loss mse: {loss_mse}')