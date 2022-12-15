import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.0, 2.0], [2.0, 2.0], [5.0, 2.5], [11.0, 3.0]]) # +2000
Y = np.array([[2.1], [4.0], [5.5], [8.9]]) # *1000

Y = np.expand_dims(Y, axis=-1) # out_features = 1

W_1 = np.zeros((2,8))
b_1 = np.zeros((8,))
W_2 = np.zeros((8,6))
b_2 = np.zeros((6, ))
W_3 = np.zeros((6,1))
b_3 = np.zeros((1, ))

def linear(W, b, x):
    prod_W = np.squeeze(W.T @ np.expand_dims(x, axis =-1), axis =-1)
    return prod_W + b

def tanh(x):
    result = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return result

def dx_tanh(x):
    result = (4 * np.exp(2*x))/((np.exp(2*x) + 1) ** 2)
    return result

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def model(x, W_1, b_1, W_2, b_2, W_3, b_3):
    layer_1 = linear(W_1, b_1, x)
    layer_2 = tanh(layer_1)
    layer_3 = linear(W_2, b_2, layer_2)
    layer_4 = tanh(layer_3)
    layer_5 = linear(W_3, b_3, layer_4)
    return layer_5


def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim - np.expand_dims(y, axis=-1)))

def loss_mse(y_prim, y):
    return np.mean(np.sum((y_prim - np.expand_dims(y, axis=-1))**2))

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W

def dx_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def dy_prim_loss_mae(y_prim, y):
    return (y_prim - np.expand_dims(y, axis=-1)) / (np.abs(y_prim - np.expand_dims(y, axis=-1)) + 1e-8)

def dy_prim_loss_mse(y_prim, y):
    return 2*(y_prim - np.expand_dims(y, axis=-1)) #or is it 2*np.mean(np.sum(y_prim - y)) ?

def dW_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_1 = dW_linear(W_1, b_1, x)
    d_layer_2 = dx_sigmoid(linear(W_1, b_1, x))
    d_layer_3 = np.expand_dims(dx_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x))), axis=-1)
    d_loss = dy_prim_loss_mse(y_prim, y)
    d_dot_3 = d_loss @ d_layer_3
    return d_dot_3 * d_layer_2 * d_layer_1

def db_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_1 = db_linear(W_1, b_1, x)
    d_layer_2 = dx_sigmoid(linear(W_1, b_1, x))
    d_layer_3 = np.expand_dims(dx_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x))), axis=-1)
    d_loss = dy_prim_loss_mse(y_prim, y)
    d_dot_3 = np.squeeze(d_loss @ d_layer_3, axis=-1).T
    return d_dot_3 * d_layer_2 * d_layer_1

def dW_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_3 = dW_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mse(y_prim, y)
    return d_loss * d_layer_3

def db_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_3 = db_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mse(y_prim, y)
    return d_loss * d_layer_3


learning_rate = 1e-2
losses = []
for epoch in range(1000):

    Y_prim = model(X, W_1, b_1, W_2, b_2, W_3, b_3)
    loss = loss_mse(Y_prim, Y)

    dW_1 = np.sum(dW_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    dW_2 = np.sum(dW_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    db_1 = np.sum(db_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    db_2 = np.sum(db_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))

    W_1 -= dW_1 * learning_rate
    W_2 -= dW_2 * learning_rate
    b_1 -= db_1 * learning_rate
    b_2 -= db_2 * learning_rate


    print(f'Y_prim: {Y_prim}')
    print(f'Y: {Y}')
    print(f'loss: {loss}')
    losses.append(loss)

plt.plot(losses)
plt.show()