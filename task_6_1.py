import numpy as np
import csv


def get_data(filename):
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x_data.append([[int(pixel)] for pixel in row[1:]])
            y_data.append([[1] if int(row[0]) == i else [0] for i in range(10)])
    return np.array(x_data) / 255, np.array(y_data)


x_train, y_train = get_data('mnist_train.csv')
x_test, y_test = get_data('mnist_test.csv')


N_in = 28 * 28
N_out = 10
N_hidden = 64
lmd = 0.001


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    return np.where(x > 0, 1, 0)


W1 = np.random.rand(N_hidden, N_in) * 0.1
W2 = np.random.rand(N_out, N_hidden) * 0.1

batch_size = 30

def back_propagation(epochs):
    global W1, W2
    for i in range(epochs):
        k = np.random.randint(0, len(x_train) - batch_size - 1)
        for x_in, y_in in zip(x_train[k:k + batch_size], y_train[k:k + batch_size]):
            y1 = np.dot(W1, x_in)
            z = ReLU(y1)
            y2 = np.dot(W2, z)
            out = sigmoid(y2)
            e = out - y_in
            D2 = dsigmoid(out) * e
            W2 -= lmd * np.dot(D2, z.T)
            D1 = np.dot(W2.T, D2) * dReLU(z)
            W1 -= lmd * np.dot(D1, x_in.T)


def forward(x_in):
    y1 = np.dot(W1, x_in)
    z = ReLU(y1)
    y2 = np.dot(W2, z)
    out = sigmoid(y2)
    return out


back_propagation(10000)
accuracy = 0
for i in range(1000):
    pred = list([float(n[0]) for n in forward(x_test[i])])
    true = list([float(n[0]) for n in y_test[i]])
    if pred.index(max(pred)) == true.index(max(true)):
        accuracy += 1
    # print(*['{:0.9f}'.format(n[0]) for n in forward(x_test[i])])
    # print(*['{:0.9f}'.format(n[0]) for n in y_test[i]])
    # print()
print("accuracy = ", accuracy / 10)
