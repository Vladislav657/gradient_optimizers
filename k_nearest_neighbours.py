import numpy as np
import csv


def get_data(filename):
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x_data.append([int(pixel) for pixel in row[1:]])
            y_data.append(int(row[0]))
    return np.array(x_data) / 255, np.array(y_data)


x_train, y_train = get_data('mnist_train.csv')[:10000]
x_test, y_test = get_data('mnist_test.csv')[:100]


unique = np.unique(y_test)

def p(x0, x):
    return np.sum(np.abs(x0 - x), axis=1)

K = lambda x0, x: np.exp(-p(x0, x) ** 2 / 2) / (2 * np.pi) ** 0.5

a = np.array([K(x0, x_train) for x0 in x_test])

b = np.array([np.sum((y_train == i) * a, axis=1) for i in unique])

predict = np.argmax(b, axis=0).astype(float)

print(predict[100])
print(y_test[100])
