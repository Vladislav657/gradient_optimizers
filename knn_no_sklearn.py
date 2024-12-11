import numpy as np
import csv


def get_data(filename):
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x_data.append([int(pixel)for pixel in row[1:]])
            y_data.append(int(row[0]))
    return np.array(x_data) / 255, np.array(y_data)


def r(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def k_nearest_neighbors(k):
    res = []

    for x_tst in x_test:
        r_array = [r(x_tst, x_trn) for x_trn in x_train]
        sorted_indices = np.argsort(r_array)[:k]
        nearest_neighbors = [y_train[i] for i in sorted_indices]
        counts_max = np.bincount(nearest_neighbors).argmax()
        res.append(counts_max)

    return np.array(res)


x_train, y_train = get_data('mnist_train.csv')
x_test, y_test = get_data('mnist_test.csv')

predict = k_nearest_neighbors(5)
print("pred:", predict[:100])
print("real:", y_test[:100])
