from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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


x_train, y_train = get_data('mnist_train.csv')
x_test, y_test = get_data('mnist_test.csv')

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(x_train, y_train)

predict = knc.predict(x_test)

print(predict[:100])
print(y_test[:100])
print(f"accuracy = {accuracy_score(y_test, predict)}")
