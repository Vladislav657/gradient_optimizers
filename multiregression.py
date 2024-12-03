import csv


def get_data(filename):
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in list(reader)[1:]:
            x_data.append([float(x) for x in row[:-1]])
            y_data.append(float(row[-1]))
    return x_data, y_data


def det(m):
    if len(m) != len(m[0]):
        raise ValueError("Matrix must be square")
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    res = 0
    for i in range(len(m)):
        submatrix = []
        for j in range(1, len(m)):
            submatrix.append([])
            for k in range(len(m)):
                if k != i:
                    submatrix[j - 1].append(m[j][k])
        res += m[0][i] * det(submatrix) * (-1) ** i
    return res


x_train, y_train = get_data("house_price_regression_dataset.csv")
x_len = len(x_train[0])


X = [[0 for _ in range(x_len)] for i in range(x_len)]

X[0][0] = len(y_train)

for i in range(1, x_len):
    X[0][i] = sum(x[i - 1] for x in x_train)

for i in range(1, x_len):
    for j in range(x_len):
        if j < i:
            X[i][j] = X[j][i]
        else:
            X[i][j] = sum(x[i - 1] * x[j - 1] for x in x_train)

Y = [0 for _ in range(x_len)]
Y[0] = sum(y_train)
for i in range(1, x_len):
    Y[i] = sum(y_train[j] * x_train[j][i - 1] for j in range(len(x_train)))


detX = det(X)
W = []


for i in range(x_len):
    X_i = [row.copy() for row in X]
    for j in range(x_len):
        X_i[j][i] = Y[j]
    W.append(det(X_i) / detX)


def regression(x, w):
    x = [1, *x]
    return sum(x * w for x, w in zip(x, w))


print("real:", end='\t\t')
print(*[round(y, 2) for y in y_train], sep='\t')
print("predicted:", end='\t')
print(*[round(regression(x, W), 2) for x in x_train], sep='\t')
