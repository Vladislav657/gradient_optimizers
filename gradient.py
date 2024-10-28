import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('TkAgg')


def func(x):
    return 0.6 * x ** 2 - 0.3 * np.sin(5*x)


# здесь объявляйте функцию df (производную) и продолжайте программу
def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4*x)


lmd = 0.02
x0 = -1.5
N = 100

# x = x0
# for _ in range(N):
#     x -= 0.01 * df(x)
# print(x)

X = np.arange(-2, 1, 0.01)
Y = np.array([func(x) for x in X])

plt.ion()
plt.plot(X, Y)
dot, = plt.plot(x0, func(x0), 'ro')

x = x0
for _ in range(N):
    dot.set_data([x], [func(x)])
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.02)

    x -= lmd * df(x)

plt.ioff()
# print(x)
