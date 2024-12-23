import numpy as np
from math import factorial


lmd = 4
mu_1 = 2
mu_2 = 3

p = lmd / (mu_1 * mu_2) ** 0.5

N = 1000

clients = np.random.exponential(1 / lmd, N)

proc_1 = np.random.exponential(1 / mu_1, N)
proc_2 = np.random.exponential(1 / mu_2, N)

client_time = [clients[0]]
for i in range(1, N):
    client_time.append(client_time[i-1] + clients[i])
client_time = np.array(client_time)


def smo_no_queue():
    T0 = T1 = T2 = 0

    time_1 = 0
    time_2 = 0

    for i in range(N):
        if client_time[i] >= time_1 and client_time[i] >= time_2:
            T0 += client_time[i] - max(time_1, time_2)
            T1 += max(time_1, time_2) - min(time_1, time_2)
            time_1 = time_2 = client_time[i]
            time_1 += proc_1[i]

        elif time_1 >= client_time[i] >= time_2:
            T1 += client_time[i] - time_2
            time_2 = client_time[i] + proc_2[i]
            T2 += min(time_1, time_2) - client_time[i]

        elif time_2 >= client_time[i] >= time_1:
            T1 += client_time[i] - time_1
            time_1 = client_time[i] + proc_1[i]
            T2 += min(time_1, time_2) - client_time[i]

    T = sum(clients)

    print('----No Queue----')
    print("P0 =", T0 / T)
    print("P1 =", T1 / T)
    print("P2 =", T2 / T)


def smo_queue():
    T0 = T1 = T2 = 0

    time_1 = 0
    time_2 = 0

    for i in range(N):
        if client_time[i] >= time_1 and client_time[i] >= time_2:
            T0 += client_time[i] - max(time_1, time_2)
            T1 += max(time_1, time_2) - min(time_1, time_2)
            time_1 = time_2 = client_time[i]
            time_1 += proc_1[i]

        elif time_1 >= client_time[i] >= time_2:
            T1 += client_time[i] - time_2
            time_2 = client_time[i] + proc_2[i]
            T2 += min(time_1, time_2) - client_time[i]

        elif time_2 >= client_time[i] >= time_1:
            T1 += client_time[i] - time_1
            time_1 = client_time[i] + proc_1[i]
            T2 += min(time_1, time_2) - client_time[i]

        else:
            if time_1 > time_2:
                T2 += min(time_1, time_2 + proc_2[i]) - time_2
                time_2 += proc_2[i]
            else:
                T2 += min(time_2, time_1 + proc_1[i]) - time_1
                time_1 += proc_1[i]

    T = sum(clients)

    print('----Queue----')
    print("P0 =", T0 / T)
    print("P1 =", T1 / T)
    print("P2 =", T2 / T)


def erlang():
    print('----Erlang----')
    print("P0 =", 1 / sum(p ** i / factorial(i) for i in range(3)))
    print("P1 =", p / sum(p ** i / factorial(i) for i in range(3)))
    print("P2 =", p * p / 2 / sum(p ** i / factorial(i) for i in range(3)))


smo_no_queue()
print()
smo_queue()
print()
erlang()
