import numpy as np


def g(x):
    return (2 * x - 1) ** 3 + 4 * (4 - 1024 * x) ** 3


def iter(x_k1, x_k2):
    return x_k2 - ((x_k2 - x_k1) / (g(x_k2) - g(x_k1))) * g(x_k2)


x_k1 = 0
x_k2 = 1
eps = 1e-5
while True:
    if abs(x_k1 - x_k2) < abs(x_k1) * eps:
        break
    x_k1, x_k2 = x_k2, iter(x_k1, x_k2)
    print(x_k1, x_k2)

print("solution:", x_k1)
print("g(x): ", g(x_k1))
