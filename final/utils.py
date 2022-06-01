import numpy as np


def read_data():
    f = open(r"efit1.dat", encoding="utf-8")
    raw_data = []
    for line in f:
        raw_data.append(line.strip().split("\t"))

    data = []
    for line in raw_data[3:]:
        data.append([float(line[0].split("    ")[0]), float(line[0].split("    ")[1])])

    data = np.array(data)
    return data


def M(x, data):
    return x[2, 0] * np.exp(x[0, 0] * data[:, 0]) + x[3, 0] * np.exp(
        x[1, 0] * data[:, 0]
    )


def F(x, data):
    return 1 / 2 * np.linalg.norm(data[:, 1] - M(x, data)) ** 2


def partial_derivative(x, data, i):
    very_small = 1e-9
    x_left = x.copy()
    x_right = x.copy()
    x_left[i, 0] -= very_small
    x_right[i, 0] += very_small

    return (M(x_right, data) - M(x_left, data)) / (2 * very_small)


def Jacobian(x, data):
    J = np.zeros((data.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        J[:, i] = partial_derivative(x, data, i)

    # 因为此处函数M作为被减数，所以要取负号
    return -J
