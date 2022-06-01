import numpy as np
from utils import read_data, M, F, Jacobian
import matplotlib.pyplot as plt


def LM(x, data, tao, eps_1, eps_2, k_max):
    k = 0
    niu = 2
    J = Jacobian(x, data)
    A = (J.T).dot(J)
    g = (J.T).dot(data[:, 1] - M(x, data)).reshape(-1, 1)
    found = np.max(g) <= eps_1
    miu = tao * A.max()
    while not found and k < k_max:
        print(x)
        k += 1
        h = (np.linalg.inv(A + miu * np.eye(A.shape[0])).dot(-g)).reshape(-1, 1)
        if np.linalg.norm(h) <= eps_2 * (np.linalg.norm(x) + eps_2):
            found = True
        else:
            x_new = x + h
            rou = (F(x, data) - F(x_new, data)) / (1 / 2 * (h.T).dot(miu * h - g))
            if rou > 0:
                x = x_new
                print(x)
                J = Jacobian(x, data)
                A = (J.T).dot(J)
                g = (J.T).dot(data[:, 1] - M(x, data)).reshape(-1, 1)
                found = np.max(g) <= eps_1
                miu = miu * max(1 / 3, 1 - (2 * rou - 1) ** 3)
                niu = 2
            else:
                miu = miu * niu
                niu = 2 * niu
    return k, x


def main():
    data = read_data()

    # super parameters
    tao = 1e-3
    eps_1 = 1e-8
    eps_2 = 1e-8
    k_max = 200

    x = np.zeros((4, 1))
    x[0, 0] = -1
    x[1, 0] = -2
    x[2, 0] = 1
    x[3, 0] = -1

    times, result = LM(x, data, tao, eps_1, eps_2, k_max)
    t = np.arange(0, 100) * 0.01

    plt.scatter(data[:, 0], data[:, 1], color="b")
    plt.plot(
        t,
        result[2, 0] * np.exp(result[0, 0] * t)
        + result[3, 0] * np.exp(result[1, 0] * t),
        "r",
    )
    print('迭代次数：', times)
    print('求解参数：\n', result)
    plt.savefig("lm.png")
    

if __name__ == "__main__":
    main()
