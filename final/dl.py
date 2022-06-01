import numpy as np
from utils import read_data, M, F, Jacobian
import matplotlib.pyplot as plt


def DL(x, data, delta, eps_1, eps_2, eps_3, k_max):
    k = 0
    J = Jacobian(x, data)
    f = (data[:, 1] - M(x, data)).reshape(-1, 1)
    g = (J.T).dot(f).reshape(-1, 1)
    found = np.max(f) <= eps_3 or np.max(g) <= eps_1
    while not found and k < k_max:
        k += 1
        alpha = np.linalg.norm(g) ** 2 / np.linalg.norm(J.dot(g)) ** 2
        h_sd = -g
        h_gn = np.linalg.inv(J.T.dot(J)).dot(-J.T).dot(f)

        if np.linalg.norm(h_gn) <= delta:
            h_dl = h_gn
        elif alpha * np.linalg.norm(h_sd) >= delta:
            h_dl = (delta / np.linalg.norm(h_sd)) * h_sd
        else:
            a = alpha * h_sd
            b = h_gn
            c = a.T.dot(b - a)
            if c <= 0:
                beta = (
                    -c
                    + np.sqrt(
                        c * c
                        + np.linalg.norm(b - a) ** 2
                        * (delta * delta - np.linalg.norm(a) ** 2)
                    )
                ) / np.linalg.norm(b - a) ** 2
            else:
                beta = (delta * delta - np.linalg.norm(a) ** 2) / (
                    c
                    + np.sqrt(
                        c * c
                        + np.linalg.norm(b - a) ** 2
                        * (delta * delta - np.linalg.norm(a) ** 2)
                    )
                )

            h_dl = alpha * h_sd + beta * (h_gn - alpha * h_sd)

        if np.linalg.norm(h_dl) <= eps_2 * (np.linalg.norm(x) + eps_2):
            found = True
        else:
            x_new = x + h_dl
            L0_minus_Lh_dl = (
                1 / 2 * (np.linalg.norm(f) ** 2)
                - 1 / 2 * np.linalg.norm(f + J.dot(h_dl)) ** 2
            )
            rou = (F(x, data) - F(x_new, data)) / L0_minus_Lh_dl
            if rou > 0:
                x = x_new
                J = Jacobian(x, data)
                f = (data[:, 1] - M(x, data)).reshape(-1, 1)
                g = (J.T).dot(f).reshape(-1, 1)
                found = np.max(f) <= eps_3 or np.max(g) <= eps_1
            if rou > 0.75:
                delta = max(delta, 3 * np.linalg.norm(h_dl))
            elif rou < 0.25:
                delta /= 2
                found = delta <= eps_2 * (np.linalg.norm(x) + eps_2)
    return k, x


def main():
    data = read_data()

    # super parameters
    delta = 1
    eps_1 = 1e-8
    eps_2 = 1e-8
    eps_3 = 1e-8
    k_max = 200

    x = np.zeros((4, 1))
    x[0, 0] = -1
    x[1, 0] = -2
    x[2, 0] = 1
    x[3, 0] = -1

    times, result = DL(x, data, delta, eps_1, eps_2, eps_3, k_max)
    t = np.arange(0, 100) * 0.01

    plt.scatter(data[:, 0], data[:, 1], color="b")
    plt.plot(
        t,
        result[2, 0] * np.exp(result[0, 0] * t)
        + result[3, 0] * np.exp(result[1, 0] * t),
        "r",
    )
    plt.savefig("dl.png")
    print('迭代次数：', times)
    print('求解参数：\n', result)


if __name__ == "__main__":
    main()
