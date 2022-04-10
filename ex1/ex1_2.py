import numpy as np


def linesearch_secant(grad, f, x, d):
    def iter_dfx(dfx_k2):
        return dfx_k2, np.sum(grad(f, x + alpha_k2 * d) * d)

    def iter_alpha(alpha_k1, alpha_k2):
        return alpha_k2, (alpha_k1 * dfx_k2 - alpha_k2 * dfx_k1) / (dfx_k2 - dfx_k1)

    eps = 1e-4
    alpha_k1 = 0
    alpha_k2 = 0.0001
    dfx_k1 = np.sum(grad(f, x) * d)
    dfx_k2 = dfx_k1

    while True:
        if abs(np.sum(grad(f, x + alpha_k2 * d) * d)) <= abs(eps * np.sum(grad(f, x) * d)):
            break
        dfx_k1, dfx_k2 = iter_dfx(dfx_k2)
        alpha_k1, alpha_k2 = iter_alpha(alpha_k1, alpha_k2)

    return alpha_k2

def f(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def grad(f, x):
    very_small = 1e-9
    grad = []
    grad.append(
        (f([x[0] + very_small, x[1]]) - f([x[0] - very_small, x[1]])) / (very_small * 2)
    )
    grad.append(
        (f([x[0], x[1] + very_small]) - f([x[0], x[1] - very_small])) / (very_small * 2)
    )

    return np.array(grad)

x = np.array([-2, 2])
d = -1 * grad(f, x)

print("alpha: ", linesearch_secant(grad, f, x, d))
