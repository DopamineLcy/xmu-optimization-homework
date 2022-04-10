import numpy as np

from ex1_2 import linesearch_secant


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
eps = 1e-4

while True:
    d = -1 * grad(f, x)
    print(np.linalg.norm(-d, ord = 2))
    if np.linalg.norm(-d, ord = 2)<eps:
        break
    alpha = linesearch_secant(grad, f, x, d)
    x = x + alpha*d
    print(x,f(x))
    
print('solution: ',x)