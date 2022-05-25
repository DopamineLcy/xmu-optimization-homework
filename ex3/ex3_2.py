import numpy as np
from ex3_1 import simplex


def two_stages_simplex(A, b, c):
    A_extended = np.concatenate([A, np.eye(len(b))], axis=1)
    c_tmp = np.zeros((A_extended.shape[1], 1))
    for i in range(1, len(b) + 1):
        c_tmp[-i, 0] = 1

    index = []
    base = len(c)
    for i in range(len(b)):
        index.append(base)
        base += 1
    b, A, index = simplex(A_extended, b, c_tmp, index)
    A = A[:, : -len(b)]
    result, _, index = simplex(A, b, c, index)
    return result, index


A = np.array(
    [
        [1, 1, 1, 0],
        [5, 3, 0, -1],
    ]
)

b = np.array(
    [
        [4],
        [8],
    ]
)

c = np.array(
    [
        [-3],
        [-5],
        [0],
        [0],
    ]
)

result, index = two_stages_simplex(A, b, c)
print(result,index)
