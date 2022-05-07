import sys
sys.path.append("..")
import numpy as np
from ex1.ex1_2 import linesearch_secant

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

def hestenes_stiefel(g_k,g_k1,d_k):
    return np.sum(g_k1*(g_k1-g_k))/np.sum(d_k*(g_k1-g_k))

def polak_ribiere(g_k,g_k1):
    return np.sum(g_k1*(g_k1-g_k))/np.sum(g_k*g_k)

def fletcher_reeves(g_k,g_k1):
    return np.sum(g_k1*g_k1)/np.sum(g_k*g_k)

def conjugate_gradient(f,x_0,cal_beta='Hestenes-Stiefel'):
    cnt = 0
    x = x_0
    g_k = grad(f, x_0)
    if abs(g_k[0])<1e-9 and abs(g_k[1])<1e-9:
        return x, cnt
    d = -1 * g_k
    while True:
        alpha = linesearch_secant(grad, f, x, d)
        x = x + alpha*d
        g_k1 = grad(f, x)
        if abs(g_k1[0])<1e-9 and abs(g_k1[1])<1e-9:
            return x, cnt
        if cal_beta =='Hestenes-Stiefel':
            beta = hestenes_stiefel(g_k, g_k1, d)
        elif cal_beta == 'Polak-Ribiere':
            beta = polak_ribiere(g_k, g_k1)
        elif cal_beta == 'Fletcher-Reeves':
            beta = fletcher_reeves(g_k, g_k1)
        d = -1 * g_k1 + beta*d
        g_k = g_k1

        cnt+=1
        if cnt%6==0:
            d = -1 * g_k

x_0 = np.array([-2, 2])

result1 = conjugate_gradient(f,x_0,cal_beta='Hestenes-Stiefel')
result2 = conjugate_gradient(f,x_0,cal_beta='Polak-Ribiere')
result3 = conjugate_gradient(f,x_0,cal_beta='Fletcher-Reeves')
print('使用Hestenes-Stiefel公式  ' + '结果: ', result1[0], '迭代次数: ', result1[1])
print('使用Polak-Ribiere公式  ' + '结果: ', result2[0], '迭代次数: ', result2[1])
print('使用Fletcher-Reeves公式  ' + '结果: ', result3[0], '迭代次数: ', result3[1])