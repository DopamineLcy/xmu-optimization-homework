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

def rank1(H,delta_x,delta_g):
    numerator = np.dot(np.array([delta_x - np.dot(H,delta_g)]).T, np.array([delta_x-np.dot(H,delta_g)]))
    denominator = np.sum(delta_g*(delta_x-np.dot(H,delta_g)))
    return H + numerator/denominator

def DFP(H,delta_x,delta_g):
    first_item = np.dot(np.array([delta_x]).T, np.array([delta_x]))/np.sum(delta_x * delta_g)
    second_item = np.dot(np.array([np.dot(H,delta_g)]).T, np.array([np.dot(H,delta_g)]))/np.sum(delta_g*np.dot(H,delta_g))
    return H + first_item - second_item

def BFGS(H,delta_x,delta_g):
    first_item = (1+np.sum(delta_g*np.dot(H,delta_g))/np.sum(delta_g*delta_x)) * (np.dot(np.array([delta_x]).T, np.array([delta_x]))/np.sum(delta_x*delta_g))
    second_item = (np.dot(np.array([np.dot(H,delta_g)]).T,np.array([delta_x])) + np.dot(np.array([np.dot(H,delta_g)]).T , np.array([delta_x])).T)/np.sum(delta_g * delta_x)
    return H + first_item - second_item

def quasi_newton(f,x_0,H_0,cal_H='rank1'):
    cnt = 0
    x = x_0
    H = H_0
    g_k = grad(f, x)
    while True:
        if abs(g_k[0])<1e-9 and abs(g_k[1])<1e-9:
            return x, cnt
        d = -np.dot(H, g_k)
        alpha = linesearch_secant(grad, f, x, d)
        x = x + alpha*d
        delta_x = alpha*d
        g_k1 = grad(f, x)
        delta_g = g_k1 - g_k
        if cal_H == 'rank1':
            H = rank1(H,delta_x,delta_g)
        elif cal_H == 'DFP':
            H = DFP(H,delta_x,delta_g)
        if cal_H == 'BFGS':
            H = BFGS(H,delta_x,delta_g)

        if abs(g_k1[0])<1e-9 and abs(g_k1[1])<1e-9:
            return x, cnt
        g_k = g_k1
        
        cnt+=1
        if cnt%6==0:
            d = -1 * g_k

x_0 = np.array([-2, 2])
H_0 = np.array([[1,0],[0,1]])
result1 = quasi_newton(f, x_0, H_0, cal_H='rank1')
result2 = quasi_newton(f, x_0, H_0, cal_H='DFP')
result3 = quasi_newton(f, x_0, H_0, cal_H='BFGS')
# print(result1,result2,result3)
print('使用秩1算法  ' + '结果: ', result1[0], '迭代次数: ', result1[1])
print('使用DFP算法  ' + '结果: ', result2[0], '迭代次数: ', result2[1])
print('使用BFGS算法  ' + '结果: ', result3[0], '迭代次数: ', result3[1])