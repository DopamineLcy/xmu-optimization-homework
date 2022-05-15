from errno import ENETRESET
import numpy as np

# 16.21
A = np.array(
    [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
    ]
)

b = np.array(
    [
        [4],
        [6],
        [8],
    ]
)

c = np.array(
    [
        [-2],
        [-5],
        [0],
        [0],
        [0],
    ]
)


v = [2,3,4]


def simplex(A, b, c, v):
    cB = c[v]
    r = c.T-np.dot(cB.T,A)
    corner = np.zeros((1,1))
    while True:
        if (r>=0).sum() == r.shape[1]:
            break
        
        r_min = r.min()
        r_min_index = np.where(r==r_min)[1][0]

        enter_vector = A[:,r_min_index]
        division = (b.T/enter_vector)[0]
        division_min = division.min()
        division_min_index = np.where(division==division_min)[0][0]
        q,p = r_min_index,division_min_index
        flag = A[p][q]
        
        Abc = np.concatenate([np.concatenate([A,b],axis=1),np.concatenate([r,corner],axis=1)],axis = 0)
        new_Abc = np.zeros_like(Abc)
        for i in range(0,Abc.shape[0]):
            for j in range(0,Abc.shape[1]):
                if i==p:
                    new_Abc[p][j] = Abc[p][j] / flag
                else:
                    new_Abc[i][j] = Abc[i][j] - Abc[p][j]/flag*Abc[i][q]

        A = new_Abc[0:A.shape[0],0:A.shape[1]]
        b = new_Abc[:-1,-1].reshape((b.shape[0],b.shape[1]))
        r = new_Abc[-1,:-1].reshape((r.shape[0],r.shape[1]))
        corner[0][0] = new_Abc[-1][-1]
        v[p] = q
        print(new_Abc)
        print('--------')
    print(1)




simplex(A,b,c,v)