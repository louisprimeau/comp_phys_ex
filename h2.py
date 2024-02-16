

import numpy as np
import scipy
from numpy import pi, exp, sqrt
from numpy.linalg import norm
from scipy.special import erf

a = [38.474970, 5.782948, 1.242567, 0.298073]

def basis_function(i, center):
    def func(r):
        return exp(-a[i]*(center - r)**2)
    return func

# Make S:
def overlap_integral(i, j):
    return (pi / (a[i] + a[j]))**(3/2)
    
def kinetic_integral(i, j):
    return 3 * a[i] * a[j] * pi**(3/2) / (a[i] + a[j])**(5/2)

def coulomb_integral(i, j):
    return - 4 * pi / (a[i] + a[j])

def ee_coulomb(p, r, q, s):
    return 2 * pi**(5/2) / ((a[p] + a[q]) * (a[r] + a[s]) * sqrt(a[p] + a[r] + a[q] + a[s]))

S = np.zeros((4, 4))
h = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        S[i,j] = overlap_integral(i, j)
        h[i,j] = kinetic_integral(i, j) + coulomb_integral(i, j)

Q = np.zeros((4, 4, 4, 4))
for p in range(4):
    for r in range(4):
        for q in range(4):
            for s in range(4):                
                Q[p,r,q,s] = ee_coulomb(p, r, q, s)
                
c = np.array([1, 1, 1, 1])
c = c / np.sqrt(np.einsum('p,pq,q', c, S, c))
for i in range(10):

    F = h + np.einsum('prqs,r,s', Q, c, c)
    eigs, eigv = scipy.linalg.eig(F, b=S)
    print(eigs)
    idxs = np.argsort(eigs)
    eigs = eigs[idxs]; eigv = eigv[:, idxs]
    c = eigv[:, 0]
    c = c / np.sqrt(np.einsum('p,pq,q', c, S, c))
    E = 2 * np.einsum('p,q,pq', c,c,h) + np.einsum('prqs,p,q,r,s', Q, c, c, c, c)
    print(E)
