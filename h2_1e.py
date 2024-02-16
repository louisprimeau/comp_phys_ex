

import numpy as np
import scipy
from numpy import pi, exp, sqrt
from numpy.linalg import norm
from scipy.special import erf

a = [13.00773, 1.962079, 0.444529, 0.1219492]

R_A = np.array([0, 0, 0.5])
R_B = np.array([0, 0, -0.5])

def basis_function(i, center):
    def func(r):
        return exp(-a[i]*(center - r)**2)
    return func

# Make S:

def overlap_integral(i, j, R_A, R_B):
    g = a[i] * a[j] / (a[i] + a[j])
    d2 = norm(R_A - R_B)**2
    return (pi / (a[i] + a[j]))**(3/2) * exp(-g * d2)
    
S = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        S[i,j] = overlap_integral(i, j, R_A, R_A)

# Make T:

def kinetic_integral(i, j, R_A, R_B):
    g = a[i] * a[j] / (a[i] + a[j])
    d2 = norm(R_A - R_B)**2
    return 0.5 * g * (6 - 4 * g * d2) * overlap_integral(i, j, R_A, R_B)

T = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        T[i,j] = kinetic_integral(i, j, R_A, R_A)

# Make A:

def F0(t):
    return erf(sqrt(t)) * sqrt(pi) / (2 * sqrt(t)) if t != 0 else 1

def coulomb_integral(i, j, R_A, R_B, R_C):
    g = a[i] * a[j] / (a[i] + a[j])
    d2 = norm(R_A - R_B)**2
    R_P = (a[i] * R_A + a[j] * R_B) / (a[i] + a[j])
    t = norm(R_P - R_C)**2 * (a[i] + a[j])
    return -2 * pi * 1 / (a[i] + a[j]) * exp(-g * d2) * F0(t)

A = np.zeros((8, 8))

for i in range(4):
    for j in range(4):
        A[i,j] = coulomb_integral(i, j, R_A, R_A, R_A) + coulomb_integral(i, j, R_A, R_A, R_B) 
        A[i+4,j+4] = coulomb_integral(i, j, R_B, R_B, R_A) + coulomb_integral(i, j, R_B, R_B, R_B)
        A[i+4,j] = coulomb_integral(i, j, R_B, R_A, R_A) + coulomb_integral(i, j, R_B, R_A, R_B)
        A[i,j+4] = coulomb_integral(i, j, R_A, R_B, R_A) + coulomb_integral(i, j, R_A, R_B, R_B)

A = np.zeros((8, 8, 8, 8))
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                

eigs, eigv = scipy.linalg.eig(T + A, b=S)
print(np.sort(eigs))
