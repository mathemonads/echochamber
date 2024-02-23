#!/usr/bin/env python

import numpy as np
np.set_printoptions(precision=2, suppress=True)

def svd(matrix, threshold=0.008):
    U, S, Vt = np.linalg.svd(matrix)
    I = np.where(S > threshold)[0]
    return U[:,I], np.diag(S[I]), Vt[I,:]


L = np.array([
    [1.25, 0.83, 0, -0.12],
    [1.05, 1.13, 0.35, np.nan],
    [1.12, 1.02, 0.21, np.nan],
    [1.57, 0.35, -0.56, np.nan],
    [np.nan, 0.18, 1.02, 0.98]
])
L = L.T
L = np.delete(L, -1, axis=0)
L = np.delete(L, -1, axis=1)
print(L)

r = 2
c = 1

U, S, Vt = svd(L)
print(U.shape, S.shape, Vt.shape)
print(U)
print(S)
print(Vt)

A = np.dot(U, np.dot(S, Vt))
print(A)
print(np.max(np.abs(L-A)))
