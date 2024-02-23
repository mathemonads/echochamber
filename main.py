#!/usr/bin/env python

from sklearn.linear_model import LinearRegression

import numpy as np
np.set_printoptions(precision=2, suppress=True)

def svd(matrix, threshold=0.008):
    U, S, Vt = np.linalg.svd(matrix)
    I = np.where(S > threshold)[0]
    return U[:,I], np.diag(S[I]), Vt[I,:]


# L = np.array([
#     [1.25, 0.83, 0, -0.12],
#     [1.05, 1.13, 0.35, np.nan],
#     [1.12, 1.02, 0.21, np.nan],
#     [1.57, 0.35, -0.56, np.nan],
#     [np.nan, 0.18, 1.02, 0.98]
# ])
# L = L.T
L = np.array([
    [ 1.25,  1.05,  1.12,  1.57,   np.nan],
    [ 0.83,  1.13,  1.02,  0.35,  0.18],
    [ 0.  ,  0.35,  0.21, -0.56,  1.02],
    [-0.12,   np.nan,   np.nan,   np.nan,  0.98]
])
L = L[:-1, :-1]

r = 1
c = 2

instances = 40000
col = L[:,c]
chamber = np.tile(col[:,np.newaxis], (1,instances))
chamber = chamber + np.random.normal(0, 0.01, chamber.shape)
L = np.hstack((L, chamber))
L[r,c] = np.nan
col = L[:,c]
minor = np.delete(L, c, axis=1)
print("L:")
print(L)
print()

print(minor)
U, S, Vt = svd(minor)
non_zero = np.nonzero(S)[0]

non_nan = ~np.isnan(L[:,c])
print(L[:,c])
A =  np.dot(U, np.dot(S, Vt))
L = np.nan_to_num(L, nan=0.0)
k = L[:,c]

mk = k[non_nan]
mA = A[non_nan, :]
distances = np.linalg.norm(mA - mk[:,np.newaxis], axis=0)
nearest_c = np.argmin(distances)
nearest_neighbor = A[:, nearest_c]
print("Neighbor:",nearest_neighbor)

estimate = nearest_neighbor + (L[:,c] - nearest_neighbor) * k / np.linalg.norm(k)
print("Estimate:", estimate)

print("A shape:", A.shape)
print("k shape:", k.shape)
Columns = mA
print("Columns shape:", Columns.shape)
print("mk shape:", mk.shape)
coefficients, _, _, _ = np.linalg.lstsq(Columns, mk, rcond=None)
print("Coefficients:", coefficients)
combo = np.dot(A, coefficients)
print("Combo:", combo)
estimate = combo + (L[:,c] - combo) * k / np.linalg.norm(k)
print("Estimate:", estimate)

Columns = np.dot(U, S)
Columns_inv = np.linalg.pinv(Columns)
print(S)
