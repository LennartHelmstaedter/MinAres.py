import numpy as np


def pad_matrix_with_zeros(A):
    return np.pad(A, ((0, 1), (0, 1)), "constant", constant_values=0)


def print_matrx_for_julia(A):
    n, m = A.shape
    print("[", end="")
    for i in range(n):
        for j in range(m):
            print(A[i, j], end="")
            if j < m - 1:
                print(" ", end="")
        if i < n - 1:
            print(";")
    print("]")
