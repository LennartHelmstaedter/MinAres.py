import numpy as np
from numpy.linalg import norm
from numerical_experiments.adapted_methods import steps_2_7
from MinAres import MinAres
import matplotlib.pyplot as plt


def jacobi(k, alpha, beta):
    T = np.zeros((k + 2, k + 1))
    T[np.arange(k + 1) + 1, np.arange(k + 1)] = beta
    T[np.arange(k + 1), np.arange(k + 1)] = alpha
    T[np.arange(k), np.arange(k) + 1] = beta
    return T @ T[: k + 1, :k]


# TT = jacobi(4, 2, 1)
# print(TT)

# n, m = TT.shape

# # print("{", end="")
# # for i in range(n):
# #     print("{", end="")
# #     for j in range(m):
# #         print(int(TT[i, j]), end="")
# #         if j < m - 1:
# #             print(",", end="")
# #     if i < n - 1:
# #         print("},", end="")
# # print("}}")

# print(np.linalg.pinv(TT))

# k = 1
# T = np.zeros((k + 2, k + 1))
# T[np.arange(k + 1) + 1, np.arange(k + 1)] = 1
# T[np.arange(k + 1), np.arange(k + 1)] = 2
# T[np.arange(k), np.arange(k) + 1] = 1
# TT = T @ T[: k + 1, :k]


def append_diff(x_k, k, norm_r, norm_Ar, diff, V, T, reduced_rhs):
    if k == 0:
        return
    x_svd = V[:, :k] @ (
        np.linalg.pinv(T[: k + 2, : k + 1] @ T[: k + 1, :k]) @ reduced_rhs[: k + 2]
    )
    diff.append(norm(x_k - x_svd) / norm(x_svd))


def print_x(x_k, k, norm_r, norm_Ar):
    print(k, x_k)


# fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))


# n = 200
# V = np.eye(n)

# alphas_ill = 2 * np.ones(n) + np.random.normal(0, 0.01, n)
# betas_ill = np.ones(n) + np.random.normal(0, 0.01, n)
# reduced_rhs_ill = np.zeros(n)
# reduced_rhs_ill[0] = betas_ill[0] * alphas_ill[0]
# reduced_rhs_ill[1] = betas_ill[0] * betas_ill[1]

# T_ill = np.zeros((n, n))
# T_ill[np.arange(n), np.arange(n)] = alphas_ill
# T_ill[np.arange(n - 1) + 1, np.arange(n - 1)] = betas_ill[1:]
# T_ill[np.arange(n - 1), np.arange(n - 1) + 1] = betas_ill[1:]

# print(np.linalg.cond(T_ill[:n, : n - 1]) * np.linalg.cond(T_ill[: n - 1, :n]))

# diff_ill = []

# steps_2_7(
#     np.eye(n),
#     alphas_ill,
#     betas_ill,
#     k_max=n - 4,
#     r_tol=-1,
#     Ar_tol=-1,
#     callback=append_diff,
#     callback_args=(diff_ill, V, T_ill, reduced_rhs_ill),
# )

# ax1.semilogy(np.arange(len(diff_ill)) + 1, diff_ill, label="ill-conditioned")


# alphas_well = 4 * np.ones(n)  # + np.random.normal(0, 0.01, n)
# betas_well = np.ones(n)  # + np.random.normal(0, 0.01, n)
# reduced_rhs_well = np.zeros(n)
# reduced_rhs_well[0] = betas_well[0] * alphas_well[0]
# reduced_rhs_well[1] = betas_well[0] * betas_well[1]

# T_well = np.zeros((n, n))
# T_well[np.arange(n), np.arange(n)] = alphas_well
# T_well[np.arange(n - 1) + 1, np.arange(n - 1)] = betas_well[1:]
# T_well[np.arange(n - 1), np.arange(n - 1) + 1] = betas_well[1:]

# print(np.linalg.cond(T_well[:n, : n - 1]) * np.linalg.cond(T_well[: n - 1, :n]))

# diff_well = []

# steps_2_7(
#     np.eye(n),
#     alphas_well,
#     betas_well,
#     k_max=n - 3,
#     r_tol=-1,
#     Ar_tol=-1,
#     callback=append_diff,
#     callback_args=(diff_well, V, T_well, reduced_rhs_well),
# )

# ax1.semilogy(np.arange(len(diff_well)) + 1, diff_well, label="well-conditioned")


# ax1.legend()
# plt.show()


n = 200
V = np.eye(n + 2, n)

cond = []
diff = []

gamma = 0.95

for alpha_param in 2 + (3 - 2) * (np.arange(100) / (99)) * gamma ** (
    100 - np.arange(100) - 1
):
    alphas = alpha_param * np.ones(n + 1) + np.random.normal(0, 0.001, n + 1)
    betas = np.ones(n + 2) + np.random.normal(0, 0.001, n + 2)

    reduced_rhs = np.zeros(n + 2)
    reduced_rhs[0] = betas[0] * alphas[0]
    reduced_rhs[1] = betas[0] * betas[1]

    T = np.zeros((n + 2, n + 1))
    T[np.arange(n + 1), np.arange(n + 1)] = alphas
    T[np.arange(n + 1) + 1, np.arange(n + 1)] = betas[1:]
    T[np.arange(n), np.arange(n) + 1] = betas[1 : n + 1]

    x_comp, info = steps_2_7(V, alphas, betas, k_max=n - 1, r_tol=-1, Ar_tol=-1)

    x_svd = V @ (np.linalg.pinv(T @ T[: n + 1, :n]) @ reduced_rhs)

    cond.append(np.linalg.cond(T) * np.linalg.cond(T[: n + 1, :n]))

    diff.append(norm(x_comp - x_svd) / norm(x_svd))

plt.loglog(cond, diff, "o")

plt.show()
