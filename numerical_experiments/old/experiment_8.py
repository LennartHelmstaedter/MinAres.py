import numpy as np
from numpy.linalg import norm
from numerical_experiments.adapted_methods import MinAres_t
import matplotlib.pyplot as plt


n = 200


lambda_min = 1
lambda_max = 1000
gamma = 0.65


A = np.diag(
    np.append(
        lambda_min
        + (lambda_max - lambda_min)
        * (np.arange(n) / (n - 1))
        * gamma ** (n - np.arange(n) - 1),
        0,
    )
)

# A = np.diag(
#     (
#         *np.random.uniform(1, 100, int(np.ceil(n / 2))),
#         *np.random.uniform(-100, 1, int(np.floor(n / 2))),
#         0,
#     )
# )

# import scipy

# a = scipy.io.loadmat("./numerical_experiments/nasa1824", spmatrix=False)
# A = a["Problem"][0][0][1]
# n = A.shape[0] - 1

b = np.ones(n + 1)
# b[-1] = 0


def append_norms(x_k, t_k, k, norm_r_k, norm_Ar_k, lx, lt):
    lx.append(norm(x_k))
    lt.append(norm(t_k))


x_norms = []
t_norms = []

MinAres_t(A, b, callback=append_norms, callback_args=(x_norms, t_norms))

plt.plot(np.arange(len(x_norms)), x_norms, "o")
plt.plot(np.arange(len(x_norms)), t_norms, "x")

plt.show()
