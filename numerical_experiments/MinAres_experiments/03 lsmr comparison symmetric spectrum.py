import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from MinAres import MinAres
from numerical_experiments.adapted_methods import lsmr
from numerical_experiments.util import optimal_backward_error_fro


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    }
)

np.random.seed(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.95)


n = 40

A_1 = np.diag(
    (
        *np.random.normal(loc=1, scale=0.1, size=n // 2),
        *np.random.normal(loc=-1, scale=0.1, size=n // 2),
        0,
    )
)

A_2 = np.diag(
    (
        *(np.diagonal(A_1)[: n // 2] + 1),
        *np.diagonal(A_1)[n // 2 : n],
        0,
    )
)

b = np.ones(n + 1)

norm_A_1_fro = norm(A_1, ord="fro")
norm_A_2_fro = norm(A_2, ord="fro")

Aresidual_norms_minares_1 = []
Aresidual_norms_minares_2 = []


def append_Ares_minares(x, k, norm_r_k, norm_Ar_k, A, b, l):
    res = b - A @ x
    l.append(norm(A @ res))


MinAres(
    A_1,
    b,
    r_tol=1e-15,
    Ar_tol=1e-15,
    callback=append_Ares_minares,
    callback_args=(A_1, b, Aresidual_norms_minares_1),
)
ax1.semilogy(
    np.arange(len(Aresidual_norms_minares_1)),
    Aresidual_norms_minares_1 / Aresidual_norms_minares_1[0],
    "-",
    markersize=3,
    label="\\textsc{MinAres}",
)

MinAres(
    A_2,
    b,
    r_tol=1e-15,
    Ar_tol=1e-15,
    callback=append_Ares_minares,
    callback_args=(A_2, b, Aresidual_norms_minares_2),
)
ax2.semilogy(
    np.arange(len(Aresidual_norms_minares_2)),
    Aresidual_norms_minares_2 / Aresidual_norms_minares_2[0],
    "-",
    markersize=3,
    label="\\textsc{MinAres}",
)


Aresidual_norms_lsmr_1 = [norm(A_1 @ b)]


def append_residuals_lsmr_1(x):
    res = b - A_1 @ x
    Aresidual_norms_lsmr_1.append(norm(A_1 @ res))


lsmr(
    A_1,
    b,
    atol=1e-15,
    btol=1e-15,
    callback=append_residuals_lsmr_1,
)
ax1.semilogy(
    np.arange(len(Aresidual_norms_lsmr_1)),
    Aresidual_norms_lsmr_1 / Aresidual_norms_lsmr_1[0],
    ":",
    markersize=3,
    label="\\textsc{Lsmr}",
)


Aresidual_norms_lsmr_2 = [norm(A_2 @ b)]


def append_residuals_lsmr_2(x):
    res = b - A_2 @ x
    Aresidual_norms_lsmr_2.append(norm(A_2 @ res))


lsmr(
    A_2,
    b,
    atol=1e-15,
    btol=1e-15,
    callback=append_residuals_lsmr_2,
)
ax2.semilogy(
    np.arange(len(Aresidual_norms_lsmr_2)),
    Aresidual_norms_lsmr_2 / Aresidual_norms_lsmr_2[0],
    ":",
    markersize=3,
    label="\\textsc{Lsmr}",
)

ax1.set_xlabel("$k$", loc="right")
ax2.set_xlabel("$k$", loc="right")

ax1.set_title("Relative $A_1$-Residual")
ax2.set_title("Relative $A_2$-Residual")

ax1.set_ylabel("$\\Vert A_1r_k\\Vert \\mathbin{/} \\Vert A_1b\\Vert$")
ax2.set_ylabel("$\\Vert A_2r_k\\Vert \\mathbin{/} \\Vert A_2b\\Vert$")

ax1.legend()
ax2.legend()
plt.show()
