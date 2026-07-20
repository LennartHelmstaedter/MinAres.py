import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from numerical_experiments.MinAres_adapted_versions import MinAres_full_ortho


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": """
            \\usepackage{amsmath}
            \\DeclareMathOperator{\\dist}{dist}
        """,
    }
)


def append_forw_err(x, k, norm_r_k, norm_Ar_k, A, b, forw_err):
    res = b - A @ x
    forw_err.append(norm(res[:-1] / np.diagonal(A)[:-1]))


n = 25

lambda_min = 0.1
lambda_max = 100
gamma = 0.65

eigenvalues = lambda_min + (lambda_max - lambda_min) * (
    np.arange(n) / (n - 1)
) * gamma ** (n - np.arange(n) - 1)


A = np.diag((*eigenvalues, 0))
a = np.ones(n + 1)
norm_A_fro = norm(A, ord="fro")

delta = 1e-13

B = np.diag(np.append(eigenvalues[:-1], (lambda_max - delta, lambda_max, 0)))
b = np.ones(n + 2)
b[n - 1 : n + 1] = 1 / np.sqrt(2)
norm_B_fro = norm(B, ord="fro")

C = np.diag(
    np.append(eigenvalues[:-1], (*(lambda_max - delta * np.arange(4, -1, -1)), 0))
)
c = np.ones(n + 5)
c[n - 1 : n + 4] = 1 / np.sqrt(5)
norm_C_fro = norm(C, ord="fro")


fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
fig.subplots_adjust(left=0.2, right=0.95)


forw_err_A = []
MinAres_full_ortho(
    A,
    a,
    1e-13,
    1e-13,
    callback=append_forw_err,
    callback_args=(A, a, forw_err_A),
)

ax.semilogy(
    np.arange(len(forw_err_A)),
    forw_err_A / forw_err_A[0],
    "--",
    markersize=3,
    label="Single eigenvalue",
    color="tab:orange",
    zorder=1,
)


forw_err_B = []
MinAres_full_ortho(
    B,
    b,
    1e-13,
    1e-13,
    callback=append_forw_err,
    callback_args=(B, b, forw_err_B),
)

ax.semilogy(
    np.arange(len(forw_err_B)),
    forw_err_B / forw_err_B[0],
    ":",
    markersize=3,
    label="Cluster of size 2",
    zorder=2,
)


forw_err_C = []
MinAres_full_ortho(
    C,
    c,
    1e-13,
    1e-13,
    callback=append_forw_err,
    callback_args=(C, c, forw_err_C),
)

ax.semilogy(
    np.arange(len(forw_err_C)),
    forw_err_C / forw_err_C[0],
    "-",
    markersize=3,
    label="Cluster of size 5",
    color="tab:green",
    zorder=0,
)


ax.set_ylabel("$\\dist(x_k,\\mathcal{L}_i) \\mathbin{/} \\Vert A_i^\\dagger b_i\\Vert$")
ax.set_title("Relative forward error")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xmargin(0.02)
ax.set_xlim(0, None)
ax.set_ylim(None, 1.1)
ax.legend()
ax.text(1.02, 0, "$k$", transform=ax.transAxes, ha="left", va="center")


plt.show()
