import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from numerical_experiments.MinAres_adapted_versions import MinAres_full_ortho
from numerical_experiments.util import optimal_backward_error_fro


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


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
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


# ax1.semilogy(
#     np.arange(len(residual_norms_A)),
#     residual_norms_A / residual_norms_A[0],
#     "-",
#     markersize=3,
#     label="Single eigenvalue ($i=1$)",
# )

# ax2.semilogy(
#     np.arange(len(Aresidual_norms_A)),
#     Aresidual_norms_A / Aresidual_norms_A[0],
#     "-",
#     markersize=3,
#     label="Single eigenvalue ($i=1$)",
# )

# ax3.semilogy(
#     np.arange(len(backward_errors_A)),
#     backward_errors_A / norm_A_fro,
#     "--",
#     markersize=3,
#     label="Single eigenvalue",
#     color="tab:orange",
# )

ax.semilogy(
    np.arange(len(forw_err_A)),
    forw_err_A / forw_err_A[0],
    "--",
    markersize=3,
    label="Single eigenvalue",
    color="tab:orange",
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


# ax1.semilogy(
#     np.arange(len(residual_norms_B)),
#     residual_norms_B / residual_norms_B[0],
#     "--",
#     markersize=3,
#     label="Cluster of size $i=2$",
# )

# ax2.semilogy(
#     np.arange(len(Aresidual_norms_B)),
#     Aresidual_norms_B / Aresidual_norms_B[0],
#     "--",
#     markersize=3,
#     label="Cluster of size $i=2$",
# )

# ax3.semilogy(
#     np.arange(len(backward_errors_B)),
#     backward_errors_B / norm_A_fro,
#     ":",
#     markersize=3,
#     label="Cluster of size 2",
# )

ax.semilogy(
    np.arange(len(forw_err_B)),
    forw_err_B / forw_err_B[0],
    ":",
    markersize=3,
    label="Cluster of size 2",
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

# ax1.semilogy(
#     np.arange(len(residual_norms_C)),
#     residual_norms_C / residual_norms_C[0],
#     ":",
#     markersize=3,
#     label="Cluster of size $i=5$",
# )

# ax2.semilogy(
#     np.arange(len(Aresidual_norms_C)),
#     Aresidual_norms_C / Aresidual_norms_C[0],
#     ":",
#     markersize=3,
#     label="Cluster of size $i=5$",
# )

# ax3.semilogy(
#     np.arange(len(backward_errors_C)),
#     backward_errors_C / norm_A_fro,
#     "-",
#     markersize=3,
#     label="Cluster of size 5",
#     color="tab:green",
# )

ax.semilogy(
    np.arange(len(forw_err_C)),
    forw_err_C / forw_err_C[0],
    "-",
    markersize=3,
    label="Cluster of size 5",
    color="tab:green",
)


# ax1.set_xlabel("$k$", loc="right")
# ax2.set_xlabel("$k$", loc="right")
# ax3.set_xlabel("$k$", loc="right")
# ax1.set_ylabel(
#     "$\\Vert\\mathrm{P}_{\\mathcal{R}(A_i)}r_k\\Vert \\mathbin{/} \\Vert\\mathrm{P}_{\\mathcal{R}(A_i)}b\\Vert$"
# )
# ax2.set_ylabel("$\\Vert A_ir_k\\Vert \\mathbin{/} \\Vert A_ib\\Vert$")
# ax3.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A_1\\Vert_F$")
# ax1.set_title("Relative projected residual")
# ax2.set_title("Relative $A_j$-residual")
# ax3.set_title("Backward error relative to $\\Vert A_1\\Vert$")
# ax1.legend()
# ax2.legend()
# ax3.legend()

ax.set_xlabel("$k$", loc="right")
ax.set_ylabel("$\\dist(x_k,\\mathcal{L}) \\mathbin{/} \\Vert A^\\dagger b_i\\Vert$")
ax.set_title("Relative forward error")
ax.legend()

plt.show()
