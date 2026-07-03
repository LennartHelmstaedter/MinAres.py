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
    }
)


def append_residuals(
    x,
    k,
    norm_r_k,
    norm_Ar_k,
    A,
    b,
    residual_norms,
    Aresidual_norms,
    backward_errors,
):
    res = b - A @ x
    Ares = A @ res
    residual_norms.append(norm(res[:-1]))
    Aresidual_norms.append(norm(Ares))
    backward_errors.append(optimal_backward_error_fro(A, b, x))


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


fig, (ax3) = plt.subplots(1, 1, figsize=(4, 4))
fig.subplots_adjust(left=0.2, right=0.95)


residual_norms_A = []
Aresidual_norms_A = []
backward_errors_A = []
MinAres_full_ortho(
    A,
    a,
    1e-13,
    1e-13,
    callback=append_residuals,
    callback_args=(A, a, residual_norms_A, Aresidual_norms_A, backward_errors_A),
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

ax3.semilogy(
    np.arange(len(backward_errors_A)),
    backward_errors_A / norm_A_fro,
    "--",
    markersize=3,
    label="Single eigenvalue",
    color="tab:orange",
)


residual_norms_B = []
Aresidual_norms_B = []
backward_errors_B = []
MinAres_full_ortho(
    B,
    b,
    1e-13,
    1e-13,
    callback=append_residuals,
    callback_args=(B, b, residual_norms_B, Aresidual_norms_B, backward_errors_B),
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

ax3.semilogy(
    np.arange(len(backward_errors_B)),
    backward_errors_B / norm_A_fro,
    ":",
    markersize=3,
    label="Cluster of size 2",
)


residual_norms_C = []
Aresidual_norms_C = []
backward_errors_C = []
MinAres_full_ortho(
    C,
    c,
    1e-13,
    1e-13,
    callback=append_residuals,
    callback_args=(C, c, residual_norms_C, Aresidual_norms_C, backward_errors_C),
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

ax3.semilogy(
    np.arange(len(backward_errors_C)),
    backward_errors_C / norm_A_fro,
    "-",
    markersize=3,
    label="Cluster of size 5",
    color="tab:green",
)


# ax1.set_xlabel("$k$", loc="right")
# ax2.set_xlabel("$k$", loc="right")
ax3.set_xlabel("$k$", loc="right")
# ax1.set_ylabel(
#     "$\\Vert\\mathrm{P}_{\\mathcal{R}(A_i)}r_k\\Vert \\mathbin{/} \\Vert\\mathrm{P}_{\\mathcal{R}(A_i)}b\\Vert$"
# )
# ax2.set_ylabel("$\\Vert A_ir_k\\Vert \\mathbin{/} \\Vert A_ib\\Vert$")
ax3.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A_1\\Vert_F$")
# ax1.set_title("Relative projected residual")
# ax2.set_title("Relative $A_j$-residual")
ax3.set_title("Backward error relative to $\\Vert A_1\\Vert$")
# ax1.legend()
# ax2.legend()
ax3.legend()

plt.show()
