import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from MinAres import MinAres
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
    back_errs,
):
    res = b - A @ x
    Ares = A @ res
    residual_norms.append(norm(res[:-1]))
    Aresidual_norms.append(norm(Ares))
    back_errs.append(optimal_backward_error_fro(A, b, x))


n = 25

lambda_min = 0.1
lambda_max = 100
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

b = np.ones(n + 1)

norm_A_fro = norm(A, ord="fro")


fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.95)


residual_norms_minares = []
Aresidual_norms_minares = []
back_err_minares = []
MinAres(
    A,
    b,
    1e-13,
    1e-13,
    beta_tol=1e-5,
    callback=append_residuals,
    callback_args=(
        A,
        b,
        residual_norms_minares,
        Aresidual_norms_minares,
        back_err_minares,
    ),
)


# ax1.semilogy(
#     np.arange(len(residual_norms_minares)),
#     residual_norms_minares / residual_norms_minares[0],
#     "-",
#     markersize=3,
#     label="\\textsc{MinAres}",
# )
ax2.semilogy(
    np.arange(len(Aresidual_norms_minares[:-5])),
    Aresidual_norms_minares[:-5] / Aresidual_norms_minares[0],
    "-",
    markersize=3,
    label="three-term recurrence",
)
ax3.semilogy(
    np.arange(len(back_err_minares)),
    back_err_minares / norm_A_fro,
    "-",
    label="three-term recurrence",
)


residual_norms_minares_full_ortho = []
Aresidual_norms_minares_full_ortho = []
back_err_minares_full_ortho = []
MinAres_full_ortho(
    A,
    b,
    1e-13,
    1e-13,
    beta_tol=1e-5,
    callback=append_residuals,
    callback_args=(
        A,
        b,
        residual_norms_minares_full_ortho,
        Aresidual_norms_minares_full_ortho,
        back_err_minares_full_ortho,
    ),
)


# ax1.semilogy(
#     np.arange(len(residual_norms_minares_full_ortho)),
#     residual_norms_minares_full_ortho / residual_norms_minares_full_ortho[0],
#     "--",
#     markersize=3,
#     label="\\textsc{MinAres}*",
# )
ax2.semilogy(
    np.arange(len(Aresidual_norms_minares_full_ortho[:-1])),
    Aresidual_norms_minares_full_ortho[:-1] / Aresidual_norms_minares_full_ortho[0],
    "--",
    markersize=3,
    label="double full\northogonalization",
)
ax3.semilogy(
    np.arange(len(back_err_minares_full_ortho)),
    back_err_minares_full_ortho / norm_A_fro,
    "--",
    label="double full\northogonalization",
)

ax2.set_xlabel("$k$", loc="right")
ax3.set_xlabel("$k$", loc="right")
# ax1.legend()
ax2.legend()
ax3.legend()
ax2.set_ylabel("$\\Vert A_1r_k\\Vert \\mathbin{/} \\Vert A_1b_1\\Vert$")
ax3.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A_1\\Vert_F$")
ax2.set_title("Relative $A_1$-residual")
ax3.set_title("Relative backward error")
plt.show()
