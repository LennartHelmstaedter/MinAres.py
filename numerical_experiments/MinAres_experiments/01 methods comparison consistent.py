import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse as sps

from MinAres import MinAres
from numerical_experiments.lsmr_lsqr_gmres import lsmr
from numerical_experiments.lsmr_lsqr_gmres import lsqr
from numerical_experiments.util import optimal_backward_error_fro


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    }
)


def append_residuals(
    x, k, norm_r_k, norm_Ar_k, A, b, residual_norms, Aresidual_norms, back_err
):
    res = b - A @ x
    Ares = A @ res
    residual_norms.append(norm(res))
    Aresidual_norms.append(norm(Ares))
    back_err.append(optimal_backward_error_fro(A, b, x))


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
fig1.subplots_adjust(wspace=0.3, left=0.1, right=0.98)
fig2, ax3 = plt.subplots(1, 1, figsize=(3.5, 3.5))
fig2.subplots_adjust(left=0.2, right=0.95)


ax1.set_xlabel("$k$", loc="right")
ax2.set_xlabel("$k$", loc="right")
ax3.set_xlabel("$k$", loc="right")
ax1.set_title("Relative Residual")
ax2.set_title("Relative $A$-Residual")
ax3.set_title("Relative Backward Error")
ax1.set_ylabel("$\\Vert r_k\\Vert \\mathbin{/} \\Vert b\\Vert$")
ax2.set_ylabel("$\\Vert Ar_k\\Vert \\mathbin{/} \\Vert Ab\\Vert$")
ax3.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A\\Vert_F$")


n = 20

A = np.diag(
    (
        *np.geomspace(0.1, 10, int(np.ceil(n / 2)), endpoint=False),
        *np.flip(np.geomspace(-100, -10, n // 2, endpoint=False)),
        0,
    )
)

b = np.ones(n + 1)
b[n] = 0


norm_A_fro = norm(A, ord="fro")
backward_error_x_zero = optimal_backward_error_fro(A, b, np.zeros(0))


residual_norms_minares = []
Aresidual_norms_minares = []
back_err_minares = []
x, info = MinAres(
    A,
    b,
    1e-13,
    1e-13,
    k_max=4 * n,
    callback=append_residuals,
    callback_args=(
        A,
        b,
        residual_norms_minares,
        Aresidual_norms_minares,
        back_err_minares,
    ),
)

ax1.semilogy(
    np.arange(len(residual_norms_minares)),
    residual_norms_minares / residual_norms_minares[0],
    "-",
    markersize=3,
    label="\\textsc{MinAres}",
    zorder=-1,
)
ax2.semilogy(
    np.arange(len(Aresidual_norms_minares)),
    Aresidual_norms_minares / Aresidual_norms_minares[0],
    "-",
    label="\\textsc{MinAres}",
    zorder=-1,
)
ax3.semilogy(
    np.arange(len(back_err_minares)),
    back_err_minares / norm_A_fro,
    "-",
    label="\\textsc{MinAres}",
    zorder=-1,
)


residual_norms_minres = [norm(b)]
Aresidual_norms_minres = [norm(A @ b)]
back_err_minres = [backward_error_x_zero]


def append_residuals_minres(x):
    res = b - A @ x
    Ares = A @ res
    residual_norms_minres.append(norm(res))
    Aresidual_norms_minres.append(norm(Ares))
    back_err_minres.append(optimal_backward_error_fro(A, b, x))


sps.linalg.minres(A, b, rtol=1e-15, callback=append_residuals_minres)


ax1.semilogy(
    np.arange(len(residual_norms_minres)),
    residual_norms_minres / residual_norms_minres[0],
    "--",
    label="\\textsc{Minres}",
    zorder=4,
)
ax2.semilogy(
    np.arange(len(Aresidual_norms_minres)),
    Aresidual_norms_minres / Aresidual_norms_minres[0],
    "--",
    label="\\textsc{Minres}",
    zorder=4,
)
ax3.semilogy(
    np.arange(len(back_err_minres)),
    back_err_minres / norm_A_fro,
    "--",
    label="\\textsc{Minres}",
    zorder=4,
)


residual_norms_lsmr = [norm(b)]
Aresidual_norms_lsmr = [norm(A @ b)]
back_err_lsmr = [backward_error_x_zero]


def append_residuals_lsmr(x):
    res = b - A @ x
    Ares = A @ res
    residual_norms_lsmr.append(norm(res))
    Aresidual_norms_lsmr.append(norm(Ares))
    back_err_lsmr.append(optimal_backward_error_fro(A, b, x))


x, istop, *_ = lsmr(
    A,
    b,
    atol=1e-15,
    btol=1e-15,
    conlim=1e8,
    callback=append_residuals_lsmr,
    maxiter=4 * n,
)

ax1.semilogy(
    np.arange(len(residual_norms_lsmr)),
    residual_norms_lsmr / residual_norms_lsmr[0],
    ":",
    label="\\textsc{Lsmr}",
    zorder=1,
)
ax2.semilogy(
    np.arange(len(Aresidual_norms_lsmr)),
    Aresidual_norms_lsmr / Aresidual_norms_lsmr[0],
    ":",
    label="\\textsc{Lsmr}",
    zorder=1,
)
ax3.semilogy(
    np.arange(len(back_err_lsmr)),
    back_err_lsmr / norm_A_fro,
    ":",
    label="\\textsc{Lsmr}",
    zorder=1,
)


residual_norms_lsqr = [norm(b)]
Aresidual_norms_lsqr = [norm(A @ b)]
back_err_lsqr = [backward_error_x_zero]


def append_residuals_lsqr(x):
    res = b - A @ x
    Ares = A @ res
    residual_norms_lsqr.append(norm(res))
    Aresidual_norms_lsqr.append(norm(Ares))
    back_err_lsqr.append(optimal_backward_error_fro(A, b, x))


x, istop, *_ = lsqr(
    A,
    b,
    atol=1e-15,
    btol=1e-15,
    conlim=1e8,
    iter_lim=4 * n,
    callback=append_residuals_lsqr,
)

ax1.semilogy(
    np.arange(len(residual_norms_lsqr)),
    residual_norms_lsqr / residual_norms_lsqr[0],
    "-.",
    label="\\textsc{Lsqr}",
    zorder=0,
)
ax2.semilogy(
    np.arange(len(Aresidual_norms_lsqr)),
    Aresidual_norms_lsqr / Aresidual_norms_lsqr[0],
    "-.",
    label="\\textsc{Lsqr}",
    zorder=0,
)
ax3.semilogy(
    np.arange(len(back_err_lsqr)),
    back_err_lsqr / norm_A_fro,
    "-.",
    label="\\textsc{Lsqr}",
    zorder=0,
)


ax1.legend()
ax2.legend()
ax3.legend()


plt.show()
