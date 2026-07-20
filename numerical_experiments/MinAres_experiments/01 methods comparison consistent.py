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
        "text.latex.preamble": """
            \\usepackage{amsmath}
            \\DeclareMathOperator{\\dist}{dist}
        """,
    }
)


def append_stats(
    x, k, norm_r_k, norm_Ar_k, A, b, residual_norms, Aresidual_norms, back_err, forw_err
):
    res = b - A @ x
    Ares = A @ res
    residual_norms.append(norm(res))
    Aresidual_norms.append(norm(Ares))
    back_err.append(optimal_backward_error_fro(A, b, x))
    forw_err.append(norm(res[:-1] / np.diagonal(A)[:-1]))


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
fig1.subplots_adjust(wspace=0.3, left=0.1, right=0.975)
fig1, (ax3, ax4) = plt.subplots(1, 2, figsize=(7, 2.5))
fig1.subplots_adjust(wspace=0.3, left=0.1, right=0.975)


ax1.set_title("Relative residual")
ax2.set_title("Relative $A$-residual")
ax3.set_title("Relative backward error")
ax4.set_title("Relative forward error")
ax1.set_ylabel("$\\Vert r_k\\Vert \\mathbin{/} \\Vert b\\Vert$")
ax2.set_ylabel("$\\Vert Ar_k\\Vert \\mathbin{/} \\Vert Ab\\Vert$")
ax3.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A\\Vert_F$")
ax4.set_ylabel("$\\dist(x_k,\\mathcal{L}) \\mathbin{/} \\Vert A^\\dagger b\\Vert$")


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
forward_error_x_zero = norm(b[:-1] / np.diagonal(A)[:-1])


residual_norms_minares = []
Aresidual_norms_minares = []
back_err_minares = []
forw_err_minares = []
x, info = MinAres(
    A,
    b,
    1e-13,
    1e-13,
    k_max=4 * n,
    callback=append_stats,
    callback_args=(
        A,
        b,
        residual_norms_minares,
        Aresidual_norms_minares,
        back_err_minares,
        forw_err_minares,
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
ax4.semilogy(
    np.arange(len(forw_err_minares)),
    forw_err_minares / forward_error_x_zero,
    "-",
    label="\\textsc{MinAres}",
    zorder=-1,
)


residual_norms_minres = [norm(b)]
Aresidual_norms_minres = [norm(A @ b)]
back_err_minres = [backward_error_x_zero]
forw_err_minres = [forward_error_x_zero]


def append_stats_minres(x):
    res = b - A @ x
    Ares = A @ res
    residual_norms_minres.append(norm(res))
    Aresidual_norms_minres.append(norm(Ares))
    back_err_minres.append(optimal_backward_error_fro(A, b, x))
    forw_err_minres.append(norm(res[:-1] / np.diagonal(A)[:-1]))


sps.linalg.minres(A, b, rtol=1e-15, callback=append_stats_minres)


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
ax4.semilogy(
    np.arange(len(forw_err_minres)),
    forw_err_minres / forward_error_x_zero,
    "--",
    label="\\textsc{Minres}",
    zorder=4,
)


residual_norms_lsmr = [norm(b)]
Aresidual_norms_lsmr = [norm(A @ b)]
back_err_lsmr = [backward_error_x_zero]
forw_err_lsmr = [forward_error_x_zero]


def append_stats_lsmr(x):
    res = b - A @ x
    Ares = A @ res
    residual_norms_lsmr.append(norm(res))
    Aresidual_norms_lsmr.append(norm(Ares))
    back_err_lsmr.append(optimal_backward_error_fro(A, b, x))
    forw_err_lsmr.append(norm(res[:-1] / np.diagonal(A)[:-1]))


x, istop, *_ = lsmr(
    A,
    b,
    atol=1e-15,
    btol=1e-15,
    conlim=1e8,
    callback=append_stats_lsmr,
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
ax4.semilogy(
    np.arange(len(forw_err_lsmr)),
    forw_err_lsmr / forward_error_x_zero,
    ":",
    label="\\textsc{Lsmr}",
    zorder=1,
)


residual_norms_lsqr = [norm(b)]
Aresidual_norms_lsqr = [norm(A @ b)]
back_err_lsqr = [backward_error_x_zero]
forw_err_lsqr = [forward_error_x_zero]


def append_stats_lsqr(x):
    res = b - A @ x
    Ares = A @ res
    residual_norms_lsqr.append(norm(res))
    Aresidual_norms_lsqr.append(norm(Ares))
    back_err_lsqr.append(optimal_backward_error_fro(A, b, x))
    forw_err_lsqr.append(norm(res[:-1] / np.diagonal(A)[:-1]))


x, istop, *_ = lsqr(
    A,
    b,
    atol=1e-15,
    btol=1e-15,
    conlim=1e8,
    iter_lim=4 * n,
    callback=append_stats_lsqr,
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
ax4.semilogy(
    np.arange(len(forw_err_lsqr)),
    forw_err_lsqr / forward_error_x_zero,
    "-.",
    label="\\textsc{Lsqr}",
    zorder=0,
)

for ax in (ax1, ax2, ax3, ax4):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xmargin(0.02)
    ax.set_xlim(0, None)
    ax.set_ylim(None, 1.1)
    ax.legend()
    ax.text(1.02, 0, "$k$", transform=ax.transAxes, ha="left", va="center")

plt.show()
