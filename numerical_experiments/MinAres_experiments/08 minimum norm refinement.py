import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from MinAres import MinAres


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


n = 20

A = np.diag(
    (
        *np.geomspace(0.1, 10, int(np.ceil(n / 2)), endpoint=False),
        *np.flip(np.geomspace(-100, -10, n // 2, endpoint=False)),
        0,
    )
)

b = np.ones(n + 1)


x_dagger = np.zeros(n + 1)
for k in range(n):
    x_dagger[k] = b[k] / A[k, k]


def append_errors(
    x_k,
    k,
    norm_r_k,
    norm_Ar_k,
    A,
    b,
    x_dagger,
    forw_err,
    forw_err_corr,
    dist,
    dist_corr,
):
    res = b - A @ x_k
    forw_err.append(norm(res[:-1] / np.diagonal(A)[:-1]))
    dist.append(norm(x_k - x_dagger) / norm(x_dagger))

    normalized_res = res / norm(res)
    x_corr = x_k - np.vdot(normalized_res, x_k) * normalized_res
    res_corr = b - A @ x_corr
    forw_err_corr.append(norm(res_corr[:-1] / np.diagonal(A)[:-1]))
    dist_corr.append(norm(x_corr - x_dagger) / norm(x_dagger))


forw_err = []
forw_err_corr = []
dist = []
dist_corr = []

MinAres(
    A,
    b,
    callback=append_errors,
    callback_args=(
        A,
        b,
        x_dagger,
        forw_err,
        forw_err_corr,
        dist,
        dist_corr,
    ),
)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.975)


ax1.semilogy(
    np.arange(len(forw_err)), forw_err / forw_err[0], "-", label="normal iterate"
)
ax1.semilogy(
    np.arange(len(forw_err_corr)),
    forw_err_corr / forw_err[0],
    "--",
    label="refined iterate",
)

ax1.set_ylabel("$\\dist(x_k,\\mathcal{L}) \\mathbin{/} \\Vert x^\\dagger\\Vert$")
ax1.set_title("Relative forward error")


ax2.semilogy(np.arange(len(dist)), dist, "-", label="normal iterate")
ax2.semilogy(np.arange(len(dist_corr)), dist_corr, "--", label="refined iterate")

ax2.set_ylabel("$\\Vert x_k-x^\\dagger\\Vert \\mathbin{/} \\Vert x^\\dagger\\Vert$")
ax2.set_title("Relative distance to minimum-norm solution M")


for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xmargin(0.02)
    ax.set_xlim(0, None)
    ax.legend()
    ax.text(1.02, 0, "$k$", transform=ax.transAxes, ha="left", va="center")


ax1.set_ylim(None, 1.1)
ax1.set_yticks([10 ** (-j) for j in range(0, 13, 3)])

plt.show()
