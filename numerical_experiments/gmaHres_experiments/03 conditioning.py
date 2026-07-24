import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from numerical_experiments.gmaHres_adapted_versions import gmaHres_I_U, gmaHres_II_U

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    }
)


def append_info_gmaHres_I(
    x,
    k,
    norm_aHr,
    U,
    U_tilde,
    A,
    b,
    exact_solution,
    cond_U,
    cond_U_tilde,
    forw_err,
):
    if U is None:
        cond_U.append(None)
    else:
        cond_U.append(np.linalg.cond(U))

    if U_tilde is None:
        cond_U_tilde.append(None)
    else:
        cond_U_tilde.append(np.linalg.cond(U_tilde))

    forw_err.append(norm(x - exact_solution))


def append_info_gmaHres_II(x, k, norm_aHr, U, A, b, exact_solution, cond_U, forw_err):
    if U is None:
        cond_U.append(None)
    else:
        cond_U.append(np.linalg.cond(U))

    forw_err.append(norm(x - exact_solution))


n = 100

A = np.diag(np.linspace(0.001, 1000, n, endpoint=True))
kappa_A = A[n - 1, n - 1] / A[0, 0]
A = np.fliplr(A)

b = np.ones(n)

exact_solution = b[::-1] / A[np.arange(n - 1, -1, -1), np.arange(n)]


cond_U_I = []
cond_U_tilde = []
forw_err_I = []
gmaHres_I_U(
    A,
    b,
    aHr_tol=1e-13,
    maxiter=90,
    callback=append_info_gmaHres_I,
    callback_type="x",
    callback_args=(
        A,
        b,
        exact_solution,
        cond_U_I,
        cond_U_tilde,
        forw_err_I,
    ),
)
cond_U_I_indices, cond_U_I = zip(
    *[(i, v) for i, v in enumerate(cond_U_I) if v is not None]
)
cond_U_tilde_indices, cond_U_tilde = zip(
    *[(i, v) for i, v in enumerate(cond_U_tilde) if v is not None]
)

cond_U_II = []
forw_err_II = []
gmaHres_II_U(
    A,
    b,
    aHr_tol=1e-13,
    maxiter=90,
    callback=append_info_gmaHres_II,
    callback_type="x",
    callback_args=(A, b, exact_solution, cond_U_II, forw_err_II),
)
cond_U_II_indices, cond_U_II = zip(
    *[(i, v) for i, v in enumerate(cond_U_II) if v is not None]
)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
fig.subplots_adjust(wspace=0.3, left=0.07, right=0.975)


ax1.semilogy(
    cond_U_I_indices, cond_U_I, "-", label="\\textsc{GmA*res}-I: $\\kappa(U_k)$"
)
ax1.semilogy(
    cond_U_tilde_indices,
    cond_U_tilde,
    "-.",
    label="\\textsc{GmA*res}-I: $\\kappa(\\widetilde{U}_k)$",
    color="tab:blue",
)
ax1.semilogy(
    cond_U_II_indices,
    cond_U_II,
    "--",
    label="\\textsc{GmA*res}-II: $\\kappa(U_k)$",
    color="tab:orange",
)
ax1.axhline(y=kappa_A, linestyle=":", color="darkgray")
ax1.axhline(y=kappa_A**3, linestyle=":", color="darkgray")


ax2.semilogy(
    np.arange(len(forw_err_I)),
    forw_err_I / forw_err_I[0],
    "-",
    label="\\textsc{GmA*res}-I",
)
ax2.semilogy(
    np.arange(len(forw_err_II)),
    forw_err_II / forw_err_II[0],
    "--",
    label="\\textsc{GmA*res}-II",
)


ax1.set_title("Condition Number")
ax1.set_xlabel("$k$", loc="right")
ax1.set_ylim(1, 1.14 * 10**18)
ax1.set_yticks([10**j for j in range(0, 19, 3)])
ax1.legend(loc="upper left", bbox_to_anchor=(0, 1))

ax2.set_title("Relative forward error")
ax2.set_ylabel("$\\Vert x-x^\\dagger\\Vert \\mathbin{/} \\Vert x^\\dagger\\Vert$")
ax2.set_ylim(None, 1.1)
ax2.set_yticks([10 ** (-j) for j in range(0, 13, 3)])
ax2.legend(loc="lower left")


for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xmargin(0.02)
    ax.set_xlim(0, None)
    ax.text(1.02, 0, "$k$", transform=ax.transAxes, ha="left", va="center")


plt.show()
