import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from numerical_experiments.MinAres_adapted_versions import MinAres_TV


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
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

cond_T = []
cond_TT = []
diff = []

beta_params = np.geomspace(1e-11, 1, 100)

for eps in beta_params:
    b[-1] = eps
    x_comp, (k, T, V, beta_1, _, __, ___) = MinAres_TV(
        A, b, k_max=35, rtol=-1, Ar_tol=-1, beta_tol=-1
    )

    reduced_rhs = np.zeros(T.shape[0])
    reduced_rhs[0] = beta_1 * T[0, 0]
    reduced_rhs[1] = beta_1 * T[1, 0]

    x_svd = V @ np.linalg.lstsq(T @ T[:-1, :-1], reduced_rhs, rcond=1e-1000)[0]

    cond_T.append(np.linalg.cond(T[:-1, :-1]))
    cond_TT.append(np.linalg.cond(T) * np.linalg.cond(T[:-1, :-1]))
    diff.append(norm(x_comp - x_svd) / norm(x_svd))

fig1, ax = plt.subplots(1, 1, figsize=(3.5, 3))
fig1.subplots_adjust(left=0.17, right=0.95, bottom=0.16)
ax.loglog(beta_params, cond_T, ".")
ax.set_xlabel("$\\Vert\\mathrm{P}_{\\mathcal{N}(A)}b\\Vert$")
ax.set_ylabel("$\\kappa(T_{37,36})$")
ax.set_title("Conditioning factor")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
fig2.subplots_adjust(wspace=0.45, left=0.12, right=0.98, bottom=0.16, top=0.85)

ax1.loglog(cond_TT, diff, ".")
ax2.loglog(cond_TT, [d / c for (d, c) in zip(diff, cond_TT)], ".")

ax1.set_xlabel("$\\kappa(T_{38,37})\\kappa(T_{37,36})$")
ax1.set_ylabel(
    "$\\displaystyle\\frac"
    "{\\Vert \\widehat{x}_{36} - x_{36}\\Vert}"
    "{\\Vert x_{36}\\Vert}"
    "$"
)
ax1.set_title("Rounding error estimate")

ax2.set_xlabel("$\\kappa(T_{38,37})\\kappa(T_{37,36})$")
ax2.set_ylabel(
    "$\\displaystyle\\frac"
    "{\\Vert \\widehat{x}_{36} - x_{36}\\Vert}"
    "{\\Vert x_{36}\\Vert}"
    "\\Bigm/"
    "\\kappa(T_{38,37})\\kappa(T_{37,36})$"
)
ax2.set_title("Quotient of rounding error\nestimate and conditioning factor")


for axs in (ax, ax1, ax2):
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

ax.set_yticks([10**j for j in range(3, 15, 2)])


plt.show()
