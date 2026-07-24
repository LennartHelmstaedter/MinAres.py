import numpy as np
from numpy.linalg import norm
import scipy.sparse as sps
import matplotlib.pyplot as plt

from gmaHres import gmaHres_I, gmaHres_II
from MinAres import MinAres
from numerical_experiments.lsmr_lsqr_gmres import lsmr, lsqr, gmres


np.random.seed(0)


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    }
)


def append_info(x, *args):
    A = args[-1]
    b = args[-2]
    n = len(b)
    Aresiduals = args[-3]
    proj_residuals = args[-4]

    res = b - A @ x
    Ares = A.T.conj() @ res

    proj_res = res - np.vdot(np.ones(n), res) / n * np.ones(n)

    proj_residuals.append(norm(proj_res))
    Aresiduals.append(norm(Ares))


m = 100
h = 1 / m
d = 10
n = m * m

a1 = 1 + d * h / 2
a2 = 1 - d * h / 2

ones = np.ones(m)

T = sps.diags_array(
    (a2 * ones, -4 * ones, a1 * ones),
    offsets=(-1, 0, 1),
    shape=(m, m),
    format="lil",  # LIL for easy modification of corner entries
)
T[m - 1, 0] = a1
T[0, m - 1] = a2
T = T.tocsr()

eye_structure = sps.diags_array(
    (ones, ones),
    offsets=(-1, 1),
    shape=(m, m),
    format="lil",
)
eye_structure[0, m - 1] = 1
eye_structure[m - 1, 0] = 1

eye = sps.eye(m)

np.random.seed(0)
A = (sps.kron(eye_structure, eye) + sps.kron(eye, T)) / (h * h)
b = np.random.randn(n)

norm_proj_b = norm(b - np.vdot(np.ones(n), b) / n * np.ones(n))
norm_aHb = norm(A.T.conj() @ b)


proj_residuals_gmahres_I = []
Aresiduals_gmahres_I = []
gmaHres_I(
    A,
    b,
    maxiter=600,
    callback=append_info,
    callback_type="x",
    callback_args=(proj_residuals_gmahres_I, Aresiduals_gmahres_I, b, A),
)
proj_residuals_gmahres_I /= norm_proj_b
Aresiduals_gmahres_I /= norm_aHb

proj_residuals_gmahres_II = []
Aresiduals_gmahres_II = []
gmaHres_II(
    A,
    b,
    maxiter=600,
    callback=append_info,
    callback_type="x",
    callback_args=(proj_residuals_gmahres_II, Aresiduals_gmahres_II, b, A),
)
proj_residuals_gmahres_II /= norm_proj_b
Aresiduals_gmahres_II /= norm_aHb

proj_residuals_gmres = [norm_proj_b]
Aresiduals_gmres = [norm_aHb]
gmres(
    A,
    b,
    rtol=1e-13,
    restart=600,
    maxiter=1,
    callback=append_info,
    callback_type="x",
    callback_args=(proj_residuals_gmres, Aresiduals_gmres, b, A),
)
proj_residuals_gmres /= norm_proj_b
Aresiduals_gmres = Aresiduals_gmres / norm_aHb


residuals_lsmr = [norm_proj_b]
Aresiduals_lsmr = [norm_aHb]
lsmr(
    A,
    b,
    atol=1e-13,
    maxiter=600,
    callback=append_info,
    callback_args=(residuals_lsmr, Aresiduals_lsmr, b, A),
)
residuals_lsmr /= norm_proj_b
Aresiduals_lsmr = Aresiduals_lsmr / norm_aHb


residuals_lsqr = [norm_proj_b]
Aresiduals_lsqr = [norm_aHb]
lsqr(
    A,
    b,
    atol=1e-13,
    iter_lim=600,
    callback=append_info,
    callback_args=(residuals_lsqr, Aresiduals_lsqr, b, A),
)
residuals_lsqr /= norm_proj_b
Aresiduals_lsqr = Aresiduals_lsqr / norm_aHb


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
fig1.subplots_adjust(wspace=0.3, left=0.1, right=0.975)


ax1.semilogy(
    np.arange(len(proj_residuals_gmahres_I)),
    proj_residuals_gmahres_I,
    "-",
    label="\\textsc{GmA*res}-I",
)
ax1.semilogy(
    np.arange(len(proj_residuals_gmahres_II)),
    proj_residuals_gmahres_II,
    "--",
    label="\\textsc{GmA*res}-II",
)
ax1.semilogy(
    np.arange(len(proj_residuals_gmres)),
    proj_residuals_gmres,
    ":",
    label="\\textsc{Gmres}",
)
ax1.semilogy(
    np.arange(len(residuals_lsmr)),
    residuals_lsmr,
    linestyle=(0, (3, 1, 1, 1, 1, 1)),
    label="\\textsc{Lsmr}",
)
ax1.semilogy(
    np.arange(len(residuals_lsqr)),
    residuals_lsqr,
    "-.",
    label="\\textsc{Lsqr}",
)

ax2.semilogy(
    np.arange(len(Aresiduals_gmahres_I)),
    Aresiduals_gmahres_I,
    "-",
    label="\\textsc{GmA*res}-I",
)
ax2.semilogy(
    np.arange(len(Aresiduals_gmahres_II)),
    Aresiduals_gmahres_II,
    "--",
    label="\\textsc{GmA*res}-II",
)
ax2.semilogy(
    np.arange(len(Aresiduals_gmres)),
    Aresiduals_gmres,
    ":",
    label="\\textsc{Gmres}",
)
ax2.semilogy(
    np.arange(len(Aresiduals_lsmr)),
    Aresiduals_lsmr,
    linestyle=(0, (3, 1, 1, 1, 1, 1)),
    label="\\textsc{Lsmr}",
)
ax2.semilogy(
    np.arange(len(Aresiduals_lsqr)),
    Aresiduals_lsqr,
    "-.",
    label="\\textsc{Lsqr}",
)


ax1.set_title("Relative Projected Residual")
ax1.set_ylabel(
    "$\\Vert\\mathrm{P}_{\\mathcal{R}(A)}r_k\\Vert"
    "\\mathbin{/}"
    "\\Vert\\mathrm{P}_{\\mathcal{R}(A)}b\\Vert$"
)

ax2.set_title("Relative $A^H$-Residual")
ax2.set_ylabel("$\\Vert A^Hr_k\\Vert \\mathbin{/} \\Vert A^Hb\\Vert$")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xmargin(0.02)
    ax.set_xlim(0, None)
    ax.set_ylim(None, 1.1)
    ax.text(1.02, 0, "$k$", transform=ax.transAxes, ha="left", va="center")

ax1.legend()


plt.show()
