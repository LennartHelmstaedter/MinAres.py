import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from MinAres import MinAres
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


def append_forw_err(
    x,
    k,
    norm_r_k,
    norm_Ar_k,
    A,
    b,
    forw_err,
):
    res = b - A @ x
    forw_err.append(norm(res[:-1] / np.diagonal(A)[:-1]))


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


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
fig.subplots_adjust(left=0.2, right=0.95)


forw_err_minares = []
MinAres(
    A,
    b,
    1e-13,
    1e-13,
    beta_tol=1e-5,
    callback=append_forw_err,
    callback_args=(
        A,
        b,
        forw_err_minares,
    ),
)

ax.semilogy(
    np.arange(len(forw_err_minares)),
    forw_err_minares / forw_err_minares[0],
    "-",
    label="three-term\nrecurrence",
)


forw_err_minares_full_ortho = []
MinAres_full_ortho(
    A,
    b,
    1e-13,
    1e-13,
    beta_tol=1e-5,
    callback=append_forw_err,
    callback_args=(
        A,
        b,
        forw_err_minares_full_ortho,
    ),
)

ax.semilogy(
    np.arange(len(forw_err_minares_full_ortho)),
    forw_err_minares_full_ortho / forw_err_minares_full_ortho[0],
    "--",
    label="double full\northogonalization",
)

ax.set_xlabel("$k$", loc="right")
ax.legend(loc="upper right")
ax.set_ylabel("$\\dist(x_k,\\mathcal{L}_1) \\mathbin{/} \\Vert A_1^\\dagger b_1\\Vert$")
ax.set_title("Relative forward error")

plt.show()
