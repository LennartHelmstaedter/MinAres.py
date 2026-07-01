import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from MinAres import MinAres
from numerical_experiments.util import optimal_backward_error_fro


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


x_dagger = np.zeros(n + 1)
for k in range(n):
    x_dagger[k] = b[k] / A[k, k]

norm_A = norm(A, ord="fro")


def append_errors(
    x_k,
    k,
    norm_r_k,
    norm_Ar_k,
    A,
    norm_A,
    b,
    x_dagger,
    forw_err,
    forw_err_corr,
    back_err,
    back_err_corr,
):
    forw_err.append(norm(x_k - x_dagger) / norm(x_dagger))
    back_err.append(optimal_backward_error_fro(A, b, x_k) / norm_A)

    res = b - A @ x_k
    normalized_res = res / norm(res)
    x_corr = x_k - np.vdot(normalized_res, x_k) * normalized_res
    forw_err_corr.append(norm(x_corr - x_dagger) / norm(x_dagger))
    back_err_corr.append(optimal_backward_error_fro(A, b, x_corr) / norm_A)


forw_err = []
forw_err_corr = []
back_err = []
back_err_corr = []

MinAres(
    A,
    b,
    callback=append_errors,
    callback_args=(
        A,
        norm_A,
        b,
        x_dagger,
        forw_err,
        forw_err_corr,
        back_err,
        back_err_corr,
    ),
)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.95)


ax1.semilogy(np.arange(len(back_err)), back_err, "-", label="normal iterate")
ax1.semilogy(
    np.arange(len(back_err_corr)), back_err_corr, "--", label="refined iterate"
)

ax1.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A\\Vert_F$")
ax1.set_xlabel("$k$", loc="right")
ax1.set_title("Relative backward error")
ax1.legend(loc="lower left")


ax2.semilogy(np.arange(len(forw_err)), forw_err, "-", label="normal iterate")
ax2.semilogy(
    np.arange(len(forw_err_corr)), forw_err_corr, "--", label="refined iterate"
)

ax2.set_ylabel("$\\Vert x_k-x^\\dagger\\Vert \\mathbin{/} \\Vert x^\\dagger\\Vert$")
ax2.set_xlabel("$k$", loc="right")
ax2.set_title("Relative distance to minimum-norm solution")
ax2.legend(loc="lower left")

plt.show()
