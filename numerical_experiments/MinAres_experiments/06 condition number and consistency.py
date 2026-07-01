import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from MinAres import MinAres
from numerical_experiments.adapted_methods import MinAres_TV
from numerical_experiments.util import optimal_backward_error_fro


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    }
)


def append_cond_back_err(
    x,
    k,
    T,
    V,
    norm_r_k,
    norm_Ar_k,
    A,
    b,
    T_conds,
    backward_errors,
):
    T_conds.append(np.linalg.cond(T))
    backward_errors.append(optimal_backward_error_fro(A, b, x))


n = 20

A = np.diag(
    (
        *np.geomspace(0.1, 10, int(np.ceil(n / 2)), endpoint=False),
        *np.flip(np.geomspace(-100, -10, n // 2, endpoint=False)),
        0,
    )
)

b_inconsistent = np.ones(n + 1)
b_inconsistent[n] = 1

b_consistent = np.ones(n + 1)
b_consistent[n] = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.95)


T_conds_inconsistent = []
backward_errors_inconsistent = []
MinAres_TV(
    A,
    b_inconsistent,
    1e-13,
    1e-13,
    callback=append_cond_back_err,
    callback_args=(
        A,
        b_inconsistent,
        T_conds_inconsistent,
        backward_errors_inconsistent,
    ),
)

ax1.semilogy(
    np.arange(len(T_conds_inconsistent)),
    T_conds_inconsistent,
    "-",
    label="Inconsistent system",
)
ax2.semilogy(
    np.arange(len(backward_errors_inconsistent)),
    backward_errors_inconsistent / norm(A, ord="fro"),
    "-",
    label="Inconsistent system",
)


T_conds_consistent = []
backward_errors_consistent = []
MinAres_TV(
    A,
    b_consistent,
    1e-13,
    1e-13,
    callback=append_cond_back_err,
    callback_args=(A, b_consistent, T_conds_consistent, backward_errors_consistent),
)

ax1.semilogy(
    np.arange(len(T_conds_consistent)),
    T_conds_consistent,
    "--",
    label="Consistent system",
)
ax2.semilogy(
    np.arange(len(backward_errors_consistent)),
    backward_errors_consistent / norm(A, ord="fro"),
    "--",
    label="Consistent system",
)


ax1.set_xlabel("$k$", loc="right")
ax2.set_xlabel("$k$", loc="right")

ax1.set_ylabel("$\\kappa(T_{k+2,k+1})$")
ax2.set_ylabel("$\\Vert E_{\\min}\\Vert_F \\mathbin{/} \\Vert A\\Vert_F$")

ax1.set_title("Conditioning of the projected matrix")
ax2.set_title("Relative backward error")

ax1.legend()
ax2.legend()

plt.show()
