import numpy as np
from numpy.linalg import norm
import scipy as sp

from MinAres import get_givens_rot


def gmaHres_I(A, b, maxiter=50, callback=None, callback_args=()):
    dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(b) else np.float64

    n = A.shape[0]

    if callback is not None:
        callback(np.zeros(n), *callback_args)

    V = np.zeros((n, maxiter + 1), dtype=dtype)
    W = np.zeros((n, maxiter + 1), dtype=dtype)
    H = np.zeros((maxiter + 1, maxiter), dtype=dtype)
    s = []
    c = []
    R = np.zeros((maxiter + 1, maxiter + 1), dtype=dtype)
    s_tilde = []
    c_tilde = []
    z = np.zeros(maxiter + 1, dtype=dtype)

    beta = norm(b)
    V[:, 0] = b / beta

    W[:, 0] = A.conj().T @ V[:, 0]
    R[0, 0] = norm(W[:, 0])
    W[:, 0] /= R[0, 0]

    z[0] = R[0, 0] * beta
    # norm_aHr = np.abs(z[0])

    stop_type = 0
    # Keeps track of the reason that the iteration stops:
    #   0 : stop because of reaching iteration maxiter
    #   1 : breakdown of the Arnoldi method
    #   2 : breakdown of the Gram–Schmidt method

    for k in range(maxiter):
        print(k)

        # Arnoldi method
        V[:, k + 1] = A @ V[:, k]
        for j in range(k + 1):
            H[j, k] = np.vdot(V[:, j], V[:, k + 1])
            V[:, k + 1] -= H[j, k] * V[:, j]

        H[k + 1, k] = norm(V[:, k + 1])
        if np.isclose(H[k + 1, k], 0):
            stop_type = 1
            break
        V[:, k + 1] /= H[k + 1, k]

        # QR factorization of $H_{k+1,k}$
        for j in range(k):
            H[j : (j + 2), k] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ H[j : (j + 2), k]
            )

        _c, _s, u_kk = get_givens_rot(H[k, k], H[k + 1, k])
        c.append(_c)
        s.append(_s)
        H[k, k] = u_kk
        H[k + 1, k] = 0

        print(np.linalg.cond(H[: (k + 1), : (k + 1)], p=2))

        # Gram–Schmidt
        W[:, k + 1] = A.conj().T @ V[:, k + 1]
        for j in range(k + 1):
            R[j, k + 1] = np.vdot(W[:, j], W[:, k + 1])
            W[:, k + 1] -= R[j, k + 1] * W[:, j]

        R[k + 1, k + 1] = norm(W[:, k + 1])
        if np.isclose(R[k + 1, k + 1], 0):
            stop_type = 2
            break
        W[:, k + 1] /= R[k + 1, k + 1]

        # Computation of $\widetilde{H}_{k+1,k}$
        R[: k + 2, k : k + 2] = R[: k + 2, k : k + 2] @ np.array(
            [[c[k], np.conj(s[k])], [s[k], -np.conj(c[k])]]
        )
        # Now, H_tilde = R[:k+2, :k+1]

        # QR factorization of \widetilde{H}_{k+1,k}
        for j in range(k):
            R[j : (j + 2), k] = (
                np.array(
                    [
                        [c_tilde[j], np.conj(s_tilde[j])],
                        [s_tilde[j], -np.conj(c_tilde[j])],
                    ]
                )
                .conj()
                .T
                @ R[j : (j + 2), k]
            )

        _c, _s, _r = get_givens_rot(R[k, k], R[k + 1, k])
        c_tilde.append(_c)
        s_tilde.append(_s)
        R[k, k] = _r
        R[k + 1, k] = 0

        print(np.linalg.cond(R[: (k + 1), : (k + 1)], p=2))
        print()

        # Computation of z_k
        z[k : (k + 2)] = (
            np.array(
                [[c_tilde[k], np.conj(s_tilde[k])], [s_tilde[k], -np.conj(c_tilde[k])]]
            )
            .conj()
            .T
            @ z[k : (k + 2)]
        )
        # norm_aHr = np.abs(z[k+1])

        # Solve triangular systems and find x_k
        t = sp.linalg.solve_triangular(
            R[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        t = sp.linalg.solve_triangular(H[: (k + 1), : (k + 1)], t, check_finite=False)
        x = V[:, : (k + 1)] @ t

        if callback is not None:
            callback(x, *callback_args)

    if stop_type == 0:
        return x
    elif stop_type == 1:
        beta_e_1 = np.zeros(k + 1, dtype=dtype)
        beta_e_1[0] = beta

        for j in range(k):
            H[j : (j + 2), k] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ H[j : (j + 2), k]
            )
            beta_e_1[j : (j + 2)] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ beta_e_1[j : (j + 2)]
            )
        print(np.linalg.cond(H[: (k + 1), : (k + 1)]))
        print(np.linalg.cond(R[:k, :k]))
        t = sp.linalg.solve_triangular(
            H[: (k + 1), : (k + 1)], beta_e_1, check_finite=False
        )
        x = V[:, : (k + 1)] @ t

        if callback is not None:
            callback(x, *callback_args)

        return x

    elif stop_type == 2:
        R[: k + 1, k : k + 2] = R[: k + 1, k : k + 2] @ np.array(
            [[c[k], np.conj(s[k])], [s[k], -np.conj(c[k])]]
        )

        for j in range(k):
            R[j : (j + 2), k] = (
                np.array(
                    [
                        [c_tilde[j], np.conj(s_tilde[j])],
                        [s_tilde[j], -np.conj(c_tilde[j])],
                    ]
                )
                .conj()
                .T
                @ R[j : (j + 2), k]
            )
        print(np.linalg.cond(H[: (k + 1), : (k + 1)]))
        print(np.linalg.cond(R[: (k + 1), : (k + 1)]))
        t = sp.linalg.solve_triangular(
            R[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        t = sp.linalg.solve_triangular(H[: (k + 1), : (k + 1)], t, check_finite=False)
        x = V[:, : (k + 1)] @ t

        if callback is not None:
            callback(x, *callback_args)

        return x


def gmaHres_II(
    A,
    b,
    aHr_tol=1e-10,
    maxiter=None,
    beta_tol=None,
    *,
    callback=None,
    callback_type="norm_aHr",
    callback_args=(),
    **callback_kwargs,
):
    """
    Solve the least squares system

            ``min ||b - Ax||_2``

    with square system matrix A using the GMA\*RES-II algorithm.

    GMA\*RES-II (Generalized Minimal A\*-Residual) is an iterative Krylov subspace
    method for linear systems and square least squares problems. In each iteration,
    the method minimizes the norm of the A*-residual

            ``||A* r_k||_2``,

    where ``r_k = b - A x_k`` is the residual at iteration ``k`` and ``A*`` denotes
    the Hermitian transpose.

    As the method searches for solutions in Krylov subspaces K_k(A,b), the least
    squares problem can only be solved if b has finite least squares grade with
    respect to A. For this, it is sufficient that A is range-Hermitian.

    Parameters
    ----------
    A : array_like or LinearOperator
        Linear operator of shape ``(n, n)`` representing the system matrix.
    b : array_like
        Right-hand side vector of length ``n``.
    aHr_tol : float, optional
        Stopping tolerance for the A*-residual norm ``||A* r_k||``.
        Default is ``1e-10``.
    maxiter : int, optional
        Maximum number of iterations. If ``None``, the default value is
        ``n``.
    beta_tol : float, optional
        Tolerance used to detect Arnoldi breakdown. If ``None``, the default
        value is ``Ar_tol``.
    callback : callable, optional
        Function called once per iteration.
        - If the callback type is ``'norm_aHr'``, the callback is invoked as<br/>
          ``callback(k, norm_aHr_k, *callback_args, **callback_kwargs)``<br/>
          Here, ``norm_aHr_k`` is an estimate of ``||A* r_k||``.
        - If the callback type is ``'x'``, the callback is invoked as<br/>
          ``callback(x_k, k, norm_aHr_k, *callback_args, **callback_kwargs)``<br/>
          This requires the computation of the iterate x_k in each iteration which
          may slow the computation.

        The callback may be used to monitor progress, collect statistics, or
        update a visualization. It may inspect or modify externally
        captured state.
    callback_type : {``'norm_aHr'``, ``'x'``}, optional
        Decides how the callback is invoked.
    callback_args : tuple, optional
        Positional arguments passed to ``callback``.
    **callback_kwargs
        Additional keyword arguments passed to ``callback``.

    Returns
    -------
    x : ndarray
        Approximate solution to Ax = b.
    stats : tuple
        Tuple ``(k, norm_r_k, norm_Ar_k, breakdown)`` where

        * ``k`` (int) is the number of iterations performed;
        * ``norm_aHr_k`` (float) is an estimate of the final A\*-residual norm
          ``||A*(b - A x_k)||_2``;
        * ``breakdown`` (str) describes the termination condition and is
          one of:
                  - ``"maximum number of iterations exceeded"``
                  - ``"beta tolerance reached"``
                  - ``"A*-residual tolerance reached"``

    Notes
    -----
    The method can be seen as a generalization of MinAres [1].

    In exact arithmetic, GMA*RES-II produces the same iterates as GMA*RES-I.
    GMA*RES-II does so at lower computational cost, but with reduced numerical
    stability.

    References
    ----------
    [1] A. Montoison, D. Orban, and M. A. Saunders,
    *MinAres: An Iterative Solver for Symmetric Linear Systems*,
    SIAM Journal on Matrix Analysis and Applications, 46(1), 509–529, 2025.
    """
    dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(b) else np.float64

    n = A.shape[0]

    if maxiter is None:
        maxiter = n

    if beta_tol is None:
        beta_tol = aHr_tol

    V = np.zeros((n, maxiter + 1), dtype=dtype)
    W = np.zeros((n, maxiter + 1), dtype=dtype)
    H = np.zeros((maxiter + 1, maxiter), dtype=dtype)
    s = []
    c = []
    z = np.zeros(maxiter + 1, dtype=dtype)

    W[:, 0] = A.conj().T @ b
    alpha = norm(W[:, 0])
    W[:, 0] /= alpha
    V[:, 0] = b / alpha

    z[0] = alpha

    if callback is not None:
        if callback_type == "norm_aHr":
            callback(0, alpha, *callback_args, **callback_kwargs)
        elif callback_type == "x":
            callback(
                np.zeros(n, dtype=dtype),
                0,
                alpha,
                *callback_args,
                **callback_kwargs,
            )
        else:
            raise ValueError(
                f"callback_type must be either 'norm_aHr' or 'x', not {callback_type}"
            )

    breakdown = "maximum number of iterations exceeded"

    for k in range(maxiter):
        # Arnoldi method
        V[:, k + 1] = A @ V[:, k]
        W[:, k + 1] = A.conj().T @ V[:, k + 1]
        for j in range(k + 1):
            H[j, k] = np.vdot(W[:, j], W[:, k + 1])
            V[:, k + 1] -= H[j, k] * V[:, j]
            W[:, k + 1] -= H[j, k] * W[:, j]

        H[k + 1, k] = norm(W[:, k + 1])
        if np.isclose(H[k + 1, k], 0, atol=beta_tol):
            breakdown = "beta tolerance reached"
            break
        V[:, k + 1] /= H[k + 1, k]
        W[:, k + 1] /= H[k + 1, k]

        # QR factorization of $H_{k+1,k}$
        for j in range(k):
            H[j : (j + 2), k] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ H[j : (j + 2), k]
            )

        _c, _s, H[k, k] = get_givens_rot(H[k, k], H[k + 1, k])
        c.append(_c)
        s.append(_s)
        H[k + 1, k] = 0

        # Computation of z_k
        z[k : (k + 2)] = (
            np.array([[c[k], np.conj(s[k])], [s[k], -np.conj(c[k])]]).conj().T
            @ z[k : (k + 2)]
        )

        norm_aHr = np.abs(z[k + 1])

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                # Solve triangular system and find x_k
                t = sp.linalg.solve_triangular(
                    H[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
                )
                x = V[:, : (k + 1)] @ t

                callback(x, k + 1, norm_aHr, *callback_args, **callback_kwargs)

        if norm_aHr <= aHr_tol:
            breakdown = "A*-residual tolerance reached"
            break

    if breakdown == "beta tolerance reached":
        for j in range(k):
            H[j : (j + 2), k] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ H[j : (j + 2), k]
            )

        t = sp.linalg.solve_triangular(
            H[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        x = V[:, : (k + 1)] @ t

        norm_aHr = 0.0

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                callback(x, k + 1, norm_aHr, *callback_args, **callback_kwargs)

    elif callback is None or callback_type != "x":
        # Solve triangular system and find x_k
        t = sp.linalg.solve_triangular(
            H[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        x = V[:, : (k + 1)] @ t

    return x, (k + 1, norm_aHr, breakdown)


if __name__ == "__main__":

    def normal_equations_residual(x, k, norm_aHr, A, b):
        # print(norm(A.conj().T @ (b - A @ x)))
        print(k, norm_aHr, norm(A.T.conj() @ (b - A @ x)))

    # n = 3
    # A = np.diag(np.linspace(1, n, n, endpoint=True))
    # b = np.ones(n)
    # gmaHres_II(A, b, 100, callback=normal_equations_residual, callback_args=(A, b))

    A = np.random.rand(100, 100)
    b = np.random.rand(100)

    x, info = gmaHres_II(
        A,
        b,
        callback=normal_equations_residual,
        callback_type="x",
        callback_args=(A, b),
    )
    print(info)
    # gmaHres(A, b, 100, normal_equations_residual, A, b)

    # A = np.triu(np.ones((100, 100)))
    # for j in range(99):
    #   A[j+1, j] = 1
    # b = np.zeros(100)
    # b[0] = 1

    # A = np.array([
    #   [2, 4, 0, 0],
    #   [0, 0, 0, 0],
    #   [0, 0, 1, 3],
    #   [0, 0, 0, 0]
    # ])
    # b = np.array([10, 20, 10, 30])

    # print(gmaHres(A, b, 10000, normal_equations_residual, A, b))

    # n = 100
    # m = 10
    # lambda_min = 0.01
    # lambda_max = 100 # cond = 100_000
    # eigenvalues = [lambda_min + (k/(n-1))**m*(lambda_max-lambda_min) for k in range(n)]
    # A = np.diag(eigenvalues)
    # b = np.ones(n)

    # gmaHres(A, b, n, normal_equations_residual, A, b)
