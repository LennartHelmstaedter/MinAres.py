import numpy as np
from numpy.linalg import norm
import scipy as sp

from MinAres import get_givens_rot


def gmaHres_I_U(
    A,
    b,
    aHr_tol=1e-10,
    maxiter=None,
    arnoldi_tol=None,
    gram_schmidt_tol=None,
    *,
    callback=None,
    callback_type="norm_aHr",
    callback_args=(),
    **callback_kwargs,
):
    """
    This is the same as gmaHres_I, except that if the callback_type is ``"x"``, the
    callback additionally receives the triangular matrices Uₖ and Ũₖ as input.
    """
    dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(b) else np.float64

    n = A.shape[0]

    if maxiter is None:
        maxiter = n

    if arnoldi_tol is None:
        arnoldi_tol = aHr_tol

    if gram_schmidt_tol is None:
        gram_schmidt_tol = aHr_tol

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
    norm_aHr = z[0]

    if callback is not None:
        if callback_type == "norm_aHr":
            callback(0, norm_aHr, *callback_args, **callback_kwargs)
        elif callback_type == "x":
            x = np.zeros(n, dtype=dtype)
            callback(x, 0, norm_aHr, None, None, *callback_args, **callback_kwargs)
        else:
            raise ValueError(
                f"callback_type must be either 'norm_aHr' or 'x', not {callback_type}."
            )

    breakdown = "maximum number of iterations exceeded"

    for k in range(maxiter):
        # Continue the Arnoldi process.
        # AVₖ = Vₖ₊₁Hₖ₊₁.ₖ
        #                     k
        # hₖ₊₁.ₖvₖ₊₁ = Avₖ -  ∑ hⱼₖ vⱼ
        #                    j=1
        V[:, k + 1] = A @ V[:, k]
        for j in range(k + 1):
            H[j, k] = np.vdot(V[:, j], V[:, k + 1])
            V[:, k + 1] -= H[j, k] * V[:, j]
        H[k + 1, k] = norm(V[:, k + 1])
        # Detection of early termination
        if np.isclose(H[k + 1, k], 0, atol=arnoldi_tol):
            breakdown = "arnoldi_tolerance reached"
        else:
            V[:, k + 1] /= H[k + 1, k]

        # Update the QR factorization Hₖ₊₁.ₖ = Qₖ [ Uₖ ].
        #                                         [ 0  ]
        # Apply the previous Givens reflections to the last column
        # [  u₁ₖ   ]       [  h₁₁   ]
        # [   •    ] = Qₖᴴ [   •    ]
        # [ uₖ₋₁.ₖ ]       [ hₖ₋₁.ₖ ]
        # [  u°ₖₖ  ]       [  hₖₖ   ]
        for j in range(k):
            H[j : (j + 2), k] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ H[j : (j + 2), k]
            )

        # Compute and apply current Givens reflection
        # [ c̅ₖ   s̅ₖ ] [  u°ₖₖ  ] = [ uₖₖ ]
        # [ sₖ  -cₖ ] [ hₖ₊₁.ₖ ]   [  0  ]
        _c, _s, H[k, k] = get_givens_rot(H[k, k], H[k + 1, k])
        c.append(_c)
        s.append(_s)
        H[k + 1, k] = 0

        if breakdown == "arnoldi_tolerance reached":
            break

        # Continue the Gram–Schmidt process.
        # AᴴVₖ₊₁ = Wₖ₊₁Rₖ₊₁
        #                         k
        # rₖ₊₁.₊₁wₖ₊₁ = Aᴴvₖ₊₁ -  ∑ rⱼₖ wⱼ
        #                        j=1
        W[:, k + 1] = A.conj().T @ V[:, k + 1]
        for j in range(k + 1):
            R[j, k + 1] = np.vdot(W[:, j], W[:, k + 1])
            W[:, k + 1] -= R[j, k + 1] * W[:, j]
        R[k + 1, k + 1] = norm(W[:, k + 1])
        # Detection of early termination
        if np.isclose(R[k + 1, k + 1], 0, atol=gram_schmidt_tol):
            breakdown = "gram_schmidt_tolerance reached"
        else:
            W[:, k + 1] /= R[k + 1, k + 1]

        # Compute H̃ₖ₊₁.ₖ by applying the Givens rotations from the right
        # [  h̃₁ₖ    h̃°₁.ₖ₊₁   ]   [ h̃°₁ₖ   r₁.ₖ₊₁  ]
        # [   •         •     ] = [  •        •    ]
        # [  h̃ₖₖ    h̃°ₖ.ₖ₊₁   ]   [ h̃°ₖₖ   rₖ.ₖ₊₁  ]
        # [ h̃ₖ₊₁.ₖ  h̃°ₖ₊₁.ₖ₊₁ ]   [  0    rₖ₊₁.ₖ₊₁ ]
        R[: (k + 2), k : (k + 2)] = R[: (k + 2), k : (k + 2)] @ np.array(
            [[c[k], np.conj(s[k])], [s[k], -np.conj(c[k])]]
        )

        # Update the QR factorization H̃ₖ₊₁.ₖ = Qₖ [ Ũₖ ].
        #                                         [ 0  ]
        #
        # Apply the previous Givens reflections to the last column
        # [  ũ₁ₖ   ]       [  h̃₁₁   ]
        # [   •    ] = Q̃ₖᴴ [   •    ]
        # [ ũₖ₋₁.ₖ ]       [ h̃ₖ₋₁.ₖ ]
        # [  ũ°ₖₖ  ]       [  h̃ₖₖ   ]
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

        # Compute and apply current Givens reflection
        # [ c̃̅ₖ   s̅̃ₖ ] [  ũ°ₖₖ  ] = [ ũₖₖ ]
        # [ s̃ₖ  -c̃ₖ ] [ h̃ₖ₊₁.ₖ ]   [  0  ]
        _c, _s, R[k, k] = get_givens_rot(R[k, k], R[k + 1, k])
        c_tilde.append(_c)
        s_tilde.append(_s)
        R[k + 1, k] = 0

        # Update zₖ = Q̃ₖ₊₁ᴴ (r₁₁ β e₁)
        # [ c̃̅ₖ   s̅̃ₖ ] [ ζ°ₖ ] = [  ζₖ   ]
        # [ s̃ₖ  -c̃ₖ ] [  0  ]   [ ζ°ₖ₊₁ ]
        z[k : (k + 2)] = (
            np.array(
                [[c_tilde[k], np.conj(s_tilde[k])], [s_tilde[k], -np.conj(c_tilde[k])]]
            )
            .conj()
            .T
            @ z[k : (k + 2)]
        )

        if breakdown == "gram_schmidt_tolerance reached":
            break

        norm_aHr = np.abs(z[k + 1])

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                # Solve triangular systems and find xₖ
                # Compute tₖ = Uₖ⁻¹ Ũₖ⁻¹ zₖ
                t = sp.linalg.solve_triangular(
                    R[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
                )
                t = sp.linalg.solve_triangular(
                    H[: (k + 1), : (k + 1)], t, check_finite=False, overwrite_b=True
                )
                # Compute xₖ = Vₖ tₖ
                x = V[:, : (k + 1)] @ t

                callback(
                    x,
                    k + 1,
                    norm_aHr,
                    H[: (k + 1), : (k + 1)],
                    R[: (k + 1), : (k + 1)],
                    *callback_args,
                    **callback_kwargs,
                )

    if breakdown == "arnoldi_tolerance reached":
        if np.isclose(H[k, k], 0, atol=arnoldi_tol):
            # The matrix Uₖ is singular. This only happens when b does not have finite
            # least squares grade with respect to A. Among the minimizers in the Krylov
            # subspace, we compute that with minimum norm.
            #
            # tₖ = [ p - γ q ]
            #      [    γ    ]
            # Here, with p = Uₖ₋₁⁻¹ Ũₖ₋₁⁻¹ zₖ₋₁, q = Uₖ₋₁⁻¹ u, u is the last column of Uₖ
            # and γ = qᴴp / ( 1 + ‖q‖² )

            p = sp.linalg.solve_triangular(R[:k, :k], z[:k], check_finite=False)
            p = sp.linalg.solve_triangular(
                H[:k, :k], p, check_finite=False, overwrite_b=True
            )
            q = sp.linalg.solve_triangular(H[:k, :k], H[:k, k], check_finite=False)
            gamma = np.vdot(q, p) / (1 + np.real(np.vdot(q, q)))

            t = np.append(p - gamma * q, gamma)

            # Compute xₖ = Vₖ tₖ
            x = V[:, : (k + 1)] @ t

            norm_aHr = np.abs(z[k])
        else:
            # Solve the linear system Uₖ tₖ = Qₖᴴ(β e₁)
            #
            # Compute Qₖᴴ(β e₁)
            beta_e_1 = np.zeros(k + 2, dtype=dtype)
            beta_e_1[0] = beta

            for j in range(k + 1):
                beta_e_1[j : (j + 2)] = (
                    np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                    @ beta_e_1[j : (j + 2)]
                )

            # Solve Uₖ⁻¹ Qₖᴴ(β e₁)
            t = sp.linalg.solve_triangular(
                H[: (k + 1), : (k + 1)], beta_e_1[: (k + 1)], check_finite=False
            )

            # Compute xₖ = Vₖ tₖ
            x = V[:, : (k + 1)] @ t

            norm_aHr = 0.0

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                callback(
                    x,
                    k + 1,
                    norm_aHr,
                    H[: (k + 1), : (k + 1)],
                    None,
                    *callback_args,
                    **callback_kwargs,
                )

    elif breakdown == "gram_schmidt_tolerance reached":
        t = sp.linalg.solve_triangular(
            R[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        t = sp.linalg.solve_triangular(
            H[: (k + 1), : (k + 1)], t, check_finite=False, overwrite_b=True
        )
        x = V[:, : (k + 1)] @ t

        norm_aHr = 0.0

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                callback(
                    x,
                    k + 1,
                    norm_aHr,
                    H[: (k + 1), : (k + 1)],
                    R[: (k + 1), : (k + 1)],
                    *callback_args,
                    **callback_kwargs,
                )

    elif callback is None or callback_type != "norm_aHr":
        # Solve triangular systems and find xₖ
        # Compute tₖ = Uₖ⁻¹ Ũₖ⁻¹ zₖ
        t = sp.linalg.solve_triangular(
            R[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        t = sp.linalg.solve_triangular(
            H[: (k + 1), : (k + 1)], t, check_finite=False, overwrite_b=True
        )
        # Compute xₖ = Vₖ tₖ
        x = V[:, : (k + 1)] @ t

    return x, (k + 1, norm_aHr, breakdown)


def gmaHres_II_U(
    A,
    b,
    aHr_tol=1e-10,
    maxiter=None,
    arnoldi_tol=None,
    *,
    callback=None,
    callback_type="norm_aHr",
    callback_args=(),
    **callback_kwargs,
):
    """
    This is the same as gmaHres_II, except that if the callback_type is ``"x"``, the
    callback additionally receives the triangular matrix Uₖ input.
    """
    dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(b) else np.float64

    n = A.shape[0]

    if maxiter is None:
        maxiter = n

    if arnoldi_tol is None:
        arnoldi_tol = aHr_tol

    V = np.zeros((n, maxiter + 1), dtype=dtype)
    W = np.zeros((n, maxiter + 1), dtype=dtype)
    H = np.zeros((maxiter + 1, maxiter), dtype=dtype)
    s = []
    c = []
    z = np.zeros(maxiter + 1, dtype=dtype)
    l = np.inf

    W[:, 0] = A.conj().T @ b
    alpha = norm(W[:, 0])
    W[:, 0] /= alpha
    V[:, 0] = b / alpha

    z[0] = alpha
    norm_aHr = alpha

    if callback is not None:
        if callback_type == "norm_aHr":
            callback(0, norm_aHr, *callback_args, **callback_kwargs)
        elif callback_type == "x":
            callback(
                np.zeros(n, dtype=dtype),
                0,
                norm_aHr,
                None,
                *callback_args,
                **callback_kwargs,
            )
        else:
            raise ValueError(
                f"callback_type must be either 'norm_aHr' or 'x', not {callback_type}."
            )

    breakdown = "maximum number of iterations exceeded"

    for k in range(maxiter):
        # Continue the Arnoldi process in the AAᴴ-inner product
        # AᴴAVₖ = Wₖ₊₁Hₖ₊₁.ₖ
        #                     k
        # hₖ₊₁.ₖvₖ₊₁ = Avₖ -  ∑ hⱼₖ vⱼ
        #                    j=1
        #
        #                       k
        # hₖ₊₁.ₖwₖ₊₁ = AᴴAvₖ -  ∑ hⱼₖ wⱼ
        #                      j=1
        V[:, k + 1] = A @ V[:, k]
        W[:, k + 1] = A.conj().T @ V[:, k + 1]
        for j in range(k + 1):
            H[j, k] = np.vdot(W[:, j], W[:, k + 1])
            V[:, k + 1] -= H[j, k] * V[:, j]
            W[:, k + 1] -= H[j, k] * W[:, j]
        H[k + 1, k] = norm(W[:, k + 1])
        # Detection of early termination
        if np.isclose(H[k + 1, k], 0, atol=arnoldi_tol):
            breakdown = "arnoldi_tolerance reached"
        else:
            V[:, k + 1] /= H[k + 1, k]
            W[:, k + 1] /= H[k + 1, k]

        # Update the QR factorization Hₖ₊₁.ₖ = Qₖ [ Uₖ ].
        #                                         [ 0  ]
        # Apply the previous Givens reflections to the last column
        # [  u₁ₖ   ]       [  h₁₁   ]
        # [   •    ] = Qₖᴴ [   •    ]
        # [ uₖ₋₁.ₖ ]       [ hₖ₋₁.ₖ ]
        # [  u°ₖₖ  ]       [  hₖₖ   ]
        for j in range(k):
            H[j : (j + 2), k] = (
                np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T
                @ H[j : (j + 2), k]
            )

        # Compute and apply current Givens reflection
        # [ c̅ₖ   s̅ₖ ] [  u°ₖₖ  ] = [ uₖₖ ]
        # [ sₖ  -cₖ ] [ hₖ₊₁.ₖ ]   [  0  ]
        _c, _s, H[k, k] = get_givens_rot(H[k, k], H[k + 1, k])
        c.append(_c)
        s.append(_s)
        H[k + 1, k] = 0

        # Update zₖ = Qₖ₊₁ᴴ (α e₁)
        # [ c̅ₖ   s̅ₖ ] [ ζ°ₖ ] = [  ζₖ   ]
        # [ sₖ  -cₖ ] [  0  ]   [ ζ°ₖ₊₁ ]
        z[k : (k + 2)] = (
            np.array([[c[k], np.conj(s[k])], [s[k], -np.conj(c[k])]]).conj().T
            @ z[k : (k + 2)]
        )

        if breakdown == "arnoldi_tolerance reached":
            break

        norm_aHr = np.abs(z[k + 1])

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                # Solve triangular systems and find xₖ
                # Compute tₖ = Uₖ⁻¹ zₖ
                t = sp.linalg.solve_triangular(
                    H[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
                )
                # Compute xₖ = Vₖ tₖ
                x = V[:, : (k + 1)] @ t

                callback(
                    x,
                    k + 1,
                    norm_aHr,
                    H[: (k + 1), : (k + 1)],
                    *callback_args,
                    **callback_kwargs,
                )

        if norm_aHr <= aHr_tol:
            breakdown = "A*-residual tolerance reached"
            break

    if breakdown == "arnoldi_tolerance reached":
        if np.isclose(H[k, k], 0, atol=arnoldi_tol):
            # The matrix Uₖ is singular. This only happens when b does not have finite
            # least squares grade with respect to A. Among the minimizers in the Krylov
            # subspace, we compute that with minimum norm.
            #
            # tₖ = [ p - γ q ]
            #      [    γ    ]
            # Here, with p = Uₖ₋₁⁻¹ zₖ₋₁, q = Uₖ₋₁⁻¹ u, u is the last column of Uₖ
            # and γ = qᴴp / ( 1 + ‖q‖² )
            p = sp.linalg.solve_triangular(H[:k, :k], z[:k], check_finite=False)
            q = sp.linalg.solve_triangular(H[:k, :k], H[:k, k], check_finite=False)
            gamma = np.vdot(q, p) / (1 + np.real(np.vdot(q, q)))
            t = np.append(p - gamma * q, gamma)
            # Compute xₖ = Vₖ tₖ
            x = V[:, : (k + 1)] @ t

            norm_aHr = norm(z[k : (k + 2)])
        else:
            # Solve triangular systems and find xₖ
            # Compute tₖ = Uₖ⁻¹ zₖ
            t = sp.linalg.solve_triangular(
                H[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
            )
            # Compute xₖ = Vₖ tₖ
            x = V[:, : (k + 1)] @ t

            norm_aHr = 0.0

        if callback is not None:
            if callback_type == "norm_aHr":
                callback(k + 1, norm_aHr, *callback_args, **callback_kwargs)
            elif callback_type == "x":
                callback(
                    x,
                    k + 1,
                    norm_aHr,
                    H[: (k + 1), : (k + 1)],
                    *callback_args,
                    **callback_kwargs,
                )

    elif callback is None or callback_type != "x":
        # Solve triangular systems and find xₖ
        # Compute tₖ = Uₖ⁻¹ zₖ
        t = sp.linalg.solve_triangular(
            H[: (k + 1), : (k + 1)], z[: (k + 1)], check_finite=False
        )
        # Compute xₖ = Vₖ tₖ
        x = V[:, : (k + 1)] @ t

    return x, (k + 1, norm_aHr, breakdown)
