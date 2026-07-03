import numpy as np
from numpy.linalg import norm


def get_givens_rot(a, b):
    if a == 0 and b == 0:
        return 1, 0, 0

    mag = norm((a, b))
    return a / mag, b / mag, mag


def MinAres(
    A,
    b,
    r_tol=1e-10,
    Ar_tol=1e-10,
    k_max=None,
    beta_tol=None,
    *,
    callback=None,
    callback_args=(),
    **callback_kwargs,
):
    """
    Solve the Hermitian least squares system

            ``min ||b - Ax||_2``

    using the MINARES algorithm.

    MINARES (Minimum A-Residual) is an iterative Krylov subspace method for
    Hermitian linear systems or least squares problems. In each iteration,
    the method minimizes the norm of the A-residual

            ``||A r_k||_2``,

    where ``r_k = b - A x_k`` is the residual at iteration ``k``.

    Parameters
    ----------
    A : array_like or LinearOperator
        Hermitian linear operator of shape ``(n, n)`` representing the system
        matrix.
    b : array_like
        Right-hand side vector of length ``n``.
    r_tol : float, optional
        Stopping tolerance for the residual norm. Default is ``1e-10``.
    Ar_tol : float, optional
        Stopping tolerance for the A-residual norm ``||A r_k||``.
        Default is ``1e-10``.
    k_max : int, optional
        Maximum number of iterations. If ``None``, the default value is
        ``2 * n``.
    beta_tol : float, optional
        Tolerance used to detect Lanczos breakdown. If ``None``, the default
        value is ``max(r_tol, Ar_tol)``.
    callback : callable, optional
        Function called once per iteration. The callback is invoked as

        ``callback(x_k, k, norm_r_k, norm_Ar_k, *callback_args,
        **callback_kwargs)``

        and may be used to monitor progress, collect statistics, or update a
        visualization. The callback may inspect or modify externally captured
        state.
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
        * ``norm_r_k`` (float) is the final residual norm
          ``||b - A x_k||_2``;
        * ``norm_Ar_k`` (float) is the final A-residual norm
          ``||A(b - A x_k)||_2``;
        * ``breakdown`` (str) describes the termination condition and is
          one of:<br/>
          ``"maximum number of iterations exceeded"``,<br/>
          ``"beta tolerance reached"``,<br/>
          ``"residual tolerance reached"``,<br/>
          ``"A-residual tolerance reached"``.

    Notes
    -----
    The method assumes that A is Hermitian. Convergence is monitored using
    estimates of both the residual norm and the A-residual norm.

    This function is a python translation of the julia implementation in [2].

    References
    ----------
    [1] A. Montoison, D. Orban, and M. A. Saunders,
    *MinAres: An Iterative Solver for Symmetric Linear Systems*,
    SIAM Journal on Matrix Analysis and Applications, 46(1), 509–529, 2025.

    [2] A. Montoison, and D. Orban (2023),
    *Krylov.jl: A Julia basket of hand-picked Krylov methods*,
    Journal of Open Source Software, 8(89), 5187.
    """

    n = A.shape[0]

    if k_max is None:
        k_max = 2 * n
    if beta_tol is None:
        beta_tol = max(r_tol, Ar_tol)

    dtype = (
        np.complex128
        if A.dtype == np.complex128 or b.dtype == np.complex128
        else np.float64
    )

    k = 0
    l = np.inf
    x_k = np.zeros(n, dtype=dtype)

    # First two iterations of Lanczos
    beta_1 = norm(b)
    v_k_plus_1 = np.divide(
        b, beta_1, dtype=dtype
    )  # Cast into complex dtype if b is real, but A is not

    v_k_plus_2 = A @ v_k_plus_1
    alpha_k_plus_1 = np.real(
        np.vdot(v_k_plus_1, v_k_plus_2)
    )  # In exact arithmetic, this should be real. We cast it into real dtype
    v_k_plus_2 -= alpha_k_plus_1 * v_k_plus_1
    beta_k_plus_2 = norm(v_k_plus_2)
    v_k_plus_2 /= beta_k_plus_2

    zeta_bar_bar_k_plus_1 = beta_1 * alpha_k_plus_1
    zeta_bar_k_plus_2 = beta_1 * beta_k_plus_2
    chi_bar_k_plus_1 = beta_1
    lambda_bar_k_plus_1 = alpha_k_plus_1
    gamma_bar_k_plus_1 = beta_k_plus_2

    norm_r_k = chi_bar_k_plus_1
    norm_Ar_k = norm((zeta_bar_bar_k_plus_1, zeta_bar_k_plus_2))

    if callback is not None:
        callback(x_k, k, norm_r_k, norm_Ar_k, *callback_args, **callback_kwargs)

    c_tilde_2k_2 = (
        c_tilde_2k_1
    ) = c_tilde_2k = s_tilde_2k_2 = s_tilde_2k_1 = s_tilde_2k = None

    while norm_r_k > r_tol and norm_Ar_k > Ar_tol and k <= k_max:
        k += 1

        # Update variables
        c_tilde_2k_3, c_tilde_2k_1 = c_tilde_2k_1, None
        c_tilde_2k_4, c_tilde_2k_2, c_tilde_2k = c_tilde_2k_2, c_tilde_2k, None
        s_tilde_2k_3, s_tilde_2k_1 = s_tilde_2k_1, None
        s_tilde_2k_4, s_tilde_2k_2, s_tilde_2k = s_tilde_2k_2, s_tilde_2k, None

        # Update the QR factorization Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
        #                                         [ 0  ]
        #
        # [ α₁ β₂ 0  •  •  •   0  ]      [ λ₁ γ₁ ϵ₁ 0  •  •  0  ]
        # [ β₂ α₂ β₃ •         •  ]      [ 0  λ₂ γ₂ •  •     •  ]
        # [ 0  •  •  •  •      •  ]      [ •  •  λ₃ •  •  •  •  ]
        # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
        # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
        # [ •        •  •  •   βₖ ]      [ •           •  • γₖ₋₁]
        # [ •           •  βₖ  αₖ ]      [ •              •  λₖ ]
        # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
        #
        # Compute the Givens reflection Qₖ.ₖ₊₁
        # [ cₖ  sₖ ] [ λbarₖ γbarₖ   0  ] = [ λₖ    γₖ      ϵₖ   ]
        # [ sₖ -cₖ ] [ βₖ₊₁  αₖ₊₁  βₖ₊₂ ]   [ 0  λbarₖ₊₁ γbarₖ₊₁ ]
        c_k, s_k, lambda_k = get_givens_rot(lambda_bar_k_plus_1, beta_k_plus_2)

        # Compute the direction wₖ, the last column of Wₖ.
        if k == 1:
            # w₁ = v₁ / λ₁
            w_k = v_k_plus_1 / lambda_k
        elif k == 2:
            # w₂ = (v₂ - γ₁w₁) / λ₂
            w_k, w_k_1 = (v_k_plus_1 - gamma_k * w_k) / lambda_k, w_k
        else:
            # wₖ = (vₖ - γₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / λₖ
            w_k, w_k_1 = (
                (v_k_plus_1 - gamma_k * w_k - epsilon_k_1 * w_k_1) / lambda_k,
                w_k,
            )

        # Continue the Lanczos process.
        # AVₖ₊₁ = Vₖ₊₂Tₖ₊₂.ₖ₊₁
        # βₖ₊₂vₖ₊₂ = Avₖ₊₁ - αₖ₊₁vₖ₊₁ - βₖ₊₁vₖ
        if k < l:
            v_k_plus_2, v_k_plus_1 = (
                A @ v_k_plus_2 - beta_k_plus_2 * v_k_plus_1,
                v_k_plus_2,
            )
            alpha_k_plus_1 = np.real(np.vdot(v_k_plus_1, v_k_plus_2))

            v_k_plus_2 -= alpha_k_plus_1 * v_k_plus_1
            beta_k_plus_2 = norm(v_k_plus_2)

            # Detection of early termination
            if np.isclose(beta_k_plus_2, 0, atol=beta_tol):
                l = k + 1
            else:
                v_k_plus_2 /= beta_k_plus_2

        # Apply the Givens reflection Qₖ.ₖ₊₁
        if k < l:
            gamma_k = c_k * gamma_bar_k_plus_1 + s_k * alpha_k_plus_1

        if k == 1:
            epsilon_k = s_k * beta_k_plus_2
        elif k < l - 1:
            epsilon_k, epsilon_k_1 = s_k * beta_k_plus_2, epsilon_k
        else:
            epsilon_k_1 = epsilon_k

        if k < l:
            lambda_bar_k_plus_1 = s_k * gamma_bar_k_plus_1 - c_k * alpha_k_plus_1
            gamma_bar_k_plus_1 = -c_k * beta_k_plus_2

        # Update the QR factorization Nₖ = Q̃ₖ [ Uₖ ].
        #                                     [ 0ᵀ ]
        #
        # [ λ₁  0   •   •   •    •   0  ]      [ μ₁  ϕ₁  ρ₁  0   •    •   0    ]
        # [ γ₁  λ₂  •                •  ]      [ 0   μ₂  ϕ₂  •   •        •    ]
        # [ ϵ₁  γ₂  λ₃  •            •  ]      [ •   •   μ₃  •   •    •   •    ]
        # [ 0   •   •   •   •        •  ]      [ •       •   •   •    •   0    ]
        # [ •   •   •   •   •    •   •  ] = Q̃ₖ [ •           •  μₖ₋₂ ϕₖ₋₂ ρₖ₋₂ ]
        # [ •       •   •   •    •   0  ]      [ •               •   μₖ₋₁ ϕₖ₋₁ ]
        # [ •           •  ϵₖ₋₂ γₖ₋₁ λₖ ]      [ •                    •   μₖ   ]
        # [ •               •   ϵₖ₋₁ γₖ ]      [ 0   •   •   •   •    •   0    ]
        # [ 0  •    •   •   •    0   ϵₖ ]      [ 0   •   •   •   •    •   0    ]
        #
        # If k = 1, we don't have any previous reflection.
        # If k = 2, we apply the reflections Q̃ₖ₊₁.ₖ₋₁ and Q̃ₖ.ₖ₋₁.
        # If k ≥ 3, we apply the reflections Q̃ₖ.ₖ₋₁, Q̃ₖ₊₁.ₖ₋₁ and Q̃ₖ.ₖ₋₂.
        if k == 1:
            mu_bar_k = lambda_k
            gamma_hat_k = gamma_k
        elif k == 2:
            lambda_hat_k = lambda_k
        elif k >= 3:
            rho_k_2 = s_tilde_2k_4 * lambda_k
            lambda_hat_k = -c_tilde_2k_4 * lambda_k

        if k >= 2:
            phi_bar_k_1 = s_tilde_2k_3 * lambda_hat_k
            mu_bar_k = -c_tilde_2k_3 * lambda_hat_k

            if k < l:
                phi_k_1 = c_tilde_2k_2 * phi_bar_k_1 + s_tilde_2k_2 * gamma_k
                gamma_hat_k = s_tilde_2k_2 * phi_bar_k_1 - c_tilde_2k_2 * gamma_k
            elif k == l:
                phi_k_1 = phi_bar_k_1

        if k < l:
            # Compute and apply current Givens reflection Q̃ₖ₊₁.ₖ
            # [ c̃₂ₖ₋₁   s̃₂ₖ₋₁    ] [ μbarₖ ] = [ μbbarₖ ]
            # [ s̃₂ₖ₋₁  -c̃₂ₖ₋₁    ] [ γhatₖ ]   [   0    ]
            # [                1 ] [  ϵₖ   ]   [   ϵₖ   ]
            c_tilde_2k_1, s_tilde_2k_1, mu_bar_bar_k = get_givens_rot(
                mu_bar_k, gamma_hat_k
            )

        if k < l - 1:
            # Compute and apply current Givens reflection Q̃ₖ₊₂.ₖ
            # [ c̃₂ₖ      s̃₂ₖ ] [ μbbarₖ ] = [ μₖ ]
            # [      1       ] [   0    ]   [ 0  ]
            # [ s̃₂ₖ     -c̃₂ₖ ] [   ϵₖ   ]   [ 0  ]
            c_tilde_2k, s_tilde_2k, mu_k = get_givens_rot(mu_bar_bar_k, epsilon_k)
        elif k == l - 1:
            mu_k = mu_bar_bar_k
        elif k == l:
            mu_k = mu_bar_k

        # Update zₖ = (Q̃ₖ)ᵀ(β₁α₁e₁ + β₁β₂e₂)
        if k > 1:
            zeta_k_1 = zeta_k
        if k < l:
            # [ c̃₂ₖ₋₁   s̃₂ₖ₋₁    ] [ ζbbarₖ  ] = [ ζcircₖ   ]
            # [ s̃₂ₖ₋₁  -c̃₂ₖ₋₁    ] [ ζbarₖ₊₁ ]   [ ζbbarₖ₊₁ ]
            # [                1 ] [    0    ]   [    0     ]
            zeta_circ_k = (
                c_tilde_2k_1 * zeta_bar_bar_k_plus_1 + s_tilde_2k_1 * zeta_bar_k_plus_2
            )

        if k < l - 1:
            # [ c̃₂ₖ      s̃₂ₖ ] [ ζcircₖ   ] = [   ζₖ     ]
            # [      1       ] [ ζbbarₖ₊₁ ]   [ ζbbarₖ₊₁ ]
            # [ s̃₂ₖ     -c̃₂ₖ ] [    0     ]   [ ζbarₖ₊₂  ]
            zeta_k = c_tilde_2k * zeta_circ_k
        elif k == l - 1:
            zeta_k = zeta_circ_k
        elif k == l:
            zeta_k = zeta_bar_bar_k_plus_1
            if np.isclose(np.abs(zeta_k), 0, atol=beta_tol) and np.isclose(
                np.abs(lambda_k), 0, atol=beta_tol
            ):
                zeta_k = 0
        if k < l:
            zeta_bar_bar_k_plus_1 = (
                s_tilde_2k_1 * zeta_bar_bar_k_plus_1 - c_tilde_2k_1 * zeta_bar_k_plus_2
            )
        if k < l - 1:
            zeta_bar_k_plus_2 = s_tilde_2k * zeta_circ_k

        # Compute the direction dₖ, the last column of Dₖ.
        if k == 1:
            # d₁ = w₁ / μ₁
            d_k = w_k / mu_k
        elif k == 2:
            # d₂ = (w₂ - ϕ₁d₁) / μ₂
            d_k, d_k_1 = (w_k - phi_k_1 * d_k) / mu_k, d_k
        else:
            # dₖ = (wₖ - ϕₖ₋₁dₖ₋₁ - ρₖ₋₂dₖ₋₂) / μₖ
            d_k, d_k_1 = (w_k - phi_k_1 * d_k - rho_k_2 * d_k_1) / mu_k, d_k

        # Update xₖ = Vₖyₖ = Dₖzₖ = xₖ₋₁ + ζₖdₖ
        x_k += zeta_k * d_k

        # Update ‖Arₖ‖ estimate
        if k < l - 1:
            norm_Ar_k = norm((zeta_bar_bar_k_plus_1, zeta_bar_k_plus_2))
        elif k == l - 1:
            norm_Ar_k = np.abs(zeta_bar_bar_k_plus_1)
        else:
            norm_Ar_k = 0  # TODO: Is this good?

        # Update the LQ factorization Uₖ = L̂ₖP̂ₖ
        #
        # [ μ₁  ϕ₁  ρ₁  0   •    •   0    ]   [ ψ₁   0    •    •     •      •       0  ]
        # [ 0   μ₂  ϕ₂  •   •        •    ]   [ θ₁   ψ₂   •                         •  ]
        # [ •   •   μ₃  •   •    •   •    ]   [ ω₁   θ₂   ψ₃   •                    •  ]
        # [ •       •   •   •    •   0    ] = [ 0    •    •    •     •              •  ] P̂ₖ
        # [ •           •  μₖ₋₂ ϕₖ₋₂ ρₖ₋₂ ]   [ •    •    •    •   ψₖ₋₂     •       •  ]
        # [ •               •   μₖ₋₁ ϕₖ₋₁ ]   [ •         •    •   θₖ₋₂  ψbbarₖ₋₁   0  ]
        # [ 0   •   •   •   •    0   μₖ   ]   [ 0    •    •    0   ωₖ₋₂  θbarₖ₋₁  ψbarₖ]
        #
        # and solve L̂ₖtₖ = zₖ.
        if k == 1:
            psi_bar_k = mu_k
            tau_bar_k = zeta_k / psi_bar_k
        elif k == 2:
            # [ ψbar₁  ϕ₁ ] [ ĉ₁   ŝ₁ ] = [ ψbbar₁    0   ]
            # [   0    μ₂ ] [ ŝ₁  -ĉ₁ ]   [ θbar₁   ψbar₂ ]
            c_hat_2k_3, s_hat_2k_3, psi_bar_bar_k_1 = get_givens_rot(psi_bar_k, phi_k_1)
            theta_bar_k_1 = s_hat_2k_3 * mu_k
            psi_bar_k = -c_hat_2k_3 * mu_k

            tau_bar_bar_k_1 = zeta_k_1 / psi_bar_bar_k_1
            tau_bar_k = (zeta_k - theta_bar_k_1 * tau_bar_bar_k_1) / psi_bar_k
            xi_k = zeta_k
        else:
            # [ ψbbarₖ₋₂   0     ρₖ₋₂ ] [ ĉ₂ₖ₋₄      ŝ₂ₖ₋₄ ]   [ ψₖ₋₂     0     0  ]
            # [ θbarₖ₋₂  ψbarₖ₋₁ ϕₖ₋₁ ] [        1         ] = [ θₖ₋₂  ψbarₖ₋₁  δₖ ]
            # [   0        0      μₖ  ] [ ŝ₂ₖ₋₄     -ĉ₂ₖ₋₄ ]   [ ωₖ₋₂     0     ηₖ ]
            c_hat_2k_4, s_hat_2k_4, psi_k_2 = get_givens_rot(psi_bar_bar_k_1, rho_k_2)
            theta_k_2 = c_hat_2k_4 * theta_bar_k_1 + s_hat_2k_4 * phi_k_1
            delta_k = s_hat_2k_4 * theta_bar_k_1 - c_hat_2k_4 * phi_k_1
            omega_k_2 = s_hat_2k_4 * mu_k
            eta_k = -c_hat_2k_4 * mu_k

            tau_k_2 = tau_bar_bar_k_1 * psi_bar_bar_k_1 / psi_k_2

            # [ ψₖ₋₂     0     0  ] [ 1                ]   [ ψₖ₋₂    0         0   ]
            # [ θₖ₋₂  ψbarₖ₋₁  δₖ ] [    ĉ₂ₖ₋₃   ŝ₂ₖ₋₃ ] = [ θₖ₋₂  ψbbarₖ₋₁    0   ]
            # [ ωₖ₋₂     0     ηₖ ] [    ŝ₂ₖ₋₃  -ĉ₂ₖ₋₃ ]   [ ωₖ₋₂  θbarₖ₋₁   ψbarₖ ]
            c_hat_2k_3, s_hat_2k_3, psi_bar_bar_k_1 = get_givens_rot(psi_bar_k, delta_k)
            theta_bar_k_1 = s_hat_2k_3 * eta_k
            psi_bar_k = -c_hat_2k_3 * eta_k

            tau_bar_bar_k_1 = (xi_k - theta_k_2 * tau_k_2) / psi_bar_bar_k_1
            xi_k = zeta_k - omega_k_2 * tau_k_2
            tau_bar_k = (xi_k - theta_bar_k_1 * tau_bar_bar_k_1) / psi_bar_k

        # Update (χ₁, ..., χₖ, χbarₖ₊₁) = (Qₖ)ᵀβ₁e₁
        if k > 1:
            chi_k_1 = chi_k
        # [ cₖ  sₖ ] [ χbarₖ ] = [    χₖ   ]
        # [ sₖ -cₖ ] [   0   ]   [ χbarₖ₊₁ ]
        chi_k = c_k * chi_bar_k_plus_1
        chi_bar_k_plus_1 = s_k * chi_bar_k_plus_1

        # Update pₖ₊₁ = [ P̂ₖ  0 ](Qₖ)ᵀβ₁e₁
        #               [ 0   1 ]
        if k == 1:
            pi_bar_k = chi_k
        elif k == 2:
            # [ ĉ₁   ŝ₁ ] [ π₁ ] = [ πbbar₁ ]
            # [ ŝ₁  -ĉ₁ ] [ χ₂ ]   [ πbar₂  ]
            pi_bar_bar_k_1 = c_hat_2k_3 * chi_k_1 + s_hat_2k_3 * chi_k
            pi_bar_k = s_hat_2k_3 * chi_k_1 - c_hat_2k_3 * chi_k
        else:
            # [ ĉ₂ₖ₋₄      ŝ₂ₖ₋₄ ] [ πbbarₖ₋₂ ]   [ πₖ₋₂    ]
            # [        1         ] [ πbarₖ₋₁  ] = [ πbarₖ₋₁ ]
            # [ ŝ₂ₖ₋₄     -ĉ₂ₖ₋₄ ] [   χₖ     ]   [  υₖ     ]
            upsilon_k = s_hat_2k_4 * pi_bar_bar_k_1 - c_hat_2k_4 * chi_k

            # [ 1                ] [ πₖ₋₂    ]   [ πₖ₋₂     ]
            # [    ĉ₂ₖ₋₃   ŝ₂ₖ₋₃ ] [ πbarₖ₋₁ ] = [ πbbarₖ₋₁ ]
            # [    ŝ₂ₖ₋₃  -ĉ₂ₖ₋₃ ] [  υₖ     ]   [ πbarₖ    ]
            pi_bar_bar_k_1 = c_hat_2k_3 * pi_bar_k + s_hat_2k_3 * upsilon_k
            pi_bar_k = s_hat_2k_3 * pi_bar_k - c_hat_2k_3 * upsilon_k

        # Update ‖rₖ‖ estimate
        # ‖rₖ‖ = √((πₖ₋₁ - τₖ₋₁)² + (πₖ - τₖ)² + (πₖ₊₁)²)
        if k == 1:
            norm_r_k = norm((pi_bar_k - tau_bar_k, chi_bar_k_plus_1))
        else:
            norm_r_k = norm(
                (
                    pi_bar_bar_k_1 - tau_bar_bar_k_1,
                    pi_bar_k - tau_bar_k,
                    chi_bar_k_plus_1,
                )
            )

        if callback is not None:
            callback(x_k, k, norm_r_k, norm_Ar_k, *callback_args, **callback_kwargs)

    if k > k_max:
        breakdown = "maximum number of iterations exceeded"
    elif k == l:
        breakdown = "beta tolerance reached"
    elif norm_r_k <= r_tol:
        breakdown = "residual tolerance reached"
    elif norm_Ar_k <= Ar_tol:
        breakdown = "A-residual tolerance reached"

    return x_k, (k, norm_r_k, norm_Ar_k, breakdown)
