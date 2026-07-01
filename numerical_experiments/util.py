import numpy as np
from numpy.linalg import norm


def optimal_backward_error_fro(A, b, x):
    """
    Given an approximate solution x to the least squares problem ```min || b - A x ||```,
    compute backward error in the Frobenius norm, i.e., compute
         min || E ||_F
         E∈P
    where P is the perturbation space
         P = { E |  ||b - (A + E) x|| = min || b - (A + E) y || }.
                                         y
    The computation is due to

    Waldén, B., Karlson, R. and Sun, J.-G. (1995), Optimal backward perturbation
    bounds for the linear least squares problem. Numer. Linear Algebra Appl., 2: 271-286.
    https://doi.org/10.1002/nla.1680020308
    """
    if norm(x) == 0 and norm(b) != 0:
        return norm(A.T.conj() @ b) / norm(b)
    elif norm(x) == 0 and norm(b) == 0:
        return 0
    else:
        n = A.shape[0]
        r = b - A @ x
        R = (norm(r) * np.eye(n) - np.outer(r, r.conj()) / norm(r)) / norm(x)
        return min(
            np.min(np.linalg.svd(np.hstack((A, R)), compute_uv=False)),
            norm(r) / norm(x),
        )
