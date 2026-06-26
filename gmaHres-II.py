import numpy as np
from numpy.linalg import norm
import scipy as sp

def gmaHres(A, b, maxiter=50, callback=None, *callback_args):
	dtype = np.complex128 if np.iscomplexobj(A) or np.iscomplexobj(b) else np.float64
	
	n = A.shape[0]
	
	if callback is not None:
		callback(np.zeros(n), *callback_args)

	V = np.zeros((n, maxiter + 1), dtype=dtype)
	W = np.zeros((n, maxiter + 1), dtype=dtype)
	H = np.zeros((maxiter + 1, maxiter), dtype=dtype)
	s = []
	c = []
	z = np.zeros(maxiter + 1, dtype=dtype)

	W[:, 0] = A.conj().T @ b
	beta = norm(W[:, 0])
	W[:, 0] /= beta
	V[:, 0] = b / beta

	z[0] = beta

	stop_type = 0
	# Keeps track of the reason that the iteration stops:
	#   0 : stop because of reaching iteration maxiter
	#   1 : breakdown of the Arnoldi method

	for k in range(maxiter):
		print(k)

		# Arnoldi method
		V[:, k+1] = A @ V[:, k]
		W[:, k+1] = A.conj().T @ V[:, k+1]
		for j in range(k+1):
			H[j, k] = np.vdot(W[:, j], W[:, k+1])
			V[:, k+1] -= H[j, k] * V[:, j]
			W[:, k+1] -= H[j, k] * W[:, j]

		H[k+1, k] = norm(W[:, k+1])
		if np.isclose(H[k+1, k], 0):
			stop_type = 1
			break
		V[:, k+1] /= H[k+1, k]
		W[:, k+1] /= H[k+1, k]

		# QR factorization of $H_{k+1,k}$
		for j in range(k):
			H[j:(j+2), k] = np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T @ H[j:(j+2), k]

		u_kk = norm((H[k, k], H[k+1, k]))
		c.append(H[k, k] / u_kk)
		s.append(H[k+1, k] / u_kk)
		H[k, k] = u_kk
		H[k+1, k] = 0

		print(np.linalg.cond(H[:(k+1),:(k+1)], p=2))

		# Computation of z_k
		z[k:(k+2)] = np.array([[c[k], np.conj(s[k])], [s[k], -np.conj(c[k])]]).conj().T @ z[k:(k+2)]

		# Solve triangular system and find x_k
		t = sp.linalg.solve_triangular(H[:(k+1), :(k+1)], z[:(k+1)], check_finite=False)
		x = V[:, :(k+1)] @ t

		if callback is not None:
			callback(x, *callback_args)

		print()

	if stop_type == 0:
		return x
	elif stop_type == 1:
		for j in range(k):
			H[j:(j+2), k] = np.array([[c[j], np.conj(s[j])], [s[j], -np.conj(c[j])]]).conj().T @ H[j:(j+2), k]
		print(np.linalg.cond(H[:(k+1), :(k+1)]))
		t = sp.linalg.solve_triangular(H[:(k+1), :(k+1)], z[:(k+1)], check_finite=False)
		x = V[:, :(k+1)] @ t
		
		if callback is not None:
			callback(x, *callback_args)

		return x



if __name__ == "__main__":
	def normal_equations_residual(x, A, b):
		print(norm(A.conj().T @ (b - A @ x)))



	A = np.random.rand(100, 100)
	b = np.random.rand(100)

	gmaHres(A, b, 100)
	# gmaHres(A, b, 100, normal_equations_residual, A, b)
	print("\n\n", np.linalg.cond(A), np.linalg.cond(A)**2)



	# A = np.triu(np.ones((100, 100)))
	# for j in range(99):
	# 	A[j+1, j] = 1
	# b = np.zeros(100)
	# b[0] = 1




	# A = np.array([
	# 	[2, 4, 0, 0],
	# 	[0, 0, 0, 0],
	# 	[0, 0, 1, 3],
	# 	[0, 0, 0, 0]
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

	# gmaHres(A, b, 100, normal_equations_residual, A, b)
