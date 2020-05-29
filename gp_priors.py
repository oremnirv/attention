import numpy as np

def generate_priors(kernel, num_points, num_func):
	L = np.linalg.cholesky(k + np.exp(-6) * np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, num_func)))
    return f_prior.reshape(num_func, -1)
