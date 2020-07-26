import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


def generate_priors(kernel, num_points, num_func, noise = 1e-10):
	L = np.linalg.cholesky(kernel + noise * np.eye(num_points))
	f_prior = np.dot(L, np.random.normal(size=(num_points, num_func)))
	return f_prior.reshape(num_func, -1)



def prior_gen_sklearn(x, kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))):
	'''

	'''
	gp = GaussianProcessRegressor(kernel=kernel)
	y_mean, y_std = gp.predict(x, return_std=True)

	return y_mean, y_std