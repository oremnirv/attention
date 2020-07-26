import numpy as np

def rbf_kernel(x, sigma=1):
    nrow, ncol = np.shape(x)
    k = np.matmul(x, np.transpose(x)) / (sigma ** 2)
    # print(k)
    d = np.diag(k)
    # print(d)
    # subtract diagonal one time as a row and once as a column
    k1 = k - (d.reshape(-1, 1) / (2 * (sigma ** 2)))
    # print(d.reshape(-1, 1))
    # print('k1: ', k1)
    k2 = k1 - (d.reshape(1, -1) / (2 * (sigma ** 2)))
    k3 = np.exp(k2)
    return k3


def rbf_n_white_kernel(ɾ, λ, σ):
    return (ɾ**2 * gp.kernels.RBF(length_scale=λ) + gp.kernels.WhiteKernel(noise_level=σ))
