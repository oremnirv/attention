import matplotlib.pyplot as plt
from gp_kernels import rbf_kernel
import matplotlib
from gp_priors import *
import numpy as np


def plot_gp_prior(num_func, data=None, n=50):
    if (data):
        x_te = data
        n = len(x_te)
    else:
        x_te = np.linspace(-5, 5, n).reshape(-1, 1)
    k = rbf_kernel(x_te)
    f_prior = generate_priors(k, n, num_func)
    with matplotlib.rc_context({'figure.figsize': [12, 5]}):
        for idx, func in enumerate(f_prior):
            plt.plot(x_te, func, label='f' + str(idx))
        plt.xlabel('X')
        plt.ylabel('fi(X) = y')
        plt.title('Prior functions drawn from Gausian distribution')
        plt.legend()
        plt.show()


def scatter_tr_vs_predicted_tr():
    with matplotlib.rc_context({'figure.figsize': [10, 2.5]}):
        plt.scatter(x.squeeze(), y)
        plt.scatter(x, μ_tr, c='black')


def plot_nin_five_conf(x_te, μ, σ):
    with matplotlib.rc_context({'figure.figsize': [10, 2.5]}):
        # squeeze() is a numpy function that turns column vectors into simple 1d vectors.
        plt.fill_between(newx, μ.squeeze()-2*σ, μ.squeeze()+2*σ, alpha=.2)
        plt.plot(newx, μ.squeeze())
        plt.scatter(x, y, color='black')
        plt.title(model1.kernel_)
        plt.show()


def d3_surf_plot(x1, x2, y):
    X = np.meshgrid(x1, x2)
    axes = plt.figure().gca(projection='3d')
    axes.plot_surface(X, y)
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    plt.show()
