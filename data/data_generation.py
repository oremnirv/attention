###########################
# Author: Omer Nivron
###########################
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF
from data import gp_kernels
from data import gp_priors


class EmbderMap:
    """This is a class to take continuous values an return
    their mapping to indices --> used in embedding layers
    """

    def __init__(self, num_vars, grids):
        super(EmbderMap, self).__init__()
        self.num_vars = num_vars
        assert num_vars == len(grids)
        self.grid = []
        self.idxs = []
        for ix, val in enumerate(grids):  # each val is a list
            self.grid.append(val)  # list of lists

    def map_value_to_grid(self, var):
        """
        map each float in var to an integer,
        starting from 200 (arbitrary choice in order to avoid zero)

        :param var: list of floats
        :return:
        list of lists where each element is a mapping
        of the appropriate sequence.
        """
        n = len(self.idxs)
        g = self.grid[n]  # list of floats
        idx = []
        for i in var:
            if n == 0:

                if i > max(g):
                    idx.append(200 + (len(g) - 1))
                elif i < min(g):
                    idx.append(200 + 0)
                else:
                    idx.append(200 + np.where(i <=g)[0][0])
            else:
                m = max(self.idxs[n - 1]) + 1
                if i > max(g):
                    idx.append(200 + (len(g) - 1) + m)
                elif i < min(g):
                    idx.append(200 + 0 + m)
                else:
                    idx.append(200 + np.where(i < g)[0][0] + m)

        return self.idxs.append(np.array(idx))


def data_gen(num_obs, tr_percent=0.8, seq_len=200, extarpo=False, extarpo_num=19, p_order=0.5,
             ordered=False,
             kernel='rbf', noise=False, diff_x=False,
             grid_d=None):
    """
    Generator for training a GPT inspired netowrk.
    -----------------------
    Parameters:
    num_obs (int): how many observations to generate
    tr_percent (float): daefult 0.8
    seq_len (int): daefult 59
    -----------------------
    Returns:
    x_tr (np array): the first rows * tr_percent from the x generated values
    xte (np array): all rows of x not chosen for training
    y_tr (np array): the first rows * tr_percent from the f_prior generated values
    y_te (np array): all rows of f_prior not chosen for training
    df_tr (np array): positions and targets combined (training) 
    df_te (np array): positions and targets combined (testing) 
    """
    if grid_d is None:
        grid_d = [1, 15.1, 0.1]
    df = np.zeros((num_obs * 2, seq_len))
    em_idx = np.zeros((num_obs, seq_len))
    em = []
    rows = df.shape[0]
    tr_rows = int(tr_percent * rows)
    tr_rows = tr_rows if tr_rows % 2 == 0 else tr_rows + 1
    grid = np.arange(*grid_d)
    for i in range(0, num_obs * 2, 2):
        if (i >= tr_rows) & extarpo & diff_x:
            x = np.concatenate((np.random.uniform(5, 15, size=(
                1, seq_len - extarpo_num)), np.random.uniform(15.1, 20, size=(1, extarpo_num))), axis=1)
        elif diff_x:
            x = np.random.uniform(5, 15, size=(1, seq_len))
        else:
            x = np.random.permutation(np.linspace(5, 15, seq_len))
            x = x.reshape(1, -1)

        idx = EmbderMap(1, [grid])
        idx.map_value_to_grid(x[0])

        if (p_order > 0) & (np.random.binomial(1, p_order) == 0):
            x = np.sort(x)
            ordered = False
        if ordered:
            x = np.sort(x)
        if kernel == 'rbf':
            k = gp_kernels.rbf_kernel(x.reshape(-1, 1))
            f_prior = gp_priors.generate_priors(k,  1)
        elif kernel == 'periodic':
            k = 1.0 * ExpSineSquared(length_scale=1.0,
                                     periodicity=3.0, length_scale_bounds=(0.1, 10.0))
            gp = GaussianProcessRegressor(kernel=k)
            f_prior = np.squeeze(gp.sample_y(x.reshape(-1, 1)))

        if noise:
            k = WhiteKernel(.05)
            gp = GaussianProcessRegressor(kernel=k)
            f_prior = f_prior + gp.sample_y(x, seq_len)
        else:
            pass
        df[i, :x.shape[1]] = x
        df[i + 1, :x.shape[1]] = f_prior
        em_idx[int(i / 2), :] = idx.idxs[0]

    df_tr = df[:tr_rows, :]
    df_te = df[tr_rows:, :]
    em_tr = em_idx[:int(tr_rows / 2), :]
    em_te = em_idx[int(tr_rows / 2):, :]
    em.append(em_tr)
    em.append(em_te)
    # get all even rows
    x_tr = df_tr[::2, :]
    x_te = df_te[::2, :]
    # get all odd rows
    y_tr = df_tr[1::2, :]
    y_te = df_te[1::2, :]

    return x_tr, x_te, y_tr, y_te, df_tr, df_te, em


def data_gen2d(num_obs, tr_percent=0.8, seq_len=200, bias='const', kernel='rbf',
               grid_d=[[1, 15.1, 0.05], [-5, 5, 0.05]], noise=False,
               ordered=False, inp_d=1, p_order=0.5):
    """
    Generator for training a GPT inspired netowrk for pairs of sequences.
    :param num_obs: (int)
    :param tr_percent: float [0, 1]
    :param seq_len: (int)
    :param bias: 'const' for constant shift, 'rbf' for rbf shift -- the shift will be different in different locations
    :param kernel: currently only 'rbf' available
    :param grid_d: (list of lists) each list is comprised of three elements: 1) min value to be emebeded
    2) max value 3) step size --> all three are the input to an np.arange() argument
    :param noise: (bool) if True adds White noise kernel to each value with .1 noise
    :param ordered: (bool) True if all the created arrays should be sorted
    :param inp_d: (int) number of dimensions to embed. inp_d=1, will enable a list of two numpy arrays.
    :param p_order: float [0, 1] indicating what percentage of rows should be sorted
    :return:
    x_tr (np array): the first rows * tr_percent from the x generated values
    x_te (np array): all rows of x not chosen for training
    y_tr (np array): the first rows * tr_percent from the f_prior generated values
    y_te (np array): all rows of f_prior not chosen for training
    df_tr (np array): positions and targets combined (training)
    df_te (np array): positions and targets combined (testing)
    em (list of np.arrays): em will be a list with 2x(inp_d + 1) elements
    the even elements associated with training and odd ones with testing
    from the even elements, the first of each block of two is the index associated with x-val
    and the second is the 0/1 value associated with the pair sequence member

    To check how the output looks like for a toy example run in main:
    x_tr, x_te, y_tr, y_te, df_tr, df_te, em = data_gen2d(5, 0.8, 3, grid_d=[[1, 15.1, 0.05]], inp_d=1)
    print((em))
    em_y list of np.arrays):
    """
    df = np.zeros((num_obs * 2, seq_len * 2))
    em = []; em_y = []
    em_idx = [np.zeros((num_obs, seq_len * 2)) for _ in range(inp_d + 1)] # list of np.arrays
    em_y_idx = np.zeros((num_obs, seq_len * 2))
    rows = df.shape[0]
    tr_rows = int(tr_percent * rows)
    tr_rows = tr_rows if tr_rows % 2 == 0 else tr_rows + 1
    grid = [np.arange(*grid) for grid in grid_d]  # List with len(grid_d) elements, each is a np.array
    for i in range(0, num_obs * 2, 2):
        if (i%5000 == 0):
            print('row number: {}., out of {}'.format(i, num_obs*2))
        x = np.random.uniform(5, 15, size=(1, seq_len * 2))
        if (p_order > 0) & (np.random.binomial(1, p_order) == 0):
            x = np.sort(x)
            ordered = False  # Safety against ordering
        if ordered:
            x = np.sort(x)
        idx = EmbderMap(len(grid_d), grid)  # To get intuition of what this does run the script in main below
        idx.map_value_to_grid(x[0])
        # if inp_d > 1:
        #     z = np.random.uniform(30, 65, size=(1, seq_len * 2))
        #     idx.map_value_to_grid(z)

        if kernel == 'rbf':
            k = RBF()
            gp = GaussianProcessRegressor(kernel=k)
            if bias == 'const':
                # sample two values from ~N(0, 2) repeat them to be in the same size of sequence length and then permute
                e = np.random.permutation(np.tile(np.random.normal(0, 2, 2), seq_len)).reshape(-1, 1)
                idd = (e == np.unique(e)[0])
                y = gp.sample_y(x.reshape(-1, 1)) + e
            elif bias == 'rbf':
                e = np.random.choice([0, 1], seq_len * 2)
                idd = (e == np.unique(e)[0])
                k1 = RBF(0.4)
                gp1 = GaussianProcessRegressor(kernel=k1)
                y = gp.sample_y(x.reshape(-1, 1)).reshape(-1) + gp1.sample_y(x.reshape(-1, 1)).reshape(-1) * e.reshape(-1)
            else:
                pass
        if noise:
            y = y + np.random.normal(0, 0.01, size=seq_len *2

        idx.map_value_to_grid(np.array(y))
        em_y_idx[int(i / 2), :] = idx.idxs[1]

        df[i, :x.shape[1]] = x
        df[i + 1, :x.shape[1]] = y.reshape(1, -1)
        for j in range(inp_d):
            em_idx[j][int(i / 2), :] = idx.idxs[j]
        em_idx[-1][int(i / 2), idd.reshape(-1)] = 1

    df_tr = df[:tr_rows, :]
    df_te = df[tr_rows:, :]
    em_y.append(em_y_idx[:int(tr_rows / 2), :])
    em_y.append(em_y_idx[int(tr_rows / 2):, :])
    for i in range(inp_d + 1):
        em.append(em_idx[i][:int(tr_rows / 2), :])
        em.append(em_idx[i][int(tr_rows / 2):, :])

    # get all even rows
    x_tr = df_tr[::2, :]
    x_te = df_te[::2, :]
    # get all odd rows
    y_tr = df_tr[1::2, :]
    y_te = df_te[1::2, :]

    return x_tr, x_te, y_tr, y_te, df_tr, df_te, em, em_y



def main():
    # Get intuition about EmbederMap class
    # a = EmbderMap(2, [np.arange(5, 15, 0.1), np.arange(-5, 5, 0.05)])
    # a.map_value_to_grid(np.array([5, 5.5, 5.05, 7.2, 4, 3, 7, 14.9,  15, 15.5]).reshape(1, -1))
    # a.map_value_to_grid(np.array(4).reshape(1, -1))
    # print(a.idxs)
    #
    # b = EmbderMap(2, [np.arange(5, 15, 0.1), np.arange(-5, 5, 0.05)])
    # b.map_value_to_grid(np.array([5, 5.5, 5.05, 7.2, 4, 3, 7, 14.9,  15, 15.5]).reshape(1, -1))
    # b.map_value_to_grid(np.array(4).reshape(1, -1))
    # print(b.idxs)

    # Example for data generation
    x_tr, x_te, y_tr, y_te, df_tr, df_te, em, em_y = data_gen2d(5, 0.8, 3, grid_d=[[1, 15.1, 0.05], [-6, 6, 0.05]], inp_d=1, noise=True)
    # print((em))
    print('y: ', y_tr)
    print('em_y: ', em_y)



if __name__ == '__main__':
    main()
