###########################
# Author: Omer Nivron
###########################
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF

from data import gp_kernels
from data import gp_priors


class EmbderMap:
    """docstring for embder_map"""

    def __init__(self, num_vars, grids):
        super(EmbderMap, self).__init__()
        self.num_vars = num_vars
        assert num_vars == len(grids)
        self.grid = []
        self.idxs = []
        for ix, val in enumerate(grids):
            self.grid.append(val)

    def map_value_to_grid(self, var):
        n = len(self.idxs)
        g = self.grid[n]
        idx = []
        for i in var[0]:
            if n == 0:

                if i > max(g):
                    idx.append(200 + (len(g) - 1))
                elif i < min(g):
                    idx.append(200 + 0)
                else:
                    idx.append(200 + np.where(i < g)[0][0])
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
    pad_pos_tr (np array): the first rows * tr_percent from the x generated values 
    pad_pos_te (np array): all rows of x not chosen for training 
    y_tr (np array): the first rows * tr_percent from the f_prior generated values
    y_te (np array): all rows of f_prior not chosen for training
    df_tr (np array): positions and targets combined (training) 
    df_te (np array): positions and targets combined (testing) 
    """
    if grid_d is None:
        grid_d = [1, 15.1, 0.1]
    df = np.zeros((num_obs * 2, seq_len))
    em_idx = np.zeros((num_obs, seq_len))
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
        idx.map_value_to_grid(x)

        if (p_order > 0) & (np.random.binomial(1, p_order) == 0):
            x = np.sort(x)
            ordered = False
        if ordered:
            x = np.sort(x)
        if kernel == 'rbf':
            k = gp_kernels.rbf_kernel(x.reshape(-1, 1))
            f_prior = gp_priors.generate_priors(k, seq_len, 1)
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
    # get all even rows
    x_tr = df_tr[::2, :]
    x_te = df_te[::2, :]
    # get all odd rows
    y_tr = df_tr[1::2, :]
    y_te = df_te[1::2, :]

    return x_tr, x_te, y_tr, y_te, df_tr, df_te, em_tr, em_te


def data_gen2d(num_obs, tr_percent=0.8, seq_len=200, bias='const', kernel='rbf',
               grid_d=[[1, 15.1, 0.05], [30, 65.1, 0.05]], noise=False,
               ordered=False, inp_d=1, p_order=0.5):
    df = np.zeros((num_obs * 2, seq_len * 2))
    em = []
    em_idx = [np.zeros((num_obs, seq_len * 2)) for _ in range(inp_d + 1)]
    rows = df.shape[0]
    tr_rows = int(tr_percent * rows)
    tr_rows = tr_rows if tr_rows % 2 == 0 else tr_rows + 1
    grid = [np.arange(*grid) for grid in grid_d]
    for i in range(0, num_obs * 2, 2):
        x = np.random.uniform(5, 15, size=(1, seq_len * 2))
        if (p_order > 0) & (np.random.binomial(1, p_order) == 0):
            x = np.sort(x)
            ordered = False
        if ordered:
            x = np.sort(x)
        idx = EmbderMap(len(grid_d), grid)
        idx.map_value_to_grid(x)
        # if inp_d > 1:
        #     z = np.random.uniform(30, 65, size=(1, seq_len * 2))
        #     idx.map_value_to_grid(z)

        if kernel == 'rbf':
            k = RBF()
            gp = GaussianProcessRegressor(kernel=k)
            if bias == 'const':
                e = np.random.permutation(np.tile(np.random.normal(0, 2, 2), seq_len)).reshape(-1, 1)
                idd = (e == np.unique(e)[0])
                y = gp.sample_y(x.reshape(-1, 1)) + e
            elif bias == 'rbf':
                e = np.random.choice([0, 1], seq_len * 2)
                idd = (e == np.unique(e)[0])
                k1 = RBF(0.4)
                gp1 = GaussianProcessRegressor(kernel=k1)
                y = gp.sample_y(x.reshape(-1, 1)).reshape(-1) + gp1.sample_y(x.reshape(-1, 1)).reshape(-1) * e.reshape(
                    -1)
            else:
                pass
        if noise:
            e = WhiteKernel(.1)
            gp = GaussianProcessRegressor(kernel=e)
            y = y + gp.sample_y(x, seq_len * 2)

        df[i, :x.shape[1]] = x
        df[i + 1, :x.shape[1]] = y.reshape(-1)
        for j in range(inp_d):
            em_idx[j][int(i / 2), :] = idx.idxs[j]
        em_idx[-1][int(i / 2), idd.reshape(-1)] = 1

    df_tr = df[:tr_rows, :]
    df_te = df[tr_rows:, :]
    for i in range(inp_d + 1):
        em.append(em_idx[i][:int(tr_rows / 2), :])
        em.append(em_idx[i][int(tr_rows / 2):, :])

    # get all even rows
    x_tr = df_tr[::2, :]
    x_te = df_te[::2, :]
    # get all odd rows
    y_tr = df_tr[1::2, :]
    y_te = df_te[1::2, :]

    return x_tr, x_te, y_tr, y_te, df_tr, df_te, em


def data_generator_river_flow(df, basins, seq_len, num_seq):
    """
    selected_basins = pd.read_csv('/Users/omernivron/Downloads/basin_list.txt', header=None)
    test_basins = df.basin.unique()[~np.isin(df.basin.unique(), selected_basins)]

    """
    context_channels2 = ['prcp(mm/day)', 'srad(W/m2)',
                         'tmax(C)', 'tmin(C)', 'vp(Pa)']
    arr_pp = np.zeros((5 * num_seq, seq_len))
    m = int(seq_len / 2)
    for i in range(len(context_channels2)):
        for row in range(0 + num_seq * i, num_seq * (i + 1), 5):
            basin = np.random.choice(basins, 2)
            df_temp = (df[df['basin'] == basin[0]]).reset_index()
            df_riv = (df[df['basin'] == basin[1]]).reset_index()
            idx_i = np.random.choice(np.arange(0, df_temp.shape[0], 1), m)
            idx_riv = np.random.choice(np.arange(0, df_riv.shape[0], 1), m)
            arr_pp[row, :] = np.concatenate((df_temp.loc[idx_i, [
                context_channels2[i]]], df_riv.loc[idx_riv, ['OBS_RUN']]), axis=0).reshape(-1)
            arr_pp[row + 1, :] = np.concatenate(
                (df_temp.loc[idx_i, ['doy_cos']], df_riv.loc[idx_riv, ['doy_cos']]), axis=0).reshape(-1)
            arr_pp[row + 2, :] = np.concatenate(
                (df_temp.loc[idx_i, ['doy_sin']], df_riv.loc[idx_riv, ['doy_sin']]), axis=0).reshape(-1)
            arr_pp[row + 3, :] = np.concatenate((np.ones(m) * i, np.ones(m) * 9))
            arr_pp[row + 4, :] = np.concatenate(
                (df_temp.loc[idx_i, ['basin']], df_riv.loc[idx_riv, ['basin']]), axis=0).reshape(-1)

    return arr_pp


def main():
    # a = EmbderMap(2, [np.arange(5, 15, 0.1), np.arange(70, 74, 0.05)])
    # a.map_value_to_grid(np.array([5, 7]).reshape(1, -1))
    # a.map_value_to_grid(np.array([74, 77]).reshape(1, -1))
    # print(a.idxs[0])
    # print(a.idxs[1])
    data_gen2d(10)


if __name__ == '__main__':
    main()
