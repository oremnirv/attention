import numpy as np
from gp_priors import *
from gp_kernels import *


def data_generator_for_gp_mimick(num_samples, obs_per_sample, kernel, tr_percent=0.8):
    '''



    '''
    df = np.zeros((num_samples * 2, obs_per_sample))
    for i in range(0, num_samples * 2, 2):
        x = np.random.uniform(-5, 5, size=(1, obs_per_sample))
        k = kernel(x)
        f_prior = generate_priors(k, obs_per_sample, 1)

        df[i, :] = x
        df[i + 1, :] = f_prior

    rows = df.shape[0]
    cols = df.shape[1]
    tr_rows = int(tr_percent * rows)
    tr_rows = tr_rows if tr_rows % 2 == 0 else tr_rows + 1
    df_tr = df[:tr_rows, :]
    df_te = df[tr_rows:, :]

    eng_tr = df_tr[:tr_rows, :(cols - 1)]
    eng_te = df_te[:, :(cols - 1)]
    fren_tr = df_tr[::2, cols - 1]
    fren_te = df_te[::2, cols - 1]
    y_fren_tr = df_tr[1::2, cols - 1]
    y_fren_te = df_te[1::2, cols - 1]

    return eng_tr.T.reshape(-1, cols - 1, 2), eng_te.T.reshape(-1, cols - 1, 2), fren_tr, fren_te, y_fren_tr.reshape(-1, 1), y_fren_te.reshape(-1, 1)




def data_generator_for_gp_mimick_gpt(num_obs, kernel, tr_percent=0.8):
    '''



    '''
    obs_per_sample = np.random.randint(20, 60, size = num_obs)
    df = np.zeros((num_obs * 2, np.max(obs_per_sample)))
    for i in range(0, num_obs * 2, 2):
        x = np.random.uniform(5, 15, size=(1, obs_per_sample[int(i / 2)]))
        k = kernel(x)
        f_prior = generate_priors(k, obs_per_sample[int(i / 2)], 1)

        df[i, :x.shape[1]] = x
        df[i + 1, :x.shape[1]] = f_prior

    rows = df.shape[0]
    cols = df.shape[1]
    tr_rows = int(tr_percent * rows)
    tr_rows = tr_rows if tr_rows % 2 == 0 else tr_rows + 1
    df_tr = df[:tr_rows, :]
    df_te = df[tr_rows:, :]

    pad_pos_tr = df_tr[::2, :]
    pad_pos_te = df_te[::2, :]
    pad_y_fren_tr = df_tr[1::2, :]
    pad_y_fren_te = df_te[1::2, :]

    return pad_pos_tr, pad_pos_te, pad_y_fren_tr, pad_y_fren_te