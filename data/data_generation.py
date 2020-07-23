###########################
# Author: Omer Nivron
###########################

import numpy as np
from data import gp_priors
from data import gp_kernels 


def dat_generator_for_gp_mimick(num_samples, obs_per_sample, kernel, tr_percent=0.8):
    '''



    '''
    df = np.zeros((num_samples * 2, obs_per_sample))
    for i in range(0, num_samples * 2, 2):
        x = np.random.uniform(-5, 5, size=(1, obs_per_sample))
        k = kernel(x)
        f_prior = gp_priors.generate_priors(k, obs_per_sample, 1)

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

def data_generator_for_gp_mimick_gpt(num_obs, kernel, tr_percent=0.8, seq_len=59, extarpo = True, extarpo_num = 19):
    '''
    Generator for training a GPT inspired netowrk.
    -----------------------
    Parameters:
    num_obs (int): how many observations to generate
    kernel (function of am SKlearn kernel object): e.g. rbf_kernel which comes from gp_kernels file
    tr_percent (float): daefult 0.8
    seq_len (int): daefult 59
    -----------------------
    Returns:
    pad_pos_tr (np array): the first rows * tr_percent from the x generated values 
    pad_pos_te (np array): all rows of x not chosen for training 
    pad_y_fren_tr (np array): the first rows * tr_percent from the f_prior generated values 
    pad_y_fren_te (np array): all rows of f_prior not chosen for training
    df_tr (np array): positions and targets combined (training) 
    df_te (np array): positions and targets combined (testing) 
    '''
    df = np.zeros((num_obs * 2, seq_len))
    rows = df.shape[0]
    cols = df.shape[1]
    tr_rows = int(tr_percent * rows)
    tr_rows = tr_rows if tr_rows % 2 == 0 else tr_rows + 1
    for i in range(0, num_obs * 2, 2):
        if ((i >= tr_rows) & (extarpo)):
            x = np.concatenate((np.random.uniform(5, 15, size=(1, seq_len - extarpo_num)), np.random.uniform(15.1, 20, size=(1, extarpo_num))), axis = 1)

        else:
            x = np.random.uniform(5, 15, size=(1, seq_len))
        k = kernel(x)
        f_prior = gp_priors.generate_priors(k, seq_len, 1)

        df[i, :x.shape[1]] = x
        df[i + 1, :x.shape[1]] = f_prior


    df_tr = df[:tr_rows, :]
    df_te = df[tr_rows:, :]
    
    # get all even rows
    pad_pos_tr = df_tr[::2, :]
    pad_pos_te = df_te[::2, :]
    # get all odd rows
    pad_y_fren_tr = df_tr[1::2, :]
    pad_y_fren_te = df_te[1::2, :]

    return pad_pos_tr, pad_pos_te, pad_y_fren_tr, pad_y_fren_te, df_tr, df_te

def data_generator_river_flow(df, basins, context_channels2, seq_len, num_seq):
    '''
    selected_basins = pd.read_csv('/Users/omernivron/Downloads/basin_list.txt', header=None)
    df = pd.read_csv('/Users/omernivron/Downloads/daymet_data_seed05.csv')
    context_channels = ['OBS_RUN',
                    'doy_cos','doy_sin',
                    'prcp(mm/day)', 
                    'srad(W/m2)',  
                    'tmax(C)',
                    'tmin(C)', 
                    'vp(Pa)'] 
                    context_channels2 = ['prcp(mm/day)', 
                    'srad(W/m2)',  
                    'tmax(C)',
                    'tmin(C)', 
                    'vp(Pa)'] 
    list_to_drop = ['MNTH', 'DY', 'hru02', 'hru04', 'RAIM', 'TAIR', 'PET', 'ET', 'SWE', 'swe(mm)', 'PRCP', 'seed', 'id_lag', 'HR', 'dayl(s)', 'YR', 'MOD_RUN', 'id', 'DOY', 'DATE']
    df.drop(columns= list_to_drop, inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['OBS_RUN'] >= 0]
    cols = df.columns.to_list()
    cols = [cols[5]] + cols[:5] + cols[6:]
    df = df[cols]
    step_tr = 0
step_te = 0
for bas in df.basin.unique():
    df_temp = (df[df['basin'] == bas]).reset_index()
    att_num = np.where(attributes_numeric['hru08'] == df_temp['hru08'].unique()[0])[0]
    rep_att_num = np.repeat(att_num, 50).reshape(-1, 1)
    for obs in range(200):
        rows = np.random.choice(np.arange(0, df_temp.shape[0], 1), 50)
        if (np.isin(bas, selected_basins)):
            arr_tr[step_tr, :, :] = np.concatenate((df_temp.loc[rows, context_channels], rep_att_num), axis = 1)
            step_tr += 1
        else:
            arr_te[step_te, :, :] = np.concatenate((df_temp.loc[rows, context_channels], rep_att_num), axis = 1)
            step_te += 1
            np.save('/Users/omernivron/Downloads/river_flow_tr', arr_tr)
        np.save('/Users/omernivron/Downloads/river_flow_te', arr_te)
    '''
    arr_pp = np.zeros((5 * num_seq, seq_len))
    for i in range(len(context_channels2)):
        for row in range(0 + num_seq * i, num_seq * (i + 1), 5):
            basin = np.random.choice(basins, 2)
            df_temp = (df[df['basin'] == basin[0]]).reset_index()
            df_riv = (df[df['basin'] == basin[1]]).reset_index()

            idx_i = np.random.choice(np.arange(0, df_temp.shape[0], 1), seq_len / 2)
            idx_riv = np.random.choice(np.arange(0, df_riv.shape[0], 1), seq_len /2)
            
            arr_pp[row, :] = np.concatenate((df_temp.loc[idx_i, [context_channels2[i]]], df_riv.loc[idx_riv, ['OBS_RUN']]), axis = 0).reshape(-1)
            arr_pp[row + 1, :] = np.concatenate((df_temp.loc[idx_i, ['doy_cos']], df_riv.loc[idx_riv, ['doy_cos']]), axis = 0).reshape(-1)
            arr_pp[row + 2, :] = np.concatenate((df_temp.loc[idx_i, ['doy_sin']], df_riv.loc[idx_riv, ['doy_sin']]), axis = 0).reshape(-1)
            arr_pp[row + 3, :] = np.concatenate((np.ones(seq_len / 2) * i, np.ones(seq_len / 2) * 9))
            arr_pp[row + 4, :] = np.concatenate((df_temp.loc[idx_i, ['basin']], df_riv.loc[idx_riv, ['basin']]), axis = 0).reshape(-1)

    return  arr_pp

