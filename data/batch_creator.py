
import numpy as np


def create_batch_gp_mim(enc_tr, dec_tr, y_tr, batch_s=128):
    '''
    This is a batch for training attention models. The idea
    is to have x_1:n, y_1:n as input to encoder part, x_n+1
    as input to decoder and try to predict y_n+1. 
    ---------------------
    Parameters:
    enc_tr (np.array): input to encoder. Dimensions: batch_size X timestamps X features  e.g. (if x and y are 1-D, features will be a 2D concatenated vector)
    dec_tr (np.array): input to decoder. i.e. x_n+1
    y_tr (np.array): target. 
    batch_s (int): deafult 128

    ---------------------
    Returns:
    batch_enc_tr (np.array): encoder input associated with randomly chosen indices
    batch_dec_tr (np.array): decoder input associated with randomly chosen indices
    batch_y_tr (np.array): target associated with randomly chosen indices
    batch_idx_tr (np.array): randomly chosen indices shape (batch_s)

    '''
    shape = enc_tr.shape[0]
    timestamps = enc_tr.shape[1]
    feat = enc_tr.shape[2]
    batch_idx_tr = np.random.choice(list(range(shape)), batch_s)
    batch_enc_tr = (enc_tr[batch_idx_tr, :, :].reshape(
        batch_s, timestamps, feat))
    batch_dec_tr = dec_tr[batch_idx_tr]
    batch_y_tr = (y_tr[batch_idx_tr].reshape(-1, 1))
    return batch_enc_tr, batch_dec_tr, batch_y_tr, batch_idx_tr


def create_batch_gp_mim_2(pos, tar, batch_s=128):
    '''
    Get a batch of positions, targets and position mask from data generated 
    by data_generator_for_gp_mimick_gpt function and from position_mask function 
    -------------------------
    Parameters:
    pos (2D np array): 1st/2nd output from data_generator_for_gp_mimick_gpt function 
    tar (2D np array): 3rd/4th output from data_generator_for_gp_mimick_gpt function  
    pos_mask (4D np.array): output from position_mask function 
    batch_s (int): deafult 128
    -------------------------
    Returns:
    batch_pos_tr (2D np array)
    batch_tar_tr (2D np array)
    batch_pos_mask (4D np array)
    batch_idx_tr (1D np array): indices (=row numbers) chosen for current batch

    '''
    shape = tar.shape[0]
    batch_idx_tr = np.random.choice(list(range(shape)), batch_s)
    batch_tar_tr = tar[batch_idx_tr]
    batch_pos_tr = pos[batch_idx_tr]
    return batch_pos_tr, batch_tar_tr, batch_idx_tr


def create_batch_foxes(token_pos, time_pos, tar, batch_s=128):
    '''


    '''
    shape = tar.shape[0]
    batch_idx_tr = np.random.choice(list(range(shape)), batch_s)
    batch_tar_tr = tar[batch_idx_tr, :]
    batch_tok_pos_tr = token_pos[batch_idx_tr, :]
    batch_tim_pos_tr = time_pos[batch_idx_tr, :]
    return batch_tok_pos_tr, batch_tim_pos_tr, batch_tar_tr, batch_idx_tr


def create_batch_river(token_pos, time_pos1, time_pos2, x, att, tar, batch_s=128):
    '''
    '''
    shape = tar.shape[0]
    batch_idx_tr = np.random.choice(list(range(shape)), batch_s)
    batch_tar_tr = tar[batch_idx_tr, :]
    batch_tok_pos_tr = token_pos[batch_idx_tr, :]
    batch_tim_pos_tr = time_pos1[batch_idx_tr, :]
    batch_tim_pos_tr2 = time_pos2[batch_idx_tr, :]
    xx = x[batch_idx_tr, :]
    batch_pos_tr = np.zeros((batch_s, att.shape[1] - 1, tar.shape[1]))
    for i in range(xx.shape[0]):
        batch_pos_tr[i, :, :] = np.concatenate(((np.repeat(np.array(att.iloc[np.where
                                                                ([att['gauge_id'] == xx[i, 0]])[1][0], 1:]).reshape(1, -1), 25)).reshape(-1, 25), (np.repeat(np.array(att.iloc[np.where
                                                                                                                                                                                                            ([att['gauge_id'] == xx[i, -1]])[1][0], 1:]).reshape(1, -1), 25)).reshape(-1, 25)), axis=1)


    return batch_tok_pos_tr, batch_tim_pos_tr, batch_tim_pos_tr2, batch_pos_tr, batch_tar_tr, batch_idx_tr
