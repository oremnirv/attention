
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
