
import numpy as np

def create_batch_gp_mim(enc_tr, dec_tr, y_tr, batch_s=128):
    '''


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
