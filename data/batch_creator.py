import numpy as np


def create_batch(em_pos, pos, tar, batch_s=128, chnge_context=True, d=False, em_2=None, time = False, context_p = 50):
    """
    Get a batch of positions, targets and position mask from data generated
    by data_generator_for_gp_mimick_gpt function and from position_mask function
    -------------------------
    Parameters:
    pos (2D np array): 1st/2nd output from data_generator_for_gp_mimick_gpt function
    tar (2D np array): 3rd/4th output from data_generator_for_gp_mimick_gpt function
    batch_s (int): deafult 128
    -------------------------
    Returns:
    batch_pos_tr (2D np array)
    batch_tar_tr (2D np array)
    batch_idx (1D np array): indices (=row numbers) chosen for current batch

    """
    b_data = []
    shape = tar.shape[0]
    cols = tar.shape[1]
    batch_idx = np.random.choice(list(range(shape)), batch_s)

    if not chnge_context:
        b_data.append(tar[batch_idx])
        b_data.append(pos[batch_idx])
        b_data.append(em_pos[batch_idx])
        if d:
            b_data.append(em_2[batch_idx])
    else:

        if time:
                permute_idx = np.concatenate(
                (np.sort(np.random.choice(range(120), context_p, replace=False)), range(120, cols, 1)))

        else:
            permute_idx = np.random.permutation(np.arange(cols))
        b_data.append(tar[batch_idx][:, permute_idx])
        b_data.append(pos[batch_idx][:, permute_idx])
        b_data.append(em_pos[batch_idx][:, permute_idx])
        if d:
            b_data.append(em_2[batch_idx][:, permute_idx])
    return b_data


def fake_batch(pos, tar, batch_s=1):
    """
    Get a batch of positions, targets and position mask from data generated
    by data_generator_for_gp_mimick_gpt function and from position_mask function
    -------------------------
    Parameters:
    pos (2D np array): 1st/2nd output from data_generator_for_gp_mimick_gpt function
    tar (2D np array): 3rd/4th output from data_generator_for_gp_mimick_gpt function
    batch_s (int): deafult 128
    -------------------------
    Returns:
    batch_pos_tr (2D np array)
    batch_tar_tr (2D np array)
    batch_idx (1D np array): indices (=row numbers) chosen for current batch

    """
    shape = tar.shape[0]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    batch_tar_tr = np.tile(tar[batch_idx], 2).reshape(2, -1)
    batch_pos_tr = np.tile(pos[batch_idx], 2).reshape(2, -1)
    return batch_pos_tr, batch_tar_tr, batch_idx


def create_batch_foxes(token_pos, time_pos, tar, batch_s=128):
    """


    """
    shape = tar.shape[0]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    batch_tar_tr = tar[batch_idx, :]
    batch_tok_pos_tr = token_pos[batch_idx, :]
    batch_tim_pos_tr = time_pos[batch_idx, :]
    return batch_tok_pos_tr, batch_tim_pos_tr, batch_tar_tr, batch_idx


def create_batch_river(token_pos, time_pos1, time_pos2, x, att, tar, batch_s=128):
    """
    """
    shape = tar.shape[0]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    batch_tar_tr = tar[batch_idx, :]
    batch_tok_pos_tr = token_pos[batch_idx, :]
    batch_tim_pos_tr = time_pos1[batch_idx, :]
    batch_tim_pos_tr2 = time_pos2[batch_idx, :]
    xx = x[batch_idx, :]
    batch_pos_tr = np.zeros((batch_s, att.shape[1] - 1, tar.shape[1]))
    for i in range(xx.shape[0]):
        batch_pos_tr[i, :, :] = np.concatenate(((np.repeat(np.array(att.iloc[np.where
                                                                             ([att['gauge_id'] == xx[i, 0]])[1][0],
                                                                    1:]).reshape(1, -1), 25)).reshape(-1, 25),
                                                (np.repeat(np.array(att.iloc[np.where
                                                                             ([att['gauge_id'] == xx[i, -1]])[1][0],
                                                                    1:]).reshape(1, -1), 25)).reshape(-1, 25)), axis=1)

    return batch_tok_pos_tr, batch_tim_pos_tr, batch_tim_pos_tr2, batch_pos_tr, batch_tar_tr, batch_idx
