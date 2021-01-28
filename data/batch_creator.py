import numpy as np


def create_batch(em_x, x, y, batch_s=128, chnge_context=True, d=False, em_2=None, time=False, context_p=50):
    """
    Get a batch of xitions, ygets and xition mask from data generated
    by data_generator_for_gp_mimick_gpt function and from xition_mask function
    -------------------------
    Parameters:
    x (2D np array): 1st/2nd output from data_generator_for_gp_mimick_gpt function
    y (2D np array): 3rd/4th output from data_generator_for_gp_mimick_gpt function
    batch_s (int): deafult 128
    -------------------------
    Returns:
    batch_x_tr (2D np array)
    batch_y_tr (2D np array)
    batch_idx (1D np array): indices (=row numbers) chosen for current batch

    """
    b_data = []
    shape = y.shape[0]
    cols = y.shape[1]
    batch_idx = np.random.choice(list(range(shape)), batch_s)

    if not chnge_context:
        b_data.append(y[batch_idx])
        b_data.append(x[batch_idx])
        b_data.append(em_x[batch_idx])
        if d:
            b_data.append(em_2[batch_idx])
    else:

        if time:
            permute_idx = np.concatenate(
                (np.sort(np.random.choice(range(120), context_p, replace=False)), range(120, cols, 1)))

        else:
            permute_idx = np.random.permutation(np.arange(cols))
        b_data.append(y[batch_idx][:, permute_idx])
        b_data.append(x[batch_idx][:, permute_idx])
        b_data.append(em_x[batch_idx][:, permute_idx])
        if d:
            b_data.append(em_2[batch_idx][:, permute_idx])
    return b_data


def fake_batch(x, y, batch_s=1):
    """
    Get a batch of xitions, ygets and xition mask from data generated
    by data_generator_for_gp_mimick_gpt function and from xition_mask function
    -------------------------
    Parameters:
    x (2D np array): 1st/2nd output from data_generator_for_gp_mimick_gpt function
    y (2D np array): 3rd/4th output from data_generator_for_gp_mimick_gpt function
    batch_s (int): deafult 128
    -------------------------
    Returns:
    batch_x_tr (2D np array)
    batch_y_tr (2D np array)
    batch_idx (1D np array): indices (=row numbers) chosen for current batch

    """
    shape = y.shape[0]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    batch_y_tr = np.tile(y[batch_idx], 2).reshape(2, -1)
    batch_x_tr = np.tile(x[batch_idx], 2).reshape(2, -1)
    return batch_x_tr, batch_y_tr, batch_idx
