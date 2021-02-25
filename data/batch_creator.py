import numpy as np


def pick_diff_cols_from_each_row(arr, cols):
    """
    An example would be when we have an array with 40000 rows and 59 columns
    and we want to pick randomly from the first 5000 rows a different set of columns
    cols: array([[24, 16,  7, 11,  6, 46],
                [29, 12, 18, 29, 17, 54],
                [34,  9, 14, 36, 34, 52],
    rows would then be array([[   0,    0,    0,    0,    0,    0],
                            [   1,    1,    1,    1,    1,    1],
                            [   2,    2,    2,    2,    2,    2],
    i.e. pick row 0 columns 24, 26, 7, 11, 6 , 46
    ....
    Returns:

    """
    n = arr.shape[0]
    m = cols.shape[1]
    rows = np.repeat(np.arange(0, n, 1), m).reshape(n, m)
    arr = arr[rows, cols]
    return arr


def rearange_tr_2d(x, y, em, em_2, context_p=50, s=1):
    """

    :param x:
    :param y:
    :param em:
    :param em_2:
    :param context_p:
    :param s:
    :return:
    """
    sorted_idx = np.argsort(x, 1)
    x = pick_diff_cols_from_each_row(x, sorted_idx)
    y = pick_diff_cols_from_each_row(y, sorted_idx)
    em = pick_diff_cols_from_each_row(em, sorted_idx)
    em_2 = pick_diff_cols_from_each_row(em_2, sorted_idx)
    c = []
    xx, yy, eem, eem2 = [], [], [], []

    for row in range(x.shape[0]):
        em_2_pre = em_2[row, :context_p * 2].reshape(-1)
        em_2_pos = em_2[row, context_p * 2:]
        cond = [np.where([em_2_pre == s])[1], np.where([em_2_pos == s])[1], np.where(~(em_2_pre == s)),
                np.where(~(em_2_pos == s))]
        y_pre = y[row, :context_p * 2].reshape(-1)
        y_post = y[row, context_p * 2:].reshape(-1)
        y_infer = np.concatenate((y_pre, y_post[cond[1]])).reshape(-1)
        c_p = len(y_infer)
        c.append(c_p)
        y_infer = np.concatenate((y_infer, y_post[cond[3]]))
        yy.append(y_infer)
        em_pre = em[row, :context_p * 2].reshape(-1)
        em_post = em[row, context_p * 2:].reshape(-1)
        em2_pre = em_2[row, :context_p * 2].reshape(-1)
        em2_post = em_2[row, context_p * 2:].reshape(-1)
        em_infer = np.concatenate((em_pre, em_post[cond[1]]))
        em_infer = np.concatenate((em_infer, em_post[cond[3]]))
        eem.append(em_infer)
        em2_infer = np.concatenate((em2_pre, em2_post[cond[1]]))
        em2_infer = np.concatenate((em2_infer, em2_post[cond[3]]))
        eem2.append(em2_infer)
        x_pre = x[row, :context_p * 2].reshape(-1); x_post = x[row, context_p * 2:].reshape(-1)
        xx_0 = np.concatenate((x_pre, x_post[cond[1]]))
        xx_0 = np.concatenate((xx_0, x_post[cond[3]]))
        xx.append(xx_0)

    return np.array(xx), np.array(yy), np.array(eem), np.array(eem2), c

def create_batch_2d_b(em_x, x, y, em_2, batch_s=128, context_p=50):
    """

    :param em_x:
    :param x:
    :param y:
    :param em_2:
    :param batch_s:
    :param context_p:
    :return:
    """
    b_data = []
    shape = y.shape[0]
    cols = y.shape[1]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    x = x[batch_idx]
    y = y[batch_idx]
    em_x = em_x[batch_idx]
    em_2 = em_2[batch_idx]
    c = context_p
    p = np.random.random()
    if p <= 0.25:
        x, y, em_x, em_2, c = rearange_tr_2d(x, y, em_x, em_2, context_p)
    elif p <= 0.5:
        x, y, em_x, em_2, c = rearange_tr_2d(x, y, em_x, em_2, context_p, s=0)
    elif p <= 0.75:
        permute_idx = np.random.permutation(np.arange(cols))
        x = x[:, permute_idx]
        y = y[:, permute_idx]
        em_x = em_x[:, permute_idx]
        em_2 = em_2[:, permute_idx]
    else:
        pass

    b_data.append(y)
    b_data.append(x)
    b_data.append(em_x)
    b_data.append(em_2)

    return b_data, c


def create_batch_2d(em_x, x, y, em_2, batch_s=128, context_p=50):
    """

    :param em_x:
    :param x:
    :param y:
    :param em_2:
    :param batch_s:
    :param context_p:
    :return:
    """
    b_data = []
    shape = y.shape[0]
    cols = y.shape[1]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    x = x[batch_idx]
    y = y[batch_idx]
    em_x = em_x[batch_idx]
    em_2 = em_2[batch_idx]
    c = context_p
    p = np.random.random()
    if p <= 0.5:
        x, y, em_x, em_2, c = rearange_tr_2d(x, y, em_x, em_2, context_p)
    else:
        x, y, em_x, em_2, c = rearange_tr_2d(x, y, em_x, em_2, context_p, s=0)

    b_data.append(y)
    b_data.append(x)
    b_data.append(em_x)
    b_data.append(em_2)
    return b_data, c


def create_batch(em_x, x, y, batch_s=128, chnge_context=True, d=False, em_2=None, time=False, context_p=50):
    """

    :param em_x:
    :param x:
    :param y:
    :param batch_s:
    :param chnge_context:
    :param d:
    :param em_2:
    :param time:
    :param context_p:
    :return:
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
        permute_idx = np.random.permutation(np.arange(cols))
        b_data.append(y[batch_idx][:, permute_idx])
        b_data.append(x[batch_idx][:, permute_idx])
        b_data.append(em_x[batch_idx][:, permute_idx])
        if d:
            b_data.append(em_2[batch_idx][:, permute_idx])
    return b_data


def fake_batch(x, y, batch_s=1):
    """

    :param x:
    :param y:
    :param batch_s:
    :return:
    """
    shape = y.shape[0]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    batch_y_tr = np.tile(y[batch_idx], 2).reshape(2, -1)
    batch_x_tr = np.tile(x[batch_idx], 2).reshape(2, -1)
    return batch_x_tr, batch_y_tr, batch_idx
