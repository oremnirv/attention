import numpy as np
from helpers import plotter


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
    (np.array) sorted for each row according to cols.
    In order to get more intuition run the following example:
    arr  = np.random.choice(np.arange(1, 10), replace = True, size = (5, 10))
    cols =  sorted_idx_samples = pd.DataFrame(arr).apply(lambda x: np.argsort(x), axis=1)
    pick_diff_cols_from_each_row(arr, cols)

    """
    n = arr.shape[0]
    m = cols.shape[1]
    rows = np.repeat(np.arange(0, n, 1), m).reshape(n, m)
    arr = arr[rows, cols]
    return arr


def rearange_tr_2d(x, y, em, em_2, context_p=50, s=1):
    """
    This function is part of the create_batch_2d (_b) functions.
    Its purpose is
    1. to sort each row from the selected batch
    2.  construct the following:
        a. Pick context points * 2 from sorted array
        b. Attach what was not picked in (a) from sequnce #1 to the end of (a) --> context points are now declared
        as the values in the sequence after stage (b)
        c. Attach what was not picked in (a) from sequence #0 to the end of (b)

    :param x: (np.array)
    :param y: (np.array)
    :param em: (np.array) the mapping of x-vals to indices
    :param em_2: (np.array) the pair member each value in x came from (0/1)
    :param context_p: (int) number of context points to consider
    :param s: if s=1, it means we are trying to predict values in series 0
    :return:
    Four rearanged arrays of x, y, em, and em_2 and a list c representing the number
    of context points in each row. This number will differ since stage (2a) does not guarantee
    we pick the same amount of values from sequence #0 at each row.
    In order to check if you understand it, run the example:
    a. load the data on UNN notebook
    b. Run:
        x = data[1][:2, :30]
        y = data[-3][:2, :30]
        em = data[4][:2, :30]
        em_2 = data[-1][:2, :30]
        batch_creator.rearange_tr_2d(x, y, em, em_2, 2)
        *** note that this is also a chance to peek at what is x, y, em, and em_2
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
        cond = [np.where(em_2_pre == s), np.where(em_2_pos == s), np.where(~(em_2_pre == s)),
                np.where(~(em_2_pos == s))]
        y_pre = y[row, :context_p * 2].reshape(-1)
        y_post = y[row, context_p * 2:].reshape(-1)
        y_infer = np.concatenate((y_pre, y_post[cond[1]])).reshape(-1)
        c_p = len(y_infer)
        y_infer = np.concatenate((y_infer, y_post[cond[3]]))
        em_infer = plotter.concat_context_to_infer(em[row], cond, context_p * 2)
        em2_infer = plotter.concat_context_to_infer(em_2[row], cond, context_p * 2)
        xx_0 = plotter.concat_context_to_infer(x[row], cond, context_p * 2)
        xx.append(xx_0);  eem2.append(em2_infer); c.append(c_p); yy.append(y_infer); eem.append(em_infer)

    return np.array(xx), np.array(yy), np.array(eem), np.array(eem2), c


def create_batch_2d_b(em_x, x, y, em_2, batch_s=128, context_p=50):
    """
    This function creates four scenarios for training for pairs of sequences and chooses each batch creation one of:
    1. (p<=0.25) Give the full sequence of values from series 1 and few values at the begining
    from series #0 and try to predict the continuation of series #0
    2. (p<=0.5) The same as scenario (1), but switching the roles of sequence #1 with
    the role of sequence #0
    3. For all members of the batch permute the columns in the same way (e.g. [3, 5, 4, 1, 2, 0])
    keep context points = context_p and arrange x, y, em_x, em_2 accordingly
    4. Just pick batch_s rows from the original data

    :param x: (np.array)
    :param y: (np.array)
    :param em_x: (np.array) the mapping of x-vals to indices
    :param em_2: (np.array) the pair member each value in x came from (0/1)
    :param batch_s: (int) how many rows to pick for each batch
    :param context_p: (int) number of context points to consider
    :return:
    A list (b_data) with four elements (each of which is np.array) and either a list of ints (c) if one of the
    first two scenarios were chosen or an int if scenarios 3/4 were chosen.
    In order to check if you understand it, run the example:
    a. uncomment the print('p: ', p) statement below
    a. load the data on UNN notebook
    b. Run:
        x = data[1][:2, :30]
        y = data[-3][:2, :30]
        em = data[4][:2, :30]
        em_2 = data[-1][:2, :30]
        batch_creator.create_batch_2d_b(em, x, y, em_2, batch_s=1, context_p=2)
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
    print('p: ', p)
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
    This function creates two scenarios for training for pairs of sequences:
    1. (p<=0.5) Give the full sequence of values from series 1 and few values at the begining
    from series #0 and try to predict the continuation of series #0
    2. The same as scenario (1), but switching the roles of sequence #1 with
    the role of sequence #0

    :param x: (np.array)
    :param y: (np.array)
    :param em_x: (np.array) the mapping of x-vals to indices
    :param em_2: (np.array) the pair member each value in x came from (0/1)
    :param batch_s: (int) how many rows to pick for each batch
    :param context_p: (int) number of context points to consider
    :return:
    A list (b_data) with four elements (each of which is np.array) and  a list of ints (c) indicating
    the number of context points per row chosen.
    In order to check if you understand it, run the example:
    a. uncomment the print('p: ', p) statement below
    b. load the data on UNN notebook
    c. Run:
        x = data[1][:2, :30]
        y = data[-3][:2, :30]
        em = data[4][:2, :30]
        em_2 = data[-1][:2, :30]
        batch_creator.create_batch_2d(em, x, y, em_2, batch_s=1, context_p=2)
    """
    print('full infer')
    b_data = []
    shape = y.shape[0]
    batch_idx = np.random.choice(list(range(shape)), batch_s)
    x = x[batch_idx]
    y = y[batch_idx]
    em_x = em_x[batch_idx]
    em_2 = em_2[batch_idx]
    p = np.random.random()
    # print('p: ', p)
    if p <= 0.5:
        x, y, em_x, em_2, c = rearange_tr_2d(x, y, em_x, em_2, context_p)
    else:
        x, y, em_x, em_2, c = rearange_tr_2d(x, y, em_x, em_2, context_p, s=0)

    b_data.append(y)
    b_data.append(x)
    b_data.append(em_x)
    b_data.append(em_2)
    return b_data, c


def create_batch(em_x, x, y, batch_s=128, chnge_context=True, d=False, em_2=None):
    """

    :param x: (np.array)
    :param y: (np.array)
    :param em_x: (np.array) the mapping of x-vals to indices
    :param batch_s: (int)
    :param chnge_context: (bool) if True, shuffle the cols so that you pick different context points in different batches
    :param d: (bool) is it a batch of pairs of sequences?
    :param em_2: (np.array) if d =True else None
    :return:
    A list (b_data) with three/four(if d =TRUE) elements (each of which is np.array)

    In order to check if you understand it, run the example:
    a. load the data on UNN notebook
    b. Run:
        x = data[1][:2, :30]
        y = data[-2][:2, :30]
        em = data[4][:2, :30]
        batch_creator.create_batch(em, x, y, batch_s=1)
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


def batch_regime_2d(x, y, em, em_2, kind='shuffle', context_p=50, batch_s=64):
    """
    This is a wrapper for the three different batching options


    :param x: (np.array)
    :param y: (np.array)
    :param em: (np.array)
    :param em_2: (np.array) consisting of zeros and ones
    :param kind: (str) 'shuffle', 'full infer' for seeing one sequence pair member
    in full and inferring the second sequence given initial context points, or if neither then do half-half
    :param c: (int or list of ints) context points
    :param batch_s: (int)
    :return:
    list of four np.arrays where b_data[0] corresponds to y
    b_data[1] corresponds to x, b_data[2] corresponds to indices of embedded x
    b_data[2] corresponds to 0/1 values specifying the sequence member
    """
    if kind == 'shuffle':
        b_data = create_batch(em, x, y, batch_s=batch_s, d=True, em_2=em_2)
        c = context_p
    elif kind == 'full infer':
        b_data, c = create_batch_2d(em, x, y, em_2, batch_s=batch_s, context_p=context_p)
    else:
        b_data, c = create_batch_2d_b(em, x, y, em_2, batch_s=batch_s, context_p=context_p)

    return b_data, c



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
