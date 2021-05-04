import numpy as np


def r_sq_2d(y, pred_te, em_2, context_p):
    """
    Measures the R^2 metric for pairs of sequences
    :param y:
    :param pred_te: (np.array)
    :param em_2: (np.array) label for each data point in y: 0/1
    :param context_p: (int) number of context points that were considered
    :return:
    """
    n = y.shape[0]
    r_sq0 = []
    r_sq1 = []
    for row in range(n):
        idx0 = np.where(em_2[row, context_p:] == 0)
        idx1 = np.where(em_2[row, context_p:] == 1)
        y1 = y[row, context_p:]
        pred1 = pred_te[row, context_p:]
        mse_ratio0 = np.sum((y1[idx0] - pred1[idx0]) ** 2) / np.sum((y1[idx0] - np.mean(y1[idx0])) ** 2)
        mse_ratio1 = np.sum((y1[idx1] - pred1[idx1]) ** 2) / np.sum((y1[idx1] - np.mean(y1[idx1])) ** 2)
        r_sq0.append(1 - mse_ratio0)
        r_sq1.append(1 - mse_ratio1)
    return np.mean(r_sq0), np.mean(r_sq1)


def mse(y_true, y_pred):
    """
    Mean squared error for multiple sequences, i.e. calculate MSE for each row and then average these MSEs
    :param y_true:
    :param y_pred:
    :return: (float)
    """

    return np.mean((y_true - y_pred) ** 2, 1)

def r_squared(pred, y_true, batch_s=64, c=280):
    """
    Measures the R^2 metric
    :param mse_model: (float) mean squared error derived by comparing predictions with y_true
    :param y_true: (np.array) the true values of y
    :param batch_s: (int) size of batch
    :return: (float)
    """
    n = y_true.shape[1]
    y_mean = np.repeat(np.mean(y_true, 1), n).reshape(batch_s, -1)
    mse_model = mse(y_true[:, c:], pred)
    return 1 - (mse_model / mse(y_true[:, c:], y_mean[:, c:]))