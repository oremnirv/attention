import numpy as np


def KUEE(y_true, μ_te, σ_te):
    '''
    A metric to empirically estimate aleatoric uncertainty (all observations are indpendent)
    ------------------------


    '''

    y_true = round(y_true, 4)
    lower = μ_te - 2 * σ_te
    upper = μ_te + 2 * σ_te

    cond = ((y_true >= lower) & (y_true <= upper))
    if cond:
        return np.sum(cond)
    else:
        return 0


def r_squared(mse_model, y_true, batch_s=64):
    '''
    '''
    n = y_true.shape[1]
    y_mean = np.repeat(np.mean(y_true, 1), n).reshape(batch_s, -1)
    return 1 - (mse_model / mse(y_true, y_mean))



def r_sq_2d(y, pred_te, em_2, context_p):
    n = y.shape[0]
    r_sq0 =[]; r_sq1 = []
    for row in range(n):
        idx0 = np.where(em_2[row, (context_p):] == 0)
        idx1 = np.where(em_2[row, (context_p):] == 1)
        y1 = y[row, (context_p):]
        pred1 = pred_te[row, (context_p):]
        mse_ratio0 = np.sum((y1[idx0] - pred1[idx0]) ** 2) / np.sum((y1[idx0] - np.mean(y1[idx0])) ** 2)
        mse_ratio1 = np.sum((y1[idx1] - pred1[idx1]) ** 2) / np.sum((y1[idx1] - np.mean(y1[idx1])) ** 2)
        r_sq0.append(1 - mse_ratio0)
        r_sq1.append(1 - mse_ratio1)
    return np.mean(r_sq0), np.mean(r_sq1) 

def mse(y_true, y_pred):
    '''

    '''

    return np.mean(np.mean((y_true - y_pred) ** 2, 1))


def rmse(y_true, y_pred):
    '''

    '''

    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

def mae(y_true, y_pred):
    '''

    '''

    return np.abs(y_true - y_pred)

