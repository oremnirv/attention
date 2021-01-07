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

def r_squared(mse_model, y_true, batch_s = 64):
	'''
	'''
	n = y_true.shape[1]
	y_mean = np.repeat(np.mean(y_true, 1), n).reshape(batch_s, -1)
	return 1 - (mse_model / mse(y_true, y_mean))

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

