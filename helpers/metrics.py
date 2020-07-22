import numpy as np

def KUEE(y_true, μ_te, σ_te):
    '''
    A metric to empirically estimate aleatoric uncertainty (all observations are indpendent)
    ------------------------

    
    '''
    if ((y_true >= μ_te - 2 * σ_te) & (y_true <= μ_te + 2 * σ_te)):
        return np.sum((y_true >= μ_te - 2 * σ_te) & (y_true <= μ_te + 2 * σ_te))
    else: 
        return 0

def r_squared(y_true, y_pred, y_mean):
	'''
	'''
	return 1 - (mse(y_true, y_pred) / mse(y_true, y_mean))

def mse(y_true, y_pred):
	'''

	'''

	return np.sum((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
	'''

	'''

	return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

def mae(y_true, y_pred):
	'''

	'''

	return np.abs(y_true - y_pred)

