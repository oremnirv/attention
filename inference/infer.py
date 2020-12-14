from helpers import masks 
import tensorflow as tf
import numpy as np


def evaluate(model, pos, tar, mh = False):
    '''
    Run a forward pass of the network
    ------------------
    Parameters:
    model: trained instace of GPT decoder class
    pos: position tensor with at least len(tar) + 1 values
    tar: targets tensor
    pos_mask: position mask tensor to hide unseen positions from current prediction 
    ------------------
    Returns:
    pred (tf tensor float64): the prediction of the next location in the sequence 
    pred_log_sig (tf tensor float64)
    
    '''
    pos1 = pos[:, :-1]
    current_pos = pos[:, 1:]
    combined_mask_pos = masks.create_masks(pos1)
    combined_mask_tar = masks.create_masks(tar)
    if mh:
        pred = model(pos1, current_pos, tar, False, combined_mask_pos, combined_mask_tar, batch_size = 1)
        sample_y = np.random.normal(pred[ -1, 0], np.exp(pred[ -1, 1]))
    else:
        pred = model(pos1, current_pos, tar, False, combined_mask_pos, combined_mask_tar)
        # print(pred.shape)
        sample_y = np.random.normal(pred[ -1, 0], np.exp(pred[ -1, 1]))

    return pred[:, 0], pred[:, 1], sample_y 



def inference(model, pos, tar, num_steps = 1, mh = False):
    '''
    how many steps to infer -- this could be used both for interpolation and extrapolation 
    ------------------
    Parameters:
    pos (2D np array): (n + num_steps) positions 
    tar (2D np array): n targets 
    num_steps (int): how many inference steps are required
    ------------------
    Returns:
    pred (tf.tensor float64): the predictions for all timestamps up to n + num_steps  
    pred_log_sig
    '''
    n = tar.shape[1]
    temp_pos = pos[:, :(n + 1)]
    if  mh:
        pred, pred_log_sig, sample_y = evaluate(model, temp_pos, tar, mh = True)
    else:
        pred, pred_log_sig, sample_y = evaluate(model, temp_pos, tar)


    tar = tf.concat((tar, tf.reshape(sample_y, [1, 1])), axis = 1)
    if num_steps > 1:
        model, pos, tar = inference(model, pos, tar, num_steps - 1)
    
    return model, pos, tar
    