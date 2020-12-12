from helpers import masks 
import tensorflow as tf


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
    combined_mask_pos = masks.create_masks(pos)
    combined_mask_tar = masks.create_masks(tar)
    if mh:
        pred, pred_log_sig = model(pos, tar, False, combined_mask_pos, combined_mask_tar, batch_size = 1)
        sample_y = np.random.normal(pred, np.exp(pred_log_sig))
    else:
        pred, pred_log_sig = model(pos, tar, False, combined_mask_pos, combined_mask_tar)
        sample_y = np.random.normal(pred, np.exp(pred_log_sig))
        # pred_log_sig = None
    return pred, pred_log_sig, sample_y 



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
        # pred_log_sig = None


    tar = tf.concat((tar, sample_y), axis = 1)
    if num_steps > 1:
        model, pos, tar = inference(model, pos, tar, num_steps - 1)
    
    return model, pos, tar
    