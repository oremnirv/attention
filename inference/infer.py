from helpers import masks 


def evaluate(model, pos, tar, pos_mask):
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
    combined_mask_tar = masks.create_masks(tar)
    pred, pred_log_sig = model(pos, tar, False, pos_mask, combined_mask_tar)
    return pred, pred_log_sig 



def inference(model, pos, tar, num_steps = 1):
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
    pos_mask = masks.position_mask(temp_pos)
    
    pred, pred_log_sig = evaluate(model, temp_pos, tar, pos_mask)
    tar = tf.concat((tar, tf.reshape(pred[n - 1], [1, 1])), axis = 1)
    if num_steps > 1:
        pred, pred_log_sig  = inference(pos, tar, num_steps - 1)
    
    return pred, pred_log_sig 
    