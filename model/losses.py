import tensorflow as tf

loss_object = tf.keras.losses.MeanSquaredError()

def loss_function(real, pred, pred_sig):
    '''
    Masked MSE. Since the target sequences are padded, 
    it is important to apply a padding mask when calculating the loss.
    ----------------
    Parameters:
    real (tf.tensor float64): shape batch_size X max_seq_len. True values of sequences.
    pred (tf.tensor float64): shape batch_size X max_seq_len. Predictions from GPT network. 
    
    ----------------
    Returns: 
    loss value (tf.float64)
    '''
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.math.divide(loss_object(real, pred), pred_sig) - tf.math.log(pred_sig)
    
#     print('loss_ :', loss_)
#     shape= (128X58)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)