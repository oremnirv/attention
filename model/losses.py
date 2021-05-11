###########################
# Author: Omer Nivron
###########################
import tensorflow as tf


def loss_function(real, pred, pred_log_sig=None, epsilon=0.001):
    """
    Masked MSE. Since the target sequences are padded,
    it is important to apply a padding mask when calculating the loss.
    ----------------
    Parameters:
    real (tf.tensor float64): shape batch_size X max_seq_len. True values of sequences.
    pred (tf.tensor float64): shape batch_size X max_seq_len. Predictions from GPT network.
    pred_log_sig
    epsilon
    ----------------
    Returns:
    loss value (tf.float64), mean squared error, masking
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # tf.print(mask)
    mse = tf.math.square(tf.math.subtract(real, pred))
    # tf.print(mse)
    loss_ = 1 / 2 * (tf.math.divide(mse, tf.math.square(tf.math.exp(pred_log_sig)) + epsilon) + pred_log_sig)
    # print('loss: ', loss_)
    # tf.print(loss_)
    mask = tf.cast(mask, dtype=loss_.dtype)
    mask_b = tf.cast(mask, dtype=mse.dtype)
    loss_ *= mask
    mse *= mask_b
    return tf.reduce_mean(tf.reduce_sum(loss_, 1) / tf.reduce_sum(mask, 1)), tf.reduce_mean(tf.reduce_sum(mse, 1) / tf.reduce_sum(mask_b, 1)), mask


