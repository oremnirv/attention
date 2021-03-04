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
    mse = tf.math.square(tf.math.subtract(real, pred))
    loss_ = 1 / 2 * (tf.math.divide(mse, tf.math.square(tf.math.exp(pred_log_sig)) + epsilon) + pred_log_sig)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask), mse, mask


