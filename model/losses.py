###########################
# Author: Omer Nivron
###########################
import tensorflow as tf

loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


def loss_function(real, pred, pred_log_sig=None, epsilon=0.001):
    """
    Masked MSE. Since the target sequences are padded,
    it is important to apply a padding mask when calculating the loss.
    ----------------
    Parameters:
    real (tf.tensor float64): shape batch_size X max_seq_len. True values of sequences.
    pred (tf.tensor float64): shape batch_size X max_seq_len. Predictions from GPT network.

    ----------------
    Returns:
    loss value (tf.float64)
    """
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    mse = tf.math.square(tf.math.subtract(real, pred))
    # print(mse)
    # print(pred_log_sig)
    loss_ = 1 / 2 * (tf.math.divide(mse, tf.math.square(tf.math.exp(pred_log_sig)) + epsilon) + pred_log_sig)
    loss_ = tf.math.reduce_mean(loss_, axis=0)
    mse_  = tf.math.reduce_mean(mse, axis=0)
    # print(loss_)
    # mask = tf.cast(mask, dtype=loss_.dtype)
    # loss_ *= mask
    # tf.print(mask)
    # tf.print(tf.reduce_sum(mask))
    # tf.print(tf.reduce_sum(loss_) / tf.reduce_sum(mask))
    return tf.math.reduce_mean(loss_), tf.math.reduce_mean(mse_), mask
