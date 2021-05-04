###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def dot_product_attention(q, k, v, mask, infer=False, x=None, y=None, n=0, x0=None, y0=None, x1=None, y1=None):
    """
    This

    For more info see: https://www.tensorflow.org/tutorials/text/transformer
    :param q: (tf.tensor) the embedding of x-vals
    :param k: (tf.tensor) the embedding of x-vals
    :param v: (tf.tensor) the embedding of y-vals (or just the repetition of y-vals)
    :param mask: (tf.tnsor) shape is (seq_len, seq_len)
    to see an example run: from helpers import masks; masks.create_masks(data[1][1, :].reshape(1, -1))
    :param infer: (bool) if TRUE plot attention given to each prediction
    :param y1: (tf.tensor) the y-vals associated with pair member #1
    :param x1: (tf.tensor) the x-vals associated with pair member #1
    :param y0: (tf.tensor) the y-vals associated with pair member #0
    :param x0: (tf.tensor) the x-vals associated with pair member #0
    :param n: (int) just used to init figures in inference. Otherwise all will be plotted on one plot.
    :param y: (tf.tensor) all y-vals
    :param x: (tf.tensor) all x-vals

    :return:
    """
    mask = tf.repeat(mask[:, tf.newaxis, :, :], 528, axis=1)

    print('mask: ', mask)
    print('q: ', q)
    matmul_qk = tf.transpose(
        tf.squeeze(tf.tensordot(tf.squeeze(q)[tf.newaxis, :, :, :], tf.transpose(tf.squeeze(k)[tf.newaxis, :, :, :], perm=[0, 1, 3, 2]), axes=[[3], [2]])),
                 perm=[0, 2, 1, 3])
    print('matmul_qk: ', matmul_qk)

    # Notice that matmul_qk will produce (batch size, num heads, seq_len + 1, seq_len +1)
    # tensor. However, we are not interested in the first row since it tells us about the dot product of
    # x1 with xi for all i. But our first prediction is for y2 associated with x2, i.e., the second row.
    # In the same way we are not interested in the last column, since we want to know the dot product
    # only of x_i with x_j for i<j.
    matmul_qk = matmul_qk[ :, :, 1:, :-1]  # (batch size, num heads, seq_len, seq_len)
    dk = tf.cast(tf.shape(k)[-1], tf.float64)
    nl_qk = tf.cast((matmul_qk / tf.math.sqrt(dk)), tf.float64)
    print('nl_qk', nl_qk)
    # Why do we divide by sqrt(dk)? For example, consider that Q and K have a mean of 0 and variance of 1.
    # Their matrix multiplication will have a mean of 0 and variance of dk.
    # So the square root of dk is used for scaling so you get a consistent variance regardless of the value of dk
    # each entry is a some random distribution with 0 mean and 1 variance
    if mask is not None:
        # This step is to make sure that values that are not allowed to be "seen" at a certain step
        # will receive a huge negative value that will translate to 0 weighting after softmax
        nl_qk += ((tf.cast(mask, tf.float64)) * -1e9)  # (batch size, num heads, seq_len, seq_len)

    print('nl_qk 2: ', nl_qk)
    att_weights = tf.reshape(tf.nn.softmax(tf.reshape(nl_qk, [409, -1]), axis=-1, name='att_weights'), [528, 528, 409, -1])
    print('att_weights: ', att_weights)
    print('v: ', v)
    out_tar = tf.tensordot(att_weights, tf.cast(tf.squeeze(v), tf.float64), axes=[[1, 3], [0, 1]])
    print(out_tar)
    return out_tar, att_weights, matmul_qk



class MultiHeadAttention2D(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, 528, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 3, 1, 2, 4])

    def call(self, v, k, q, mask, infer=False, x=None, y=None, n=0, x0=None, y0=None, x1=None, y1=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        scaled_att, att_weights, _ = dot_product_attention(
            q, k, v, mask, infer=infer, x=x, y=y, n=n, x0=x0, y0=y0, x1=x1, y1=y1)
        output = self.dense(scaled_att)  # (batch_size, seq_len_q, d_model)
        return output, att_weights
