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
    mask = mask[:, tf.newaxis, :, :]
    matmul_qk = tf.matmul(q, k, transpose_b=True, name='qk')
    # Notice that matmul_qk will produce (batch size, num heads, seq_len + 1, seq_len +1)
    # tensor. However, we are not interested in the first row since it tells us about the dot product of
    # x1 with xi for all i. But our first prediction is for y2 associated with x2, i.e., the second row.
    # In the same way we are not interested in the last column, since we want to know the dot product
    # only of x_i with x_j for i<j.
    matmul_qk = matmul_qk[:, :, 1:, :-1]  # (batch size, num heads, seq_len, seq_len)
    dk = tf.cast(tf.shape(k)[-1], tf.float64)
    nl_qk = tf.cast(matmul_qk / tf.math.sqrt(dk), tf.float64)
    # Why do we divide by sqrt(dk)? For example, consider that Q and K have a mean of 0 and variance of 1.
    # Their matrix multiplication will have a mean of 0 and variance of dk.
    # So the square root of dk is used for scaling so you get a consistent variance regardless of the value of dk
    # each entry is a some random distribution with 0 mean and 1 variance
    if mask is not None:
        # This step is to make sure that values that are not allowed to be "seen" at a certain step
        # will receive a huge negative value that will translate to 0 weighting after softmax
        nl_qk += ((tf.cast(mask, tf.float64)) * -1e9)  # (batch size, num heads, seq_len, seq_len)

    att_weights = tf.nn.softmax(nl_qk, axis=-1, name='att_weights')  # (batch_size X d_model X seq_len X seq_len)
    if infer:
        # This block has the following intention:
        # a) get the last row of the attention weigths
        # If we are trying to infer the 51st element then the last row
        # should represent the attention weight of <x_51, x_1> ....<x_51, x_50>
        # b) get top (biggest) 5 vals & idxs from the last row
        # c) If num_head = 1 this is exactly what we want to show, else we have multiple heads so:
        # d) take the indices of top 5 values for all heads and return the top 10 repeating values (of indices)
        k_vals, k_ind = tf.math.top_k(att_weights[0, :, -1, :], k=5, sorted=True, name=None)
        k_vals_agg, k_ind_agg = tf.math.top_k(k_ind.numpy().reshape(-1), k=10, sorted=True, name=None)
        print('att: ', att_weights[0, :, -1, :].shape)
        plt.figure(n)
        plt.plot(x0, y0, c='lightcoral')
        plt.plot(x1, y1, c='black')
        print('x: ', x)
        print('x shape: ', x.shape)
        plt.scatter(x[k_vals_agg.numpy()], y[k_vals_agg.numpy()], color='darkorange', s=52, label='attention points')
        plt.scatter(x[n], y[n], s=52, color='limegreen')
        plt.savefig(os.path.expanduser('~/Downloads/attention_plots/step_{}'.format(n)))

    # Notice that for all the rows where 
    # everything is 1, the masking will turn everything to -inf
    # and the output from the softmax would be 1/num_cols 
    # (try a = tf.constant([-1e9, -1e9, -1e9]), tf.nn.softmax(a))
    # So we can expect an output from these rows which we want to ignore
    # this will be enforced in the masking of the loss function

    out_tar = tf.matmul(att_weights, tf.cast(v, tf.float64))  # (batch size, num heads, seq_len, 32)
    return out_tar, att_weights, matmul_qk


# Instead of one single attention head, Q, K, and V are split into multiple heads because
# it allows the model to jointly attend to information at different positions from different representational space


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_inp=1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.q_layers = [tf.keras.layers.Dense(d_model, name='q' + str(z))
                         for z in range(num_inp)]
        self.k_layers = [tf.keras.layers.Dense(d_model, name='k' + str(z))
                         for z in range(num_inp)]
        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        if len(q.shape) > 3:
            for d in range(q.shape[2]):
                if d == 0:
                    qq = self.q_layers[0](q[:, :, d, :])
                    kk = self.k_layers[0](k[:, :, d, :])

                else:
                    qq += self.q_layers[d](q[:, :, d, :])
                    kk += self.k_layers[d](k[:, :, d, :])
        else:
            qq = self.q_layers[0](q)
            kk = self.k_layers[0](k)
        v = self.wv(v)

        q = self.split_heads(qq, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(kk, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_att, att_weights, _ = dot_product_attention(
            q, k, v, mask)
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_att = tf.reshape(scaled_att,
                                (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_att)  # (batch_size, seq_len_q, d_model)

        return output, att_weights


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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

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
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_att = tf.reshape(scaled_att,
                                (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_att)  # (batch_size, seq_len_q, d_model)
        return output, att_weights
