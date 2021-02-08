###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
import matplotlib.pyplot as plt


def dot_product_attention(q, k, v, mask, infer=False, x=None, y=None, n=0, x0=None, y0=None, x1=None, y1=None):
    """

    :param y1:
    :param x1:
    :param y0:
    :param x0:
    :param n:
    :param y:
    :param x:
    :param infer:
    :param q:
    :param k:
    :param v:
    :param mask:
    :return:
    """
    # q = tf.divide(q, tf.norm(q, axis=))
    print(q.shape)
    mask = mask[:, tf.newaxis, :, :]
    matmul_qk = tf.matmul(q, k, transpose_b=True, name='qk')
    matmul_qk = matmul_qk[:, :, 1:, :-1]
    # print(matmul_qk.shape)
    dk = tf.cast(tf.shape(k)[-1], tf.float64)
    nl_qk = tf.cast(tf.nn.relu(matmul_qk / tf.math.sqrt(dk), name='nl_qk'), tf.float64)
    if mask is not None:
        nl_qk += ((tf.cast(mask, tf.float64)) * -1e9)
    att_weights = tf.nn.softmax(nl_qk, axis=-1, name='att_weights')  # batch_size X d_model X seq_len X seq_len
    if infer:
        k_vals, k_ind = tf.math.top_k(att_weights[0, -1, -1, :], k=10, sorted=True, name=None)
        print('k_ind: ', k_ind)
        plt.figure(n)
        # print('x0: ', x0)
        plt.plot(x0, y0, c='lightcoral')
        plt.plot(x1, y1, c='black')
        plt.scatter(x[k_ind.numpy()], y[k_ind.numpy()], color='darkorange', s=52, label='attention points')
        plt.scatter(x[n], y[n], s=52, color ='limegreen')
        plt.savefig('/Users/omernivron/Downloads/attention_plots/step_{}'.format(n))
        # print('top k indices: ', k_ind.numpy())

    # if any(k_ind.numpy() < 21):
    #     print(matmul_qk[:, -1, -1, :])att
    # if all(k_ind.numpy() < 21):
    #     print("PROBLEM ###########")
    #     print(matmul_qk[:, -1, -1, :])
    # print('top k vals: ', k_vals.numpy())
    # print(att_weights)
    # Notice that for all the rows where 
    # everything is 0, the masking will turn everything to -inf
    # and the output from the softmax would be 1/num_cols 
    #  (try a = tf.constant([-1e9, -1e9, -1e9]), tf.nn.softmax(a))
    # So we can expect an output from these rows which we want to ignore
    # this will be enforced in the masking of the loss function

    out_tar = tf.matmul(att_weights, tf.cast(v, tf.float64))
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
        self.wq2 = tf.keras.layers.Dense(d_model, name='wq2')
        self.wq3 = tf.keras.layers.Dense(d_model, name='wq3')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wk2 = tf.keras.layers.Dense(d_model, name='wk2')
        self.wk3 = tf.keras.layers.Dense(d_model, name='wk3')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, e, mask, infer=False, x=None, y=None, n=0, x0=None, y0=None, x1=None, y1=None):
        batch_size = tf.shape(q)[0]
        # q = self.wq3(tf.nn.leaky_relu(self.wq(q) + self.wq2(e)))  # (batch_size, seq_len, d_model)
        # k = self.wk3(tf.nn.leaky_relu(self.wk(k) + self.wk2(e)))
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_att, att_weights, _ = dot_product_attention(
            q, k, v, mask, infer=infer, x=x, y=y, n=n, x0=x0, y0=y0, x1=x1, y1=y1)

        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_att = tf.reshape(scaled_att,
                                (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_att)  # (batch_size, seq_len_q, d_model)

        return output, att_weights
