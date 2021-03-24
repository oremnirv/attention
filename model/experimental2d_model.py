###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
from tensorflow.keras import regularizers
from model import dot_prod_attention


class Decoder(tf.keras.Model):
    def __init__(self, e, l1=256, l2=128, l3=32, num_heads=1, input_vocab_size=2000):
        super(Decoder, self).__init__()
        self.e = e
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e, name='embedding')
        self.embedding_k = tf.keras.layers.Embedding(2, e, name='embedding_k')

        self.mha = dot_prod_attention.MultiHeadAttention2D(e, num_heads)
        self.A1 = tf.keras.layers.Dense(l1, name='A1')
        self.A2 = tf.keras.layers.Dense(l1, name='A2')
        self.A3 = tf.keras.layers.Dense(l1, name='A3')
        self.A4 = tf.keras.layers.Dense(l2, name='A4')
        self.A5 = tf.keras.layers.Dense(l3, name='A5')
        self.A6 = tf.keras.layers.Dense(2, name='A6')

    def call(self, x, x_2, y, training, x_mask, infer=False, ix=None, iy=None, n=0, x0=None, y0=None, x1=None, y1=None):
        """

        :param x: (np.array of int) of indices associated with a range of continuous x-vals
        :param x_2: (np.array) of zeros and ones, identifying from which sequence member (out of a pair) each
        data point came from.
        :param y: (np.array float) target variables.
        :param training: (bool) must be TRUE when training and False otherwise
        :param x_mask: (np.array) of zeros where a prediction is wanted and 1 otherwise
        :return:
        2D (tf.tensor) with first dimension being the mean and second the log sigma
        """
        y = y[:, :, tf.newaxis]  # (batch_size, seq_len, 1)
        x = self.embedding(x)  # (batch_size, seq_len + 1, e)
        x_2 = self.embedding_k(x_2)  # (batch_size, seq_len + 1, e)
        # y = self.embedding(y)
        y_attn, _, v = self.mha(y, x, x, x_mask, infer=infer, x=ix, y=iy, n=n, x0=x0, y0=y0, x1=x1, y1=y1)  # (batch_size, seq_len, e)
        out1 = self.layernorm1(y_attn + v)
        out1 = tf.nn.leaky_relu(out1)
        current_position = x[:, 1:, :]  # (batch_size, seq_len, e)
        current_series = x_2[:, 1:, :]  # (batch_size, seq_len, e)
        L = self.A1(out1) + self.A2(current_position) + self.A3(current_series)  # (batch_size, seq_len, l1)
        L = tf.nn.leaky_relu(L)
        L = tf.nn.leaky_relu(self.A4(L))  # (batch_size, seq_len, l2)
        L = tf.nn.leaky_relu(self.A5(L))  # (batch_size, seq_len, l3)
        L = self.A6(L)  # (batch_size, seq_len, 2)
        return tf.squeeze(L)
