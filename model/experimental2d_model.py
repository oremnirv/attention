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
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e)
        self.mha = dot_prod_attention.MultiHeadAttention2D(e, num_heads)
        self.A1 = tf.keras.layers.Dense(l1, name='A1')
        self.A2 = tf.keras.layers.Dense(l1, name='A2')
        self.A3 = tf.keras.layers.Dense(l1, name='A3')
        self.A4 = tf.keras.layers.Dense(l2, name='A4')
        self.A5 = tf.keras.layers.Dense(l3, name='A5')
        self.A6 = tf.keras.layers.Dense(2, name='A6')

    def call(self, x, x_2, y, training, x_mask, infer=False, ix=None, iy=None, n=0, x0=None, y0=None, x1=None, y1=None):
        y = y[:, :, tf.newaxis]
        x = self.embedding(x)
        # norm = tf.reshape(tf.repeat(tf.norm(x, axis=-1), self.e), shape=[-1, x.shape[1] ,self.e])
        # print(tf.divide(x, norm))
        x_2 = self.embedding(x_2)
        y_attn, _ = self.mha(y, x, x, x_2, x_mask, infer=infer, x=ix, y=iy, n=n, x0=x0, y0=y0, x1=x1, y1=y1)
        # y_attn = tf.nn.leaky_relu(y_attn)
        current_position = x[:, 1:, :]
        current_series = x_2[:, 1:, :]
        # print(current_series.shape)
        L = self.A1(y_attn) + self.A2(current_position) + self.A3(current_series)
        # print(self.A3(current_series)[:, -1, :])
        # print(self.A2(current_position)[:, -1, :])
        L = tf.nn.leaky_relu(L)
        L = tf.nn.leaky_relu(self.A4(L))
        L = tf.nn.leaky_relu(self.A5(L))
        L = self.A6(L)
        return tf.squeeze(L)
