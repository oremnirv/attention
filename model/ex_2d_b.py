###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
from tensorflow.keras import regularizers
from model import dot_prod_attention


class Decoder(tf.keras.Model):
    def __init__(self, e, l1=256, l2=128, l3=32, num_heads=1, input_vocab_size=2000, rate = 0.1):
        super(Decoder, self).__init__()
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
        self.l1 = l1


        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)



        self.e = e
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e)
        self.mha = dot_prod_attention.MultiHeadAttention2D(e, num_heads)
        self.A1 = tf.keras.layers.Dense(l1, name='A1')
        self.A2 = tf.keras.layers.Dense(l1, name='A2')
        self.A3 = tf.keras.layers.Dense(l1, name='A3')
        self.A4 = tf.keras.layers.Dense(l2, name='A4')
        self.A5 = tf.keras.layers.Dense(l3, name='A5')
        self.A6 = tf.keras.layers.Dense(2, name='A6')
        self.A7 = tf.keras.layers.Dense(2, name='A7')

    def call(self, x, x_2, y, training, x_mask, infer=False, ix=None, iy=None, n=0, x0=None, y0=None, x1=None, y1=None):
        batch_size = tf.shape(y)[0]
        yy = tf.reshape(tf.repeat(y, self.e), [batch_size, tf.shape(y)[1], self.e])
        y = y[:, :, tf.newaxis]


        xx = x_2
        x = self.embedding(x)
        x  = self.dropout(x, training)
        # norm = tf.reshape(tf.repeat(tf.norm(x, axis=-1), self.e), shape=[-1, x.shape[1] ,self.e])
        # print(tf.divide(x, norm))
        # tf.print(x_2[0][1])
        x_2 = self.embedding(x_2)
        x_2  = self.dropout1(x_2, training)
        # tf.print(x_2[0][1])
        y_attn, _ = self.mha(y, x, x, x_2, x_mask, infer=infer, x=ix, y=iy, n=n, x0=x0, y0=y0, x1=x1, y1=y1)
        print(y_attn)
        y_attn = self.dropout2(y_attn, training)
        y_attn = self.layernorm1(y_attn + yy)
        # y_attn = tf.nn.leaky_relu(y_attn)
        current_position = x[:, 1:, :]
        current_series = x_2[:, 1:, :]
        # print(current_series.shape)
        L0 = tf.math.add(self.A1(y_attn), self.A2(current_position)) 
        L0 = tf.math.add(self.A3(current_series), L0)
        L0 = tf.nn.leaky_relu(L0)
        # print(self.A3(current_series)[:, -1, :])
        # print(self.A2(current_position)[:, -1, :])
        L = self.dropout3(L0, training)
        L = self.layernorm2(L + L0)
        L1 = self.A4(L)
        L2 = tf.nn.leaky_relu(L1)
        L2 = self.dropout4(L1, training)
        L2 = self.layernorm3(L2 + L1)
        L2 = self.A6(L2) + self.A7(xx[:, 1:, tf.newaxis])
        return tf.squeeze(L2)
