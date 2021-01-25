###########################
# Author: Omer Nivron
###########################
import tensorflow as tf
from model import dot_prod_attention


class Decoder(tf.keras.Model):
    def __init__(self, e, l1=512, l2=256, l3=32, rate=0, num_heads=1, input_vocab_size=400):
        super(Decoder, self).__init__()
        self.rate = rate
        self.e = e
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e)
        self.mha = dot_prod_attention.MultiHeadAttention(e, num_heads)
        self.A1 = tf.keras.layers.Dense(l1, name='A1')
        self.A2 = tf.keras.layers.Dense(l1, name='A2')
        self.A3 = tf.keras.layers.Dense(l2, name='A3')
        self.A4 = tf.keras.layers.Dense(l3, name='A4')
        self.A5 = tf.keras.layers.Dense(2, name='A5')

    # a call method, the layer's forward pass
    def call(self, x, y, training, x_mask):
        # x is [batch_s, dim, seq_len]
        # Adding extra dimension to allow multiplication of
        # a sequnce with itself.
        y = y[:, :, tf.newaxis]
        if len(x.shape) > 2:
            print('hi x')
            x = tf.stack([self.embedding(x[:, i, :] for i in range(x.shape[1]))])
        else:
            x = self.embedding(x)
        print(x.shape)
        attn, _ = self.mha(y, x, x, x_mask)
        attn = (tf.nn.leaky_relu(attn))
        curr_x = x[:, 1:, :]
        L2 = self.A1(attn) + self.A2(curr_x)
        L2 = tf.nn.leaky_relu(L2)
        L2 = tf.nn.leaky_relu(self.A3(L2))
        L2 = tf.nn.leaky_relu(self.A4(L2))
        L2 = self.A5(L2)
        return tf.squeeze(L2)
