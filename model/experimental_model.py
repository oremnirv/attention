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
    def call(self, tar_position, tar_inp, training, pos_mask):
        # Adding extra dimension to allow multiplication of 
        # a sequnce with itself.
        tar_inp = tar_inp[:, :, tf.newaxis]
        tar_position = self.embedding(tar_position)
        tar_attn1, _ = self.mha(tar_inp, tar_position, tar_position, pos_mask)
        tar_attn1 = (tf.nn.leaky_relu(tar_attn1))
        current_position = tar_position[:, 1:, :]
        L2 = self.A1(tar_attn1) + self.A2(current_position)
        L2 = tf.nn.leaky_relu(L2)
        L2 = tf.nn.leaky_relu(self.A3(L2))
        L2 = tf.nn.leaky_relu(self.A4(L2))
        L2 = self.A5(L2)
        return tf.squeeze(L2)
