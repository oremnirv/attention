###########################
# Author: Omer Nivron
###########################
import tensorflow as tf

from model import dot_prod_attention


class Decoder(tf.keras.Model):
    def __init__(self, e, l1=512, l2=256, l3=32, rate=0.1, num_heads=1, input_vocab_size=720):
        super(Decoder, self).__init__()
        self.rate = rate
        self.e = e
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e, name='embedding_x')
        self.embedding_y = tf.keras.layers.Embedding(input_vocab_size, e, name='embedding_y')

        self.mha = dot_prod_attention.MultiHeadAttention(e, num_heads)
        self.mha2 = dot_prod_attention.MultiHeadAttention(e, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.A1 = tf.keras.layers.Dense(l1, name='A1')
        self.A2 = tf.keras.layers.Dense(l1, name='A2')
        self.A3 = tf.keras.layers.Dense(l2, name='A3')
        self.A4 = tf.keras.layers.Dense(l3, name='A4')
        self.A5 = tf.keras.layers.Dense(2, name='A5')

    def call(self, x, y, training, x_mask):
        y = self.embedding_y(y)
        x = self.embedding(x)
        attn, _ = self.mha(y, x, x, x_mask)
        attn_output = self.dropout1(attn, training=training)
        current_x = x[:, 1:, :]
        out1 = self.layernorm1(current_x + attn_output)  # (batch_size, input_seq_len, d_model)
        attn2, _ = self.mha2(out1, x, x, x_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.nn.leaky_relu(self.A1(out2))  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.nn.leaky_relu(self.A2(ffn_output))
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        L2 = tf.nn.leaky_relu(self.A3(out3))
        L2 = tf.nn.leaky_relu(self.A4(L2))
        L2 = self.A5(L2)
        return tf.squeeze(L2)
