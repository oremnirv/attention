###########################
# Author: Omer Nivron
###########################
from tensorflow.keras import regularizers
from model import dot_prod_attention
from keras import layers
import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(self, l, rate=0.1, num_heads = 8, depth = 8):
        super(Decoder, self).__init__()
        
        self.l = l
        self.num_heads = num_heads
        self.depth = depth
        
        self.wq = tf.keras.layers.Dense(l, name = 'wq')
        self.wk = tf.keras.layers.Dense(l, name = 'wk')
        self.wv = tf.keras.layers.Dense(l, name = 'wk')                    
        
        self.hq = tf.keras.layers.Dense(l, name = 'hq')
        self.hk = tf.keras.layers.Dense(l, name = 'hk')
        self.hv = tf.keras.layers.Dense(l, name = 'hv')

        self.dense = tf.keras.layers.Dense(l, name = 'dense_after_mha')

        
        self.B = tf.keras.layers.Dense(l, activation = 'relu', name = 'B')
        self.A = tf.keras.layers.Dense(1, name = 'A')

        self.Bsig = tf.keras.layers.Dense(l, activation = 'relu', name = 'Bsig')
        self.Asig = tf.keras.layers.Dense(1, name = 'Asig')


        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    #a call method, the layer's forward pass
    def call(self, tar_position, tar_inp, training, pos_mask, tar_mask, batch_size = 64):
        
        # Adding extra dimension to allow multiplication of 
        # a sequnce with itself. 
        tar_position = tar_position[:, :, tf.newaxis]
        
        q_p = self.wq(tar_position) 
        k_p = self.wk(tar_position)
        v_p = self.wk(tar_position)


        # print('v_p: ', v_p)
        #shape=(128, 59, 16)
        
        pos_attn1 = dot_prod_attention.dot_prod_position(q_p, k_p, v_p, mask = pos_mask)
        pos_attn1 = self.dropout1(pos_attn1, training = training)
        pos_attn1 = self.layernorm1(pos_attn1)
        # print('pos_attn1 :', pos_attn1)
#       shape=(128, 58, 16, 16)
    
        tar_inp = tar_inp[:, :, tf.newaxis]

        
        q = self.hq(tar_inp) 
        k = self.hk(tar_inp)
        v = self.hv(tar_inp)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # print('q :', q)
#       shape=(128, 58, 16)

        tar_attn1, _, _ = dot_prod_attention.dot_product_attention(q, k, v, tar_mask, head = True)


        tar_attn1 = tf.transpose(tar_attn1, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        tar_attn1 = tf.reshape(tar_attn1, 
                                  (batch_size, -1, self.l))  # (batch_size, seq_len_q, d_model)

        tar_attn1 = self.dense(tar_attn1)  # (batch_size, seq_len_q, d_model)

        tar_attn1 = self.dropout2(tar_attn1, training = training)
        tar_attn1 = self.layernorm2(tar_attn1)




        # tar_attn1 is (batch_size, max_seq_len - 1, tar_d_model)

        # print('tar_attn1 :', tar_attn1)
#       shape=(128, 58, l)
#       shape=(128, 58, 16)
        tar_attn1 = tar_attn1[:, :, :, tf.newaxis]
        
        tar1 = self.B(tar_attn1)
        
        # print('tar1 :', tar1)
        # shape=(128, 58, 16, 16)

        L = tf.matmul(tf.cast(tar1, tf.float64), tf.cast(pos_attn1, tf.float64))
        L = self.dropout3(L, training = training) 
        L = self.layernorm3(L)
        
        # print('L :', L)
        # shape=(128, 58, 16, 16)
        
        L2 = self.A(tf.reshape(L, shape = [tf.shape(L)[0], tf.shape(L)[1] ,self.l ** 2])) 

        Lsig1 = self.Bsig(L) 
        Lsig2 = self.Asig(tf.reshape(Lsig1, shape = [tf.shape(L)[0], tf.shape(L)[1] ,self.l ** 2])) 
        
#         print('L2 :', L2)
      # shape=(128, 58, 1)  
        
        return tf.squeeze(L2), tf.squeeze(Lsig2)