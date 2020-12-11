###########################
# Author: Omer Nivron
###########################
from tensorflow.keras import regularizers
from model import dot_prod_attention
from keras import layers
import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(self, l, rate=0):
        super(Decoder, self).__init__()
        
        self.l = l
        
        self.wq = tf.keras.layers.Dense(l, name = 'wq')
        self.wk = tf.keras.layers.Dense(l, name = 'wk')
        self.wv = tf.keras.layers.Dense(l, name = 'wk')                    
        
        self.hq = tf.keras.layers.Dense(l, name = 'hq')
        self.hk = tf.keras.layers.Dense(l, name = 'hk')
        self.hv = tf.keras.layers.Dense(l, name = 'hv')
        
        self.B = tf.keras.layers.Dense(l, name = 'B')
        self.A = tf.keras.layers.Dense(1, name = 'A')

        self.A1 = tf.keras.layers.Dense(64, name = 'A1')
        self.A2 = tf.keras.layers.Dense(32, name = 'A2')
        self.A3 = tf.keras.layers.Dense(16, name = 'A3')
        self.A4 = tf.keras.layers.Dense(1, name = 'A4')


        self.Bsig = tf.keras.layers.Dense(l, activation = 'relu', name = 'Bsig')
        self.Asig = tf.keras.layers.Dense(1, name = 'Asig')


        # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    
        # self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)
        # self.dropout3 = tf.keras.layers.Dropout(rate)


    #a call method, the layer's forward pass
    def call(self, tar_position, tar_inp, training, pos_mask, tar_mask):
        
        # Adding extra dimension to allow multiplication of 
        # a sequnce with itself. 
        tar_position = tar_position[:, :, tf.newaxis]
        
        q_p = self.wq(tar_position) 
        k_p = self.wk(tar_position)
        v_p = self.wk(tar_position) 
        # v_p *= 1 - tf.cast(tar_mask, tf.float64)


        print('v_p: ', v_p)
        #shape=(128, 59, 16)
        
        pos_attn1, _, _ = dot_prod_attention.dot_product_attention(q_p, k_p, v_p, pos_mask)
        # pos_attn1 = self.dropout1(pos_attn1, training = training)
        # pos_attn1 = self.layernorm1(pos_attn1)
        print('pos_attn1 :', pos_attn1)
#       shape=(128, 58, 16, 16)
    
        tar_inp = tar_inp[:, :, tf.newaxis]

        
        q = self.hq(tar_inp) 
        k = self.hk(tar_inp)
        v = self.hv(tar_inp)
        
        print('q :', q)
#       shape=(128, 58, 16)

        tar_attn1, _, _ = dot_prod_attention.dot_product_attention(q, k, v, tar_mask)
        # tar_attn1 = self.dropout2(tar_attn1, training = training)
        # tar_attn1 = self.layernorm2(tar_attn1)
        # tar_attn1 is (batch_size, max_seq_len - 1, tar_d_model)

        print('tar_attn1 :', tar_attn1)
#       shape=(128, 58, l)
#       shape=(128, 58, 16)
        # tar_attn1 = tar_attn1[:, :, :, tf.newaxis]
        
        # tar1 = self.B(tar_attn1)
        
        print('tar1 :', tar_attn1)
        # shape=(128, 58, 16, 16)

        # L = tf.matmul(tf.cast(tar1, tf.float64), tf.cast(pos_attn1, tf.float64))
        L = tf.concat([tf.cast(tar_attn1, tf.float64), tf.cast(pos_attn1, tf.float64)], axis = 2)

        # L = self.dropout3(L, training = training) 
        # L = self.layernorm3(L)
        
        print('L :', L)
        # shape=(128, 58, 16, 16)
        
        # L2 = self.A(tf.reshape(L, shape = [tf.shape(L)[0], tf.shape(L)[1] ,self.l ** 2])) 
        L2 = tf.nn.leaky_relu(self.A1(L))
        print('L2 after A1', L2) 
        L2 = self.A2(L2) 
        L2 = tf.nn.leaky_relu(self.A3(L2)) 
        print('L2', L2)
        L2 = self.A4(L2)






        # Lsig1 = self.Bsig(L) 
        # Lsig2 = self.Asig(tf.reshape(Lsig1, shape = [tf.shape(L)[0], tf.shape(L)[1] ,self.l ** 2])) 
        
        print('L2 :', L2)
      # shape=(128, 58, 1)  
        
        return tf.squeeze(L2) #, tf.squeeze(Lsig2)