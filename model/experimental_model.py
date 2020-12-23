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

        self.BN1 = tf.keras.layers.BatchNormalization(name = 'BN1') 
        self.BN2 = tf.keras.layers.BatchNormalization(name = 'BN2') 
        self.BN3 = tf.keras.layers.BatchNormalization(name = 'BN3') 
        self.BN4 = tf.keras.layers.BatchNormalization(name = 'BN4') 
        self.BN5 = tf.keras.layers.BatchNormalization(name = 'BN5') 
        self.BN6 = tf.keras.layers.BatchNormalization(name = 'BN6') 
        self.BN7 = tf.keras.layers.BatchNormalization(name = 'BN7') 

        
        self.wq = tf.keras.layers.Dense(l, name = 'wq')
        self.wk = tf.keras.layers.Dense(l, name = 'wk')
        self.wv = tf.keras.layers.Dense(l, name = 'wv')                    
        
        self.hq = tf.keras.layers.Dense(l, name = 'hq')
        self.hk = tf.keras.layers.Dense(l, name = 'hk')
        self.hv = tf.keras.layers.Dense(l, name = 'hv')


        self.A2 = tf.keras.layers.Dense(128, name = 'A2')
        self.A3 = tf.keras.layers.Dense(128, name = 'A3')
        self.A4 = tf.keras.layers.Dense(32, name = 'A4')
        self.A5 = tf.keras.layers.Dense(2, name = 'A5')





    #a call method, the layer's forward pass
    def call(self, tar_position, tar_inp, training, pos_mask):
        
        # Adding extra dimension to allow multiplication of 
        # a sequnce with itself. 

        print('pos_mask: ', pos_mask)
        # print('tar_position shape: ', tar_position.shape)
        tar_position = tar_position[:, :, tf.newaxis]
        tar_inp = tar_inp[:, :, tf.newaxis]
                # print('tar: ', tar)

        q = self.wq(tar_position)
        k = self.wk(tar_position)
        v = self.wv(tar_inp)
        # v_p *= 1 - tf.cast(tar_mask, tf.float64)
        print('q_x: ', q)

        print('k_x: ', k)
        print('v_y: ', v)
        #shape=(128, 59, 16)

        tar_attn1, _, _ = dot_prod_attention.dot_product_attention(q, k, v, pos_mask)

        print('tar_attn1: ', tar_attn1)


        
        current_position = tar_position[:, 1:, :]
        print('current_position: ', current_position)
        L2 = self.A2(tar_attn1) + self.A3(current_position) 

        
        L2 = tf.nn.leaky_relu(L2)

        # print('L2', L2)
        L2 = tf.nn.leaky_relu(self.A4(L2)) 
        L2 = self.A5(L2)



        
        # print('L2 :', L2)
      # shape=(128, 58, 1)  
        
        return tf.squeeze(L2) #, tf.squeeze(Lsig2)