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
        self.wv = tf.keras.layers.Dense(l, name = 'wk')                    
        
        self.hq = tf.keras.layers.Dense(l, name = 'hq')
        self.hk = tf.keras.layers.Dense(l, name = 'hk')
        self.hv = tf.keras.layers.Dense(l, name = 'hv')


        self.A2 = tf.keras.layers.Dense(128, name = 'A2')
        self.A4 = tf.keras.layers.Dense(2, name = 'A4')


        self.Bsig = tf.keras.layers.Dense(32, name = 'Bsig')
        self.Asig = tf.keras.layers.Dense(128, name = 'Asig')

    #a call method, the layer's forward pass
    def call(self, tar_position, current_pos, tar_inp, training, pos_mask, tar_mask):
        
        # Adding extra dimension to allow multiplication of 
        # a sequnce with itself. 
        # print('tar_position shape: ', tar_position.shape)
        
        tar_position = tar_position[:, :, tf.newaxis]
        tar_inp = tar_inp[:, :, tf.newaxis]
        
        tar = tf.concat([tar_position, tar_inp], axis = 2)
        print('tar: ', tar)

        q = self.BN1(self.wq(tar))
        k = self.BN2(self.wk(tar))
        v = self.BN3(self.wv(tar))
        # v_p *= 1 - tf.cast(tar_mask, tf.float64)


        print('v: ', v)
        #shape=(128, 59, 16)

        tar_attn1, _, _ = dot_prod_attention.dot_product_attention(q, k, v, tar_mask)


        

        L2 = self.A2(tar_attn1) + self.Asig(current_pos[:, :, tf.newaxis]) 

        
        L2 = tf.nn.leaky_relu(self.BN7(L2)) 

        # print('L2', L2)
        L2 = tf.nn.leaky_relu(self.Bsig(L2)) 
        L2 = self.A4(L2)



        
        # print('L2 :', L2)
      # shape=(128, 58, 1)  
        
        return tf.squeeze(L2) #, tf.squeeze(Lsig2)