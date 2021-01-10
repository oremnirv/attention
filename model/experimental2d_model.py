###########################
# Author: Omer Nivron
###########################
from tensorflow.keras import regularizers
from model import dot_prod_attention
from keras import layers
import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, e, l1 = 512, l2 = 256, l3=32, rate=0, num_heads = 1, input_vocab_size = 2000):
        super(Decoder, self).__init__()
        
        self.e = e
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, e)
        self.mha = dot_prod_attention.MultiHeadAttention2D(e, num_heads)              
        

        self.A1 = tf.keras.layers.Dense(l1, name = 'A1')

        self.A2 = tf.keras.layers.Dense(l1, name = 'A2')
        self.A3 = tf.keras.layers.Dense(l2, name = 'A3')
        self.A4 = tf.keras.layers.Dense(l3, name = 'A4')
        self.A5 = tf.keras.layers.Dense(2, name = 'A5')





    #a call method, the layer's forward pass
    def call(self, tar_position, tar_position_2, tar_inp, training, pos_mask):
        
        # Adding extra dimension to allow multiplication of 
        # a sequnce with itself. 

        # print('pos_mask: ', pos_mask)
        # print('tar_position shape: ', tar_position.shape)
        # tar_position = tar_position[:, :, tf.newaxis]
        tar_inp = tar_inp[:, :, tf.newaxis]
                # print('tar: ', tar)

        tar_position = self.embedding(tar_position)
        tar_position_2 = self.embedding(tar_position_2)
        # q = self.wq(tar_position)
        # k = self.wk(tar_position)
        # v = self.wv(tar_inp)
        # v_p *= 1 - tf.cast(tar_mask, tf.float64)
        # print('q_x: ', q)

        # print('k_x: ', k)
        # print('v_y: ', v)
        #shape=(128, 59, 16)

        # tar_attn1, _, _ = dot_prod_attention.dot_product_attention(q, k, v, pos_mask)

        tar_attn1, _ = self.mha(tar_inp, tar_position, tar_position, tar_position_2, pos_mask)


        tar_attn1 = (tf.nn.leaky_relu(tar_attn1))
        # print('tar_attn1: ', tar_attn1)


        
        current_position = tar_position[:, 1:, :]
        # print('current_position: ', current_position)
        L2 = self.A1(tar_attn1) + self.A2(current_position) 

        
        L2 = tf.nn.leaky_relu(L2)

        # print('L2', L2)
        L2 = tf.nn.leaky_relu(self.A3(L2)) 
        L2 = tf.nn.leaky_relu(self.A4(L2))
        L2 = self.A5(L2)


        
        # print('L2 :', L2)
      # shape=(128, 58, 1)  
        
        return tf.squeeze(L2) #, tf.squeeze(Lsig2)

