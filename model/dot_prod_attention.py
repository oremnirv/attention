###########################
# Author: Omer Nivron
###########################
import tensorflow as tf


def dot_prod_position(q, k, v, mask):
    '''
    Used to create a pseudo XX^T covariance matrix for each 
    positional sequence in the batch.
    ------------------
    Parameters: 
    q : shape (batch_size X max_seq_len X l). Position outptut from create_batch_gp_mim_2 function (or after another Dense layer) 
    k : shape (batch_size X max_seq_len X l). Position outptut from create_batch_gp_mim_2 function (or after another Dense layer) 
    v : shape (batch_size X max_seq_len X l). Position outptut from create_batch_gp_mim_2 function (or after another Dense layer) 
    mask: shape (batch_size X (max_seq_len - 1) X max_seq_len X max_seq_len). The positional mask created by position_mask function and selected in batch indices 
    
    ------------------
    Returns:
    u2 (tf.tensor float64): shape (batch_size X (max_seq_len - 1) X l X l).
    Each observation (1st dim) has seq_len - 1 timestamps (2nd dim) and each timestamp has an associated
    l X l pseudo covariance matrix (3rd & 4th dims).
    
    '''
    qk = tf.matmul(q, k, transpose_b = True)
    qk = tf.cast(qk[:, tf.newaxis, :, :], tf.float64)
#     print('qk1: ', qk)
#     shape=(128, 1, 59, 59)

#     print('pos_mask: ', mask)
#     shape=(128, 58, 59, 59)
    if mask is not None:
        qk +=  ((tf.cast(mask, tf.float64)) * -1e9)
        
#     print('qk2: ', qk)
# shape=(128, 58, 59, 59)

    qk = tf.reshape(qk, shape = [tf.shape(mask)[0], tf.shape(mask)[1], -1])
    
#     print('qk3: ', qk)
#     shape=(128, 58, 3481)
    
    qk = tf.reshape(tf.nn.softmax(qk, axis = -1), shape = [tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(mask)[2], tf.shape(mask)[3]])
    
#     print('qk4: ', qk)
    #shape=(128, 58, 59, 59)
    
    v = tf.cast(v[:, tf.newaxis, :, :], tf.float64)
    
    u = tf.transpose(tf.matmul(qk, v), perm = [0, 1, 3 ,2])
    
#     print('u: ', u)
#     shape=(128, 58, 16, 59)
    
    u2 = tf.matmul(u, v)
    
    
    return u2



def dot_product_attention(q, k, v, mask):
    '''
    Attention inspired by Transformer (but not the same). The Transformer embeds the 
    target words to q (query), k (key), v (value). So if we have a batch of 128 sequences 
    with max length 40 and embedding layer is 20, we will get shape q = shape k = shape v
    = (128 X  max sequence length X 20). The Transformer then transposes k 
    to get after matmul (128 X max seq X max seq) matrix. We then apply relu layer (unlike in Transformer)
    ---------------------
    Parameters:
    q (tf.tensor float64): shape (batch_size, max_seq_len - 1, l)
    k (tf.tensor float64): shape (batch_size, max_seq_len - 1, l)
    v (tf.tensor float64): shape (batch_size, max_seq_len - 1, l)
    mask (tf.tensor float64): shape (batch_size, max_seq_len - 1, max_seq_len - 1)
    ---------------------
    Returns:
    out_tar: shape (batch_size, max_seq_len - 1, l). The sequences after embedding (or Dense layer) weighted by attention_weights. 
    attention_weights : shape (batch_size, max_seq_len - 1, max_seq_len - 1). Weights to assign for each sequence member at each timestamp (2nd dim).
    matmul_qk: shape (batch_size, max_seq_len - 1, max_seq_len - 1)
    
    
    '''
    # similarity
    # q = k = v  shape := (batch_size, max_seq_len - 1, l)
    mask = mask[:, tf.newaxis, :, :]
    # print('mask: ', mask)

    matmul_qk = tf.matmul(q, k, transpose_b = True, name = 'qk')
    # print('matmul_qk before filter: ', matmul_qk)
    # matmul_qk = matmul_qk[:, 1:, :-1]
    matmul_qk = matmul_qk[:, :, 1:, :-1]
    # print('matmul_qk: ', matmul_qk)
    # print('1', matmul_qk)


    dk = tf.cast(tf.shape(k)[-1], tf.float64)
    nl_qk = tf.cast(tf.nn.relu(matmul_qk / tf.math.sqrt(dk), name = 'nl_qk'), tf.float64) 

    if mask is not None:
    # mask[:, tf.newaxis, :, :]
        nl_qk +=  ((tf.cast(mask, tf.float64)) * -1e9)
            # print('2', nl_qk)
    
        
    # print('nl_qk after mask: ', nl_qk)
#     shape=(128, 58, 58)
        
     # turn simialrity to scores
    attention_weights = tf.nn.softmax(nl_qk, axis = -1, name = 'attention_weights')
    # Notice that for all the rows where 
    # everything is 0, the masking will turn everything to -inf
    # and the output from the softmax would be 1/num_cols 
    # (try a = tf.constant([-1e9, -1e9, -1e9]), tf.nn.softmax(a))
    # So we can expect an output from these rows which we want to ignore
    # this will be enforced in the masking of the loss function 
    
    # print('attention_weights: ', attention_weights)
#     shape=(128, 58, 58)
   
    # weight values 
    # attention_weights shape := (batch_size, max_seq_len - 1, max_seq_len - 1), 
    # v shape := batch_size X (max_seq_len - 1) X l
    out_tar = tf.matmul(attention_weights, tf.cast(v, tf.float64))
    
    # print('out_tar: ', out_tar)
#   shape=(128, 58, l)
    
    return out_tar, attention_weights, matmul_qk

# Instead of one single attention head, Q, K, and V are split into multiple heads because 
# it allows the model to jointly attend to information at different positions from different representational space


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model, name = 'wq')
    self.wk = tf.keras.layers.Dense(d_model, name = 'wk')
    self.wv = tf.keras.layers.Dense(d_model, name = 'wv')  

    self.hq = tf.keras.layers.Dense(self.depth, name = 'hq')
    self.hk = tf.keras.layers.Dense(self.depth, name = 'hk')
    self.hv = tf.keras.layers.Dense(self.depth, name = 'hv')  

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]



    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    q = self.hq(tf.nn.leaky_relu(q))
    k = self.hk(tf.nn.leaky_relu(k))
    v = self.hv(tf.nn.leaky_relu(v))

    # print('q: ', q)
    # print('v: ', v)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights, _ = dot_product_attention(
        q, k, v, mask)

    # print('scaled_attention: ', scaled_attention)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    # print('scaled_attention after transpose: ', scaled_attention)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    # print('concat_attention: ', concat_attention)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights