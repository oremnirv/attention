import tensorflow as tf
import numpy as np


def position_mask(arr):
    '''
    This tries to emulate the kernel matrix. 
    In the first stage we have a 2X2 matrix of zeros, next
    3X3 matrix of zeros, etc.
    -------------------------
    Parameters:
    arr (np array): the 1st/2nd output from data_generator_for_gp_mimick_gpt function
    -------------------------
    Returns:
    mask (4D np array): if there are 100 rows and 50 cols in arr then this will 
    return [100, 49, 50, 50] array -- where the first dim is observation number 
    second dim is timestamp and third+fourth dim are the mask matrix.
    '''
    rows = arr.shape[0]
    cols = arr.shape[1]
    mask = np.ones((rows, cols - 1, cols, cols))
    specific = np.sum(np.equal(arr, 0), 1)
    for i in range(2, cols + 1):
        mask[:, i - 2, :i, :i] = np.zeros((i, i))
    for j in range(rows):
        k  = specific[j]
        mask[j, k:, :, :] = 1
            
    return mask


def create_padding_mask(seq):
    '''
    Used to pad sequences that have zeros where there was no event.
    Typically this will be combined with create_look_ahead_mask function.
    This function is used inside an open session of tensorflow. 
    To try it out create a tf.constant tensor.
    -------------------
    Parameters:
    seq (tensor): shape is (batch_size, seq_len)
    
    -------------------
    Returns:
    A binary tensor  (batch_size, 1, seq_len): 1 where there was no event and 0 otherwise.
    
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention. Extra dimension is used in create_masks function
    return seq[:, tf.newaxis, :]  


def create_tar_mask(size):
    '''
    '''
    mask = tf.linalg.diag(tf.ones(size, size))
    return mask

def create_look_ahead_mask(size):
    '''
    Hide future outputs from a decoder style network.
    Used typically together with create_padding_mask function
    -----------------------
    Parameters:
    size (int): max sequnce length 
    
    -----------------------
    Returns:
    mask (tensor): shape is (seq_len X seq_len). Example: if size is 4, returns
    0 1 1 1
    0 0 1 1
    0 0 0 1
    0 0 0 0 
    where 1 signifies what to hide.
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(tar):
    '''
    Create unified masking hiding future from current timestamps and hiding paddings. 
    -------------------
    Parameters: 
    tar (tensor): batch of padded target sequences 
    -------------------
    Returns: 
    combined_mask_tar  (tensor): shape is batch_size X max_seq_len X max_seq_len
    '''
    
    tar_padding_mask = create_padding_mask(tar)
    ## this will be batch_size X 1 X 40

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    # if max seq length is 40 -- > this will be 40X40 
    
    
    ## This will also be (64, 40, 40)
    combined_mask_tar = tf.maximum(tar_padding_mask, look_ahead_mask)
    
    
    return combined_mask_tar


