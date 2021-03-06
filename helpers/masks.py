import tensorflow as tf


def create_padding_mask(seq, pad_val=0):
    """
    Used to pad sequences that have zeros where there was no event.
    Typically this will be combined with create_look_ahead_mask function.
    This function is used inside an open session of tensorflow.
    To try it out create a tf.constant tensor.
    -------------------
    Parameters:
    seq (tensor): shape is (batch_size, seq_len)
    pad_val (int)
    -------------------
    Returns:
    A binary tensor  (batch_size, 1, seq_len): 1 where there was no event and 0 otherwise.

    """
    seq = tf.cast(tf.math.equal(seq, pad_val), tf.float32)
    return seq[:, tf.newaxis, :]


def create_tar_mask(size):
    """
    """
    mask = tf.linalg.diag(tf.ones(size, size))
    return mask


def create_look_ahead_mask(size):
    """
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
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(tar):
    """
    Create unified masking hiding future from current timestamps and hiding paddings.

    Example:
    masks.create_masks(tf.constant([1, 2, 5, 0, 0], shape = [1, 5]))
    (see main below)
    <tf.Tensor: shape=(1, 5, 5), dtype=float32, numpy=
    array([[[0., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.]]], dtype=float32)>

    -------------------
    Parameters:
    tar (tensor): batch of padded target sequences
    -------------------
    Returns:
    combined_mask_tar  (tensor): shape is batch_size X max_seq_len X max_seq_len
    """

    tar_padding_mask = create_padding_mask(tar)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    combined_mask_tar = tf.maximum(tar_padding_mask, look_ahead_mask)

    return combined_mask_tar




def main():
    a = create_masks(tf.constant([1, 2, 5, 3, 4], shape = [1, 5]))
    print(a[:, 1:, :-1])

if __name__ == '__main__':
    main()