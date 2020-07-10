from model import classic_model, losses, dot_prod_attention
from data import data_generation, batch_creator, gp_kernels
from keras.callbacks import ModelCheckpoint
from helpers import helpers, masks
import tensorflow as tf
import time 


@tf.function
def train_step(decoder, optimizer_c, train_loss, m_tr, pos, tar):
    '''
    A typical train step function for TF2. Elements which we wish to track their gradient
    has to be inside the GradientTape() clause. see (1) https://www.tensorflow.org/guide/migrate 
    (2) https://www.tensorflow.org/tutorials/quickstart/advanced
    ------------------
    Parameters:
    pos (np array): array of positions (x values) - the 1st/2nd output from data_generator_for_gp_mimick_gpt
    tar (np array): array of targets. Notice that if dealing with sequnces, we typically want to have the targets go from 0 to n-1. The 3rd/4th output from data_generator_for_gp_mimick_gpt  
    pos_mask (np array): see description in position_mask function
    ------------------    
    '''
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    combined_mask_tar = masks.create_masks(tar_inp)
    pos_mask = masks.position_mask(pos)
    with tf.GradientTape(persistent=True) as tape:
        pred, pred_sig = decoder(pos, tar_inp, True, pos_mask, combined_mask_tar)
#         print('pred: ')
#         tf.print(pred)

        loss, mse, mask = losses.loss_function(tar_real, pred, pred_sig)

    gradients = tape.gradient(loss, decoder.trainable_variables)
#     tf.print(gradients)
# Ask the optimizer to apply the processed gradients.
    optimizer_c.apply_gradients(zip(gradients, decoder.trainable_variables))
    train_loss(loss)
    m_tr.update_state(mse, mask)


#     b = decoder.trainable_weights[0]
#     tf.print(tf.reduce_mean(b))



@tf.function
def test_step(decoder, test_loss, m_te, pos_te, tar_te):
    '''
    
    ---------------
    Parameters:
    pos (np array): array of positions (x values) - the 1st/2nd output from data_generator_for_gp_mimick_gpt
    tar (np array): array of targets. Notice that if dealing with sequnces, we typically want to have the targets go from 0 to n-1. The 3rd/4th output from data_generator_for_gp_mimick_gpt  
    pos_mask_te (np array): see description in position_mask function
    ---------------
    
    '''
    tar_inp_te = tar_te[:, :-1]
    tar_real_te = tar_te[:, 1:]
    combined_mask_tar_te = masks.create_masks(tar_inp_te)
    pos_mask_te = masks.position_mask(pos_te)
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    pred, pred_sig = decoder(pos_te, tar_inp_te, False, pos_mask_te, combined_mask_tar_te)
    t_loss, t_mse, t_mask = losses.loss_function(tar_real_te, pred, pred_sig)
    test_loss(t_loss)
    m_te.update_state(t_mse, t_mask)


def main():
    save_dir = '/home/ubuntu/GPT'
    pad_pos_tr, pad_pos_te, pad_y_fren_tr, pad_y_fren_te, _, df_te = data_generation.data_generator_for_gp_mimick_gpt(50000, gp_kernels.rbf_kernel)
    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    m_tr = tf.keras.metrics.Mean()
    m_te = tf.keras.metrics.Mean()
    writer = tf.summary.create_file_writer(save_dir + '/logs/')
    optimizer_c = tf.keras.optimizers.Adam()
    decoder = classic_model.Decoder(16)
    EPOCHS = 50
    batch_s  = 128
    run = 0; step = 0
    num_batches = int(pad_y_fren_tr.shape[0] / batch_s)
    tf.random.set_seed(1)    
    checkpoint = tf.train.Checkpoint(optimizer = optimizer_c, model = decoder)
    main_folder = "/home/ubuntu/GPT/ckpt/check_"
    folder = main_folder + str(run); helpers.mkdir(folder)
    tf.keras.backend.set_floatx('float64')

    with writer.as_default():
        for epoch in range(EPOCHS):
            start = time.time()

            for batch_n in range(num_batches):
                batch_pos_tr, batch_tar_tr, _ = batch_creator.create_batch_gp_mim_2(pad_pos_tr, pad_y_fren_tr)
                # batch_tar_tr shape := 128 X 59 = (batch_size, max_seq_len)
                # batch_pos_tr shape := 128 X 59 = (batch_size, max_seq_len)
                train_step(decoder, optimizer_c, train_loss, m_tr, batch_pos_tr, batch_tar_tr)

                if batch_n % 50 == 0:
                    batch_pos_te, batch_tar_te, _ = batch_creator.create_batch_gp_mim_2(pad_pos_te, pad_y_fren_te)
                    test_step(decoder, test_loss, m_te, batch_pos_te, batch_tar_te)
                    helpers.print_progress(epoch, batch_n, train_loss.result(), test_loss.result(), m_tr.result())
                    helpers.tf_summaries(run, step, train_loss.result(), test_loss.result(), m_tr.result(), m_te.result())
                    checkpoint.save(folder + '/')
                step += 1

            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    main()
