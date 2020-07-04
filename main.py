import tensorflow as tf

@tf.function
def train_step(pos, tar, pos_mask):
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
    combined_mask_tar = create_masks(tar_inp)
    with tf.GradientTape(persistent=True) as tape:
        pred = decoder(pos, tar_inp, True, pos_mask, combined_mask_tar)
#         print('pred: ')
#         tf.print(pred)

        loss = loss_function(tar_real, pred)

    gradients = tape.gradient(loss, decoder.trainable_variables)
#     tf.print(gradients)
# Ask the optimizer to apply the processed gradients.
    optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))
    train_loss(loss)
#     b = decoder.trainable_weights[0]
#     tf.print(tf.reduce_mean(b))



@tf.function
def test_step(pos_te, tar_te, pos_mask_te):
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
    combined_mask_tar_te = create_masks(tar_inp_te)
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
    pred = decoder(pos_te, tar_inp_te, False, pos_mask_te, combined_mask_tar_te)
    t_loss = loss_function(tar_real_te, pred)
    test_loss(t_loss)


def main():
    writer = tf.summary.create_file_writer(save_dir + '/logs/')
    optimizer_c = tf.keras.optimizers.Adam()
    decoder = Decoder(16)
    EPOCHS = 50
    batch_s  = 128
    run = 0; step = 0
    num_batches = int(pad_y_fren_tr.shape[0] / batch_s)
    tf.random.set_seed(1)    
    checkpoint = tf.train.Checkpoint(optimizer = optimizer_c, model = decoder)
    main_folder = "/Users/omernivron/Downloads/GPT/ckpt/check_"
    folder = main_folder + str(run); helpers.mkdir(folder)

    with writer.as_default():
        for epoch in range(EPOCHS):
            start = time.time()

            for batch in range(num_batches):
                batch_pos_tr, batch_tar_tr, batch_pos_mask, _ = create_batch_gp_mim_2(pad_pos_tr, pad_y_fren_tr, pp)
                # batch_tar_tr shape := 128 X 59 = (batch_size, max_seq_len)
                # batch_pos_tr shape := 128 X 59 = (batch_size, max_seq_len)
                train_step(batch_pos_tr, batch_tar_tr, batch_pos_mask)

                if batch % 50 == 0:
                    batch_pos_te, batch_tar_te, batch_pos_mask_te, _ = create_batch_gp_mim_2(pad_pos_te, pad_y_fren_te, pp_te)
                    test_step(batch_pos_te, batch_tar_te, batch_pos_mask_te)
                    helpers.print_progress(epoch, batch_n, train_loss.result(), test_loss.result())
                    helpers.tf_summaries(run, step, train_loss.result(), test_loss.result())
                    checkpoint.save(folder + '/')
                step += 1

            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    main()
