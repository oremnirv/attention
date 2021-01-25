import sys

sys.path.append("..")
import tensorflow as tf
from helpers import masks
from model import losses


def build_graph():
    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    m_tr = tf.keras.metrics.Mean()
    m_te = tf.keras.metrics.Mean()

    @tf.function
    def train_step(decoder, optimizer_c, train_loss, m_tr, x, y, context_p=50, d=False, x2=None):
        """
        A typical train step function for TF2. Elements which we wish to track their gradient has to be inside the
        GradientTape() clause. see (1) https://www.tensorflow.org/guide/migrate (2)
        https://www.tensorflow.org/tutorials/quicksyt/advanced ------------------ Parameters: x (np array): array of
        xitions (x values) - the 1st/2nd output from data_generator_for_gp_mimick_gpt y (np array): array of ygets.
        Notice that if dealing with sequnces, we typically want to have the ygets go from 0 to n-1. The 3rd/4th
        output from data_generator_for_gp_mimick_gpt x_mask (np array): see description in xition_mask function
        ------------------
        """
        y_inp = y[:, :-1]
        y_real = y[:, 1:]
        xx = x
        if len(x.shape) > 2:
            xx = x[:, 0, :]
        combined_mask_x = masks.create_masks(xx)
        with tf.GradientTape(persistent=True) as tape:
            if d:
                pred = decoder(x, x2, y_inp, True, combined_mask_x[:, 1:, :-1])
            else:
                pred = decoder(x, y_inp, True, combined_mask_x[:, 1:, :-1])
            loss, mse, mask = losses.loss_function(y_real[:, context_p:], pred=pred[:, context_p:, 0],
                                                   pred_log_sig=pred[:, context_p:, 1])

        gradients = tape.gradient(loss, decoder.trainable_variables)
        optimizer_c.apply_gradients(zip(gradients, decoder.trainable_variables))
        train_loss(loss)
        m_tr.update_state(mse, mask)
        names = [v.name for v in decoder.trainable_variables]
        shapes = [v.shape for v in decoder.trainable_variables]

        return pred[:, :, 0], pred[:, :, 1], decoder.trainable_variables, names, shapes

    @tf.function
    def test_step(decoder, test_loss, m_te, x_te, y_te, context_p=50, d=False, x2_te=None):
        """
        --------------- Parameters: x (np array): array of xitions (x values) - the 1st/2nd output from
        data_generator_for_gp_mimick_gpt y (np array): array of ygets. Notice that if dealing with sequnces,
        we typically want to have the ygets go from 0 to n-1. The 3rd/4th output from
        data_generator_for_gp_mimick_gpt x_mask_te (np array): see description in xition_mask function ---------------
        """
        y_inp_te = y_te[:, :-1]
        y_real_te = y_te[:, 1:]
        combined_mask_x_te = masks.create_masks(x_te)
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if d:
            pred_te = decoder(x_te, x2_te, y_inp_te, False, combined_mask_x_te[:, 1:, :-1])
        else:
            pred_te = decoder(x_te, y_inp_te, False, combined_mask_x_te[:, 1:, :-1])
        t_loss, t_mse, t_mask = losses.loss_function(y_real_te[:, context_p:], pred=pred_te[:, context_p:, 0],
                                                     pred_log_sig=pred_te[:, context_p:, 1])

        test_loss(t_loss)
        m_te.update_state(t_mse, t_mask)
        return pred_te[:, :, 0], pred_te[:, :, 1]

    tf.keras.backend.set_floatx('float64')
    return train_step, test_step, loss_object, train_loss, test_loss, m_tr, m_te
