import sys
sys.path.append("..")
import tensorflow as tf
from helpers import masks
from model import losses


def build_graph():
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    m_tr = tf.keras.metrics.Mean()
    m_te = tf.keras.metrics.Mean()

    @tf.function
    def train_step(decoder, optimizer_c, train_loss, m_tr, x, y, context_p=50, d=False, x2=None, to_gather=None):
        """

        :param decoder:
        :param optimizer_c:
        :param train_loss:
        :param m_tr:
        :param x:
        :param y:
        :param context_p:
        :param d:
        :param x2:
        :param to_gather:
        :return:
        """
        y_inp = y[:, :-1]
        y_real = y[:, 1:]
        xx = x
        if len(x.shape) > 2:
            xx = x[:, 0, :]
        combined_mask_x = masks.create_masks(xx)
        with tf.GradientTape(persistent=True) as tape:

            pred = decoder(x, x2, y_inp, True, combined_mask_x[:, :-1, :-1])

            pred0 = tf.squeeze(pred[:, :, 0])
            pred1 = tf.squeeze(pred[:, :, 1])
            loss, mse, mask = losses.loss_function(tf.gather_nd(y_real, to_gather, name='real'), pred= tf.gather_nd(pred0, to_gather, name='mean'),
                                                       pred_log_sig=tf.gather_nd(pred1, to_gather, name='log_sig'))

        gradients = tape.gradient(loss, decoder.trainable_variables)
        optimizer_c.apply_gradients(zip(gradients, decoder.trainable_variables))
        train_loss(loss)
        m_tr.update_state(mse)
        names = [v.name for v in decoder.trainable_variables]
        shapes = [v.shape for v in decoder.trainable_variables]
        return pred[:, :, 0], pred[:, :, 1], decoder.trainable_variables, names, shapes

    @tf.function
    def test_step(decoder, test_loss, m_te, x_te, y_te, context_p=50, d=False, x2_te=None, to_gather=None):
        """

        :param decoder:
        :param test_loss:
        :param m_te:
        :param x_te:
        :param y_te:
        :param context_p:
        :param d:
        :param x2_te:
        :return:
        """
        y_inp_te = y_te[:, :-1]
        y_real_te = y_te[:, 1:]
        xx = x_te
        if len(x_te.shape) > 2:
            xx = x_te[:, 0, :]
        combined_mask_x_te = masks.create_masks(xx)
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        pred_te = decoder(x_te, x2_te, y_inp_te, False, combined_mask_x_te[:, :-1, :-1])

        pred0_te = tf.squeeze(pred_te[:, :, 0])
        pred1_te = tf.squeeze(pred_te[:, :, 1])
        t_loss, t_mse, t_mask = losses.loss_function(tf.gather_nd(y_real_te, to_gather, name='real_te'), pred= tf.gather_nd(pred0_te, to_gather, name='mean_te'),
                                                   pred_log_sig=tf.gather_nd(pred1_te, to_gather, name='log_sig_te'))

        test_loss(t_loss)
        m_te.update_state(t_mse)
        return pred_te[:, :, 0], pred_te[:, :, 1]

    tf.keras.backend.set_floatx('float64')
    return train_step, test_step, loss_object, train_loss, test_loss, m_tr, m_te
