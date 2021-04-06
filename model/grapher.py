import sys

sys.path.append("..")
import tensorflow as tf
from helpers import masks
from model import losses


def build_graph():

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    m_tr = tf.keras.metrics.Mean()
    m_te = tf.keras.metrics.Mean()

    @tf.function
    def train_step(decoder, optimizer_c, train_loss, m_tr, x, y, d=False, to_gather=None):
        """
        # Examples for using @tf.function: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
        # Examples for using tf.GradientTape: https://www.tensorflow.org/api_docs/python/tf/GradientTape
        :param decoder: tf.keras.Model
        :param optimizer_c: tensorflow.python.keras.optimizer_v2
        :param train_loss: tf.keras.metrics
        :param m_tr: tf.keras.metrics object
        :param x: (tf.tensor)
        :param y: (tf.tensor)
        :param d: (bool) TRUE if we are dealing with pairs oof sequences
        :param x2: (tf.tensor) if d = True, otherwise None
        :param to_gather: (np.array) array sized the same as y, in each row the context points will be indicated by 0 else
        1s.
        :return:
        (tf.tensor) of mean predictions, (tf.tensor) of log sigma predictions,
        (tf.tensor) of weights, (tf.tensor) of names of weights, (tf.tensor) of shapes of weights

        """
        y_inp = y[:, :-1]
        y *= to_gather  # this is the step to make sure we only consider non context points in prediction
        y_real = y[:, 1:]
        combined_mask_x = masks.create_masks(tf.squeeze(x[:, :, 0])) # see masks.py for description
        with tf.GradientTape(persistent=True) as tape:
            if d:
                # tf.print(y_inp)
                pred = decoder(x, y_inp, True, combined_mask_x[:, :-1, :-1]) # (batch_size x seq_len x 2)
            else:
                pred = decoder(x, y_inp, True, combined_mask_x[:, :-1, :-1]) # (batch_size x seq_len x 2)

            loss, mse, mask = losses.loss_function(y_real, pred=pred[:, :, 0],
                                                       pred_log_sig=pred[:, :, 1])
        gradients = tape.gradient(loss, decoder.trainable_variables)
        optimizer_c.apply_gradients(zip(gradients, decoder.trainable_variables))
        train_loss(loss)
        m_tr.update_state(mse)
        names = [v.name for v in decoder.trainable_variables]
        shapes = [v.shape for v in decoder.trainable_variables]
        return pred[:, :, 0], pred[:, :, 1], decoder.trainable_variables, names, shapes, y_real, to_gather

    @tf.function
    def test_step(decoder, test_loss, m_te, x_te, y_te, d=False, to_gather=None):
        """

        :param decoder: tf.keras.Model
        :param test_loss: tf.keras.metrics
        :param m_te: tf.keras.metrics
        :param x_te: (tf.tensor)
        :param y_te: (tf.tensor)
        :param d: (bool) TRUE if we are dealing with pairs of sequences
        :param x2_te: (tf.tensor) if d = True, otherwise None
        :param to_gather: (np.array) array sized the same as y, in each row the context points will be indicated by 0 else
        1s.
        :return:
        """
        y_inp_te = y_te[:, :-1]
        y_te *= to_gather
        y_real_te = y_te[:, 1:]
        combined_mask_x_te = masks.create_masks(tf.squeeze(x_te[:, :, 0]))
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if d:
            pred_te = decoder(x_te, y_inp_te, False, combined_mask_x_te[:, :-1, :-1])
        else:
            pred_te = decoder(x_te, y_inp_te, False, combined_mask_x_te[:, :-1, :-1])

        t_loss, t_mse, t_mask = losses.loss_function(y_real_te, pred=pred_te[:, :, 0],
                                                         pred_log_sig=pred_te[:, :, 1])
        test_loss(t_loss)
        m_te.update_state(t_mse)
        return pred_te[:, :, 0], pred_te[:, :, 1]

    tf.keras.backend.set_floatx('float64')
    return train_step, test_step, train_loss, test_loss, m_tr, m_te
