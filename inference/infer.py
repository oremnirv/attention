from helpers import masks
import tensorflow as tf
import numpy as np
from data import data_generation


def evaluate(model, x, y, sample=True, d=False, x_2=None, xx=None, yy=None, c_step=0, infer=False, x0=None, y0=None,
             x1=None, y1=None):
    """
    This function just makes a call to the forward pass of the model. If sample=True, then we sample from
    N(mu, exp(log sigma)) where mu and log sigma are the last outputs from the forward pass, otherwise we just report
    the mean mu.

    :param model: (tf.keras.Model) a trained model
    :param x: (tf.tensor) see inference function for details
    :param y: (tf.tensor) see inference function for details
    :param sample: (bool) Whether to infer the mean (mu) or to sample from N(mu, sigma)
    :param d: (bool) Is it infernce between pairs of sequnces (TRUE) or just one sequence at a time
    :param infer: (bool) If TRUE then will plot attention given to points while making each prediction
    :param c_step: (int) the current index to be predicted
    :param x_2: (tf.tensor) each entry represents the series member (0/1) a certain value is assocated with. shape is (m + num_steps)x1.
    :param y1: (tf.tensor) the true y-vals associated with pair element #1. Used only for plotting attention
    :param x1: (tf.tensor) the true x-vals associated with pair element #1. Used only for plotting attention
    :param y0: (tf.tensor) the true y-vals associated with pair element #0. Used only for plotting attention
    :param x0: (tf.tensor) the true x-vals associated with pair element #0. Used only for plotting attention
    :param xx: (tf.tensor) unsorted true x-vals associated with both elements of the sequence pair. Used only for plotting attention
    :param yy: (tf.tensor) unsorted true y-vals associated with both elements of the sequence pair. Used only for plotting attention
    :return:
    mean, variance, sample value for the most recent prediction

    To run an example of evaluate for 1 step returning a mean and log sigma on UNN notebook:
        a) load model
        b) choose random index: idx = int(np.random.choice(np.arange(0, 30000, 1), 1))
        c) from helpers import masks
        d) combined_mask_x = masks.create_masks(data[3][idx, :51].reshape(1, -1))[:, :-1, :-1]
        e) decoder(x = data[3][idx, :51].reshape(1, -1), x_2 = data[0][idx, :51].reshape(1, -1), y = data[6][idx, :50].reshape(1, -1), training = False, x_mask = combined_mask_x)
        f) the last row of the output represents the mean prediction and log sigma prediction

    If you would like to compare the output of inferecne to that of a decoder call, run the steps above
    and then run steps (c) and (d) from inference below. compare the 50th index in the output of
    inference with the last row extracted from running the steps (a-f) above.
    """
    combined_mask_x = masks.create_masks(x)
    if d:
        pred = model(x, x_2, y, False, combined_mask_x[:, :-1, :-1], infer=infer, ix=xx, iy=yy, n=c_step, x0=x0, y0=y0,
                     x1=x1, y1=y1)
    else:
        pred = model(x, y, False, combined_mask_x[:, :-1, :-1])
    if sample:
        sample_y = np.random.normal(pred[-1, 0], np.exp(pred[-1, 1]))
    else:
        sample_y = pred[-1, 0]

    return pred[:, 0], pred[:, 1], sample_y


def inference(model, x, y, em_y,  num_steps=1, sample=True, d=False, x_2=None, infer=False, xx=None, yy=None,
              x0=None, y0=None, x1=None, y1=None):
    """
    This is a recursive function for inference. It receives y-vals, x-vals and the series which they came from (x_2-vals).
    At every inference step, if y is of length n, x and x_2 are of length n+1. We then make a prediction for y[n+1]
    using the evaluate function, attach the result to y, retrieve x[n+2], x_2[n+2] and repeat until num_steps is done.

    :param model: (tf.keras.Model) a trained model
    :param x: (tf.tensor) shape is (m + num_steps)x1
    :param y: (tf.tensor mx1) target variable
    :param num_steps: (int) how many forward steps to infer. If num_steps=999, then infer all remaining points
    :param sample: (bool) Whether to infer the mean (mu) or to sample from N(mu, sigma)
    :param d: (bool) Is it infernce between pairs of sequnces (TRUE) or just one sequence at a time
    :param x_2: (tf.tensor) each entry represents the series member (0/1) a certain value is assocated with. shape is (m + num_steps)x1.
    :param infer: (bool) If TRUE then will plot attention given to points while making each prediction
    :param y1: (tf.tensor) the true y-vals associated with pair element #1. Used only for plotting attention
    :param x1: (tf.tensor) the true x-vals associated with pair element #1. Used only for plotting attention
    :param y0: (tf.tensor) the true y-vals associated with pair element #0. Used only for plotting attention
    :param x0: (tf.tensor) the true x-vals associated with pair element #0. Used only for plotting attention
    :param xx: (tf.tensor) unsorted true x-vals associated with both elements of the sequence pair. Used only for plotting attention
    :param yy: (tf.tensor) unsorted true y-vals associated with both elements of the sequence pair. Used only for plotting attention
    :return: model object, x-vals, y-vals, number of steps

    To run an example of inference for 5 steps returning a sample on UNN notebook:
        a) load model
        b) choose random index: idx = int(np.random.choice(np.arange(0, 30000, 1), 1))
        c) from inference import infer
        d) infer.inference(decoder, x=data[3][idx, :].reshape(1, -1), y=data[6][idx, :50].reshape(1, -1), num_steps=5,
                                          sample=True, d=True, x_2=data[0][idx, :].reshape(1, -1))
    """
    n = y.shape[1]
    num_steps = x.shape[1] - n if num_steps == 999 else min(num_steps, x.shape[1] - n)
    temp_x = x[:, :(n + 1)]
    if d:
        temp_x2 = x_2[:, :(n + 1)]
        pred, pred_log_sig, sample_y = evaluate(model, temp_x, y, d=True, x_2=temp_x2, sample=sample, infer=infer,
                                                xx=xx, yy=yy, c_step=n, x0=x0, y0=y0, x1=x1, y1=y1)
    else:
        pred, pred_log_sig, sample_y = evaluate(model, temp_x, em_y, sample=sample)

    y = tf.concat((y, tf.reshape(sample_y, [1, 1])), axis=1)

    b = data_generation.EmbderMap(2, [np.arange(4.9, 15.1, 0.1), np.arange(-5, 5, 0.1)])
    b.map_value_to_grid(np.array(15).reshape(1, -1))
    b.map_value_to_grid(np.array(sample_y).reshape(1, -1))

    em_y = tf.concat((em_y, tf.reshape(b.idxs[1], [1, 1])), axis=1)
    if num_steps > 1:
        model, x, y, em_y, num_steps = inference(model, x, y, em_y,  num_steps - 1, d=d, x_2=x_2,
                                           sample=sample, xx=xx, yy=yy, infer=infer, x0=x0, y0=y0, x1=x1, y1=y1)

    return model, x, y, em_y, num_steps


def main():
    pass


if __name__ == '__main__':
    main()
