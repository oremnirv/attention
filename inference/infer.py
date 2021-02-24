from helpers import masks
import tensorflow as tf
import numpy as np


def evaluate(model, x, y, sample=True, d=False, x2=None, xx=None, yy=None, c_step=0, infer=False, x0=None, y0=None, x1=None, y1=None):
    """

    :param y1:
    :param x1:
    :param y0:
    :param x0:
    :param infer:
    :param c_step:
    :param yy:
    :param xx:
    :param model:
    :param x:
    :param y:
    :param sample:
    :param d:
    :param x2:
    :return:
    """
    combined_mask_x = masks.create_masks(x)
    if d:
        pred = model(x, x2, y, False, combined_mask_x[:, :-1, :-1], infer=infer, ix=xx, iy=yy, n=c_step, x0=x0, y0=y0, x1=x1, y1=y1)
        print('pred: ', pred)
        # tf.print(pred)

    else:
        pred = model(x, y, False, combined_mask_x[:, :-1, :-1])
    if sample:
        sample_y = np.random.normal(pred[-1, 0], np.exp(pred[-1, 1]))
    else:
        sample_y = pred[-1, 0]

    return pred[:, 0], pred[:, 1], sample_y


def inference(model, em_te, y, num_steps=1, sample=True, d=False, em_te_2=None, series=1, infer=False, xx=None, yy=None, x0=None, y0=None, x1=None, y1=None):
    """

    :param y1:
    :param x1:
    :param y0:
    :param x0:
    :param xx:
    :param yy:
    :param infer:
    :param model:
    :param em_te:
    :param y:
    :param num_steps:
    :param sample:
    :param d:
    :param em_te_2:
    :param series:
    :return:
    """
    n = y.shape[1]
    print('current step: ', n)
    if xx is not None:
        print('current_position to infer: ', xx[n])
        print('current target: ', yy[n])
    num_steps = em_te.shape[1] - n if num_steps == 999 else min(num_steps, em_te.shape[1] - n)
    temp_x = em_te[:, :(n + 1)]
    if d:
        temp_x2 = em_te_2[:, :(n + 1)]
        print('series: ', temp_x2[:, -1])
        print('current: ', temp_x[:, -1])
        pred, pred_log_sig, sample_y = evaluate(model, temp_x, y, d=True, x2=temp_x2, sample=sample, infer=infer, xx=xx, yy=yy, c_step=n, x0=x0, y0=y0, x1=x1, y1=y1)
    else:
        pred, pred_log_sig, sample_y = evaluate(model, temp_x, y, sample=sample)
    y = tf.concat((y, tf.reshape(sample_y, [1, 1])), axis=1)
    print('sample_y: ', sample_y)
    if num_steps > 1:
        model, em_te, y, num_steps = inference(model, em_te, y, num_steps - 1, d=d, em_te_2=em_te_2, series=series,
                                               sample=sample, xx=xx, yy=yy, infer=infer, x0=x0, y0=y0, x1=x1, y1=y1)

    return model, em_te, y, num_steps


def main():
    pass


if __name__ == '__main__':
    main()
