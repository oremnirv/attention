from helpers import masks
import tensorflow as tf
import numpy as np


def evaluate(model, x, y, sample=True, d=False, x2=None):
    """
    Run a forward pass of the network
    ------------------
    Parameters:
    model: trained instace of GPT decoder class
    x: xition tensor with at least len(y) + 1 values
    y: ygets tensor
    x_mask: xition mask tensor to hide unseen xitions from current prediction
    ------------------
    Returns:
    pred (tf tensor float64): the prediction of the next location in the sequence
    pred_log_sig (tf tensor float64)

    """
    combined_mask_x = masks.create_masks(x)
    if d:
        pred = model(x, x2, y, False, combined_mask_x[:, 1:, :-1])
    else:
        pred = model(x, y, False, combined_mask_x[:, 1:, :-1])
    if sample:
        # print(np.exp(pred[-1, 1]))
        # print(pred[-1, 1])
        sample_y = np.random.normal(pred[-1, 0], np.exp(pred[-1, 1]))
        # print(np.exp(pred[-1, 1]))
    else:
        sample_y = pred[-1, 0]

    return pred[:, 0], pred[:, 1], sample_y


def inference(model, em_te, y, num_steps=1, sample=True, d=False, em_te_2=None, series=1):
    """
    how many steps to infer -- this could be used both for interpolation and extrapolation
    ------------------
    Parameters:
    x (2D np array): (n + num_steps) xitions
    y (2D np array): n ygets
    num_steps (int): how many inference steps are required
    ------------------
    Returns:
    pred (tf.tensor float64): the predictions for all timestamps up to n + num_steps
    pred_log_sig
    """
    n = y.shape[1]
    num_steps = em_te.shape[1] - n if num_steps == 999 else num_steps
    temp_x = em_te[:, :(n + 1)]
    if d:
        temp_x2 = em_te_2[:, :(n + 1)]
        pred, pred_log_sig, sample_y = evaluate(model, temp_x, y, d=True, x2=temp_x2, sample=sample)
    else:
        pred, pred_log_sig, sample_y = evaluate(model, temp_x, y, sample=sample)
    # print(sample_y)
    y = tf.concat((y, tf.reshape(sample_y, [1, 1])), axis=1)
    if num_steps > 1:
        model, em_te, y = inference(model, em_te, y, num_steps - 1, d=d, em_te_2=em_te_2, series=series,
                                    sample=sample)

    return model, em_te, y


def main():
    pass


if __name__ == '__main__':
    main()
