import sys

sys.path.append("..")
import os
import numpy as np
from data import data_generation


def load_data(kernel='rbf', size=150000, rewrite='False', diff_x=False, noise=False, d=False, ordered=False, p=0.5):
    """

    """
    folder = os.path.expanduser('~/Downloads/GPT_' + kernel + '/data/')
    folder_content = os.listdir(folder)
    if (len(folder_content) == 0) or (rewrite == 'True'):
        print("Directory is empty \n Generating data..")
        kernel1 = kernel.split('_')[0]
        if d:
            bias = kernel.split('_')[1]
            x_tr, x_te, y_tr, y_te, df_tr, df_te, em, em_y = data_generation.data_gen2d(int(size),
                                                                                                              bias=bias,
                                                                                                              kernel=kernel1,
                                                                                                              noise=noise)
            np.save(folder + 'em_tr_2.npy', em[2])
            np.save(folder + 'em_te_2.npy', em[3])

        else:
            x_tr, x_te, y_tr, y_te, _, df_te, em, em_y = data_generation.data_gen(
                int(size), ordered=ordered, diff_x=diff_x, kernel=kernel1, noise=noise, p_order=p)

        np.save(folder + 'x_tr.npy', x_tr)
        np.save(folder + 'x_te.npy', x_te)
        np.save(folder + 'y_tr.npy', y_tr)
        np.save(folder + 'y_te.npy', y_te)
        np.save(folder + 'em_tr.npy', em[0])
        np.save(folder + 'em_te.npy', em[1])
        np.save(folder + 'em_y_tr.npy', em_y[0])
        np.save(folder + 'em_y_te.npy', em_y[1])

    folder_content = np.sort(os.listdir(folder))
    print(folder_content)
    data = [np.load(folder + f) for f in folder_content if f.split('_')[-1] != 'Store']

    return data


def main():
    load_data(kernel='periodic')


if __name__ == '__main__':
    main()
