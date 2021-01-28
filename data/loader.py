import sys
sys.path.append("..")
import os
import numpy as np
from data import data_generation


def load_data(kernel='rbf', size=150000, rewrite='False', diff_x=False, noise=False, d=False, ordered=False):
    folder = '/Users/omernivron/Downloads/GPT_' + kernel + '/data/'
    folder_content = os.listdir(folder)
    if (len(folder_content) == 0) or (rewrite == 'True'):
        print("Directory is empty \n Generating data..")
        kernel1 = kernel.split('_')[0]
        if d:
            bias = kernel.split('_')[1]
            print(bias)
            x_tr, x_te, y_tr, y_te, df_tr, df_te, em_tr, em_te, em_tr_2, em_te_2 = data_generation.data_gen2d(int(size),
                                                                                                              bias=bias,
                                                                                                              kernel=kernel1,
                                                                                                              noise=noise)
            np.save(folder + 'em_tr_2.npy', em_tr_2)
            np.save(folder + 'em_te_2.npy', em_te_2)
        else:
            x_tr, x_te, y_tr, y_te, _, df_te, em_tr, em_te = data_generation.data_gen(
                int(size), ordered=ordered, diff_x=diff_x, kernel=kernel1, noise=noise)

        np.save(folder + 'x_tr.npy', x_tr)
        np.save(folder + 'x_te.npy', x_te)
        np.save(folder + 'y_tr.npy', y_tr)
        np.save(folder + 'y_te.npy', y_te)
        np.save(folder + 'em_tr.npy', em_tr)
        np.save(folder + 'em_te.npy', em_te)

    folder_content = os.listdir(folder)
    print(folder_content)
    data = [np.load(folder + f) for f in folder_content if f.split('_')[-1] != 'Store']

    return data


def main():
    load_data(kernel='periodic')


if __name__ == '__main__':
    main()
