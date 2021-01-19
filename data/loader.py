
import sys
sys.path.append("..")
import os
import numpy as np
from data import data_generation



def load_data(kernel='rbf', size=150000, rewrite = 'False', diff_x = False, noise = False, d = False, ordered = False):

    folder = '/Users/omernivron/Downloads/GPT_' + kernel + '/data/'
    folder_content = os.listdir(folder)
    if (len(folder_content) == 0) or (rewrite  == 'True'):
        print("Directory is empty \n Generating data..")
        kernel1 = kernel.split('_')[0]
        if d:
            bias = kernel.split('_')[1]
            print(bias)
            pad_pos_tr, pad_pos_te, pad_y_fren_tr, pad_y_fren_te, df_tr, df_te, em_tr, em_te, em_tr_2, em_te_2 = data_generation.data_gen2d(int(size), bias = bias, kernel = kernel1, noise = noise)
            np.save(folder + 'em_tr_2.npy', em_tr_2)
            np.save(folder + 'em_te_2.npy', em_te_2)
        else:
            pad_pos_tr, pad_pos_te, pad_y_fren_tr, pad_y_fren_te, _, df_te, em_tr, em_te = data_generation.data_generator_for_gp_mimick_gpt(
            int(size), ordered=ordered, diff_x=diff_x, kernel=kernel1, noise = noise)
        

        np.save(folder + 'pad_pos_tr.npy', pad_pos_tr)
        np.save(folder + 'pad_pos_te.npy', pad_pos_te)
        np.save(folder + 'pad_y_fren_tr.npy', pad_y_fren_tr)
        np.save(folder + 'pad_y_fren_te.npy', pad_y_fren_te)
        np.save(folder + 'em_tr.npy', em_tr)
        np.save(folder + 'em_te.npy', em_te)


    folder_content = os.listdir(folder)
    print(folder_content)
    data = [np.load(folder + f) for f in folder_content if f.split('_')[-1] != 'Store']

    return data


def main():
	load_data(kernel = 'periodic')


if __name__ == '__main__':

    main()
