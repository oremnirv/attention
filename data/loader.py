
import sys
sys.path.append("..")
import os
import numpy as np
from data import data_generation



def load_data(kernel='rbf', size=150000):

    folder = '/Users/omernivron/Downloads/GPT_' + kernel + '/data/'
    folder_content = os.listdir(folder)
    if len(folder_content) == 0:
        print("Directory is empty \n Generating data..")
        pad_pos_tr, pad_pos_te, pad_y_fren_tr, pad_y_fren_te, _, df_te = data_generation.data_generator_for_gp_mimick_gpt(
            size, ordered=False, same_x=True, kernel=kernel)
        np.save(folder + 'pad_pos_tr.npy', pad_pos_tr)
        np.save(folder + 'pad_pos_te.npy', pad_pos_te)
        np.save(folder + 'pad_y_fren_tr.npy', pad_y_fren_tr)
        np.save(folder + 'pad_y_fren_te.npy', pad_y_fren_te)

    data = [np.load(f) for f in folder_content ]

    return *data


def main():
	load_data(kernel = 'periodic')


if __name__ == '__main__':

    main()
