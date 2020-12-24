
import numpy as np

def load_data(): 
	x_tr = np.load('/Users/omernivron/Downloads/pad_pos_tr.npy')
	x_te = np.load('/Users/omernivron/Downloads/pad_pos_te.npy')
	y_tr = np.load('/Users/omernivron/Downloads/pad_y_fren_tr.npy')
	y_te = np.load('/Users/omernivron/Downloads/pad_y_fren_te.npy')

	return x_tr, x_te, y_tr, y_te



def main():
	load_data()


if __name__ == '__main__':
	main()