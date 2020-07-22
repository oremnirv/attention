import numpy as np

def pick_diff_cols_from_each_row(arr, cols):
    '''

    An example would be when we have an array with 40000 rows and 59 columns
    and we want to pick randomly from the first 5000 rows a different set of columns 
	
	cols: array([[24, 16,  7, 11,  6, 46],
       			[29, 12, 18, 29, 17, 54],
       			[34,  9, 14, 36, 34, 52],

    rows would then be array([[   0,    0,    0,    0,    0,    0],
       						  [   1,    1,    1,    1,    1,    1],
       					      [   2,    2,    2,    2,    2,    2],


    i.e. pick row 0 columns 24, 26, 7, 11, 6 , 46
    ....

	--------------------------
    

    --------------------------
    Returns:
    
    '''
    n = arr.shape[0]
    m = cols.shape[1]
    rows = np.repeat(np.arange(0, n, 1), m).reshape(n, m)
    arr = arr[rows, cols]
    return arr