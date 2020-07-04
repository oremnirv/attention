import tensorflow as tf
import os


def mkdir(folder):
    '''
    '''
    if os.path.exists(folder):
        print('Already exists')
        pass
    else:
        os.mkdir(folder)
        print('New folder {}'.format(folder))


def tf_summaries(run, string, train_loss_r, test_loss_r):
    '''

    '''
    tf.summary.scalar("training loss run {}".format(run), train_loss_r
    , step=string)
    tf.summary.scalar("test loss run {}".format(run), test_loss_r,
     step=string)


def print_progress(epoch, batch_n, train_loss_r, test_loss_r):
    '''

    '''
    print('Epoch {} batch {} train Loss {:.4f} test Loss {:.4f}'.format(epoch, batch_n,
                                                                        train_loss_r, test_loss_r))


def quick_hist_counter(left, right, jump, arr):
    '''
    Create a histogram for the different values
    # observations in bin i, N = total # observations, δi = width of bin i
    density pi = ni / N * δi, where ni =
    -----------------
    Parameters:
    left (int):
    right (int):
    jump (float):
    arr (array float): 1-D array
    -----------------
    Returns:


    '''
    base = np.arange(left, right, jump)
    probs = np.array([len(arr[(low <= arr) &
                              (low + jump >= arr)]) for low in base])
    loc = np.where(probs == max(probs))
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    plt.bar(base, probs / sum(probs), color='navy')
    plt.axvline(base[loc], color='grey')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    return base, probs, loc

def hist_2d(left, right, jump, post):
    '''

    '''
    base = np.arange(left, right, jump)
    probs = np.zeros((len(base)**2, 3))
    for i, low_y in enumerate(base):
        for j, low_x in enumerate(base):
            probs[i, j] = np.array([low_x, low_y, sum(post[((low_x <= post[:, 0]) & (low_x + jump >= post[:, 0])) & ((low_y <= post[:, 1])  & (low_y + jump >= post[:, 1])), -1])])

    loc = np.where(probs[:, 2] == max(probs))
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    plt.scatter(probs[:, 0], probs[:, 1], color=probs[:, 2] / sum(probs[:, 2]))
    plt.xlabel('w_1')
    plt.ylabel('w_2')
