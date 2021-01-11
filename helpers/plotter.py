import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inference import infer
plt.style.use('ggplot')


def sample_plot_w_training(positions, targets, predictions, num_context=50, num_samples=1, title=''):
    real_x = positions
    real_y = targets
    samples = np.zeros((num_samples, len(real_x)))
    samples[0, :(num_context - 1)] = targets[:(num_context - 1)]
    samples[0, (num_context - 1):] = predictions[(num_context - 1):]
    sorted_arr = np.argsort(real_x)
    plt.plot(real_x[sorted_arr], real_y[sorted_arr], 'black')
    plt.scatter(real_x[:num_context], real_y[:num_context],
                c='black', marker="o", zorder=1, s=25)
    plt.plot(real_x[sorted_arr], samples[0, sorted_arr],
             c='lightskyblue', alpha=0.6)
    plt.title(title)
    plt.show()

# Show few graphs of how the data looks like


def plot_examples(x, y):
    idx = np.random.choice(np.arange(0, len(x)), 5, replace=False)
    for i in idx:
        sorted_idx = np.argsort(x[i, :])
        plt.plot(x[i, sorted_idx], y[i, sorted_idx])
    plt.title(
        'Five examples from the dataset generated by a GP \n with RBF kernel with σ = 1')


def plot_2d_examples(x, y, em_2):

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    custom_xlim = (4, 16)
    custom_ylim = (-8, 8)
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    idxs = np.random.choice(x.shape[0], 4)
    fig.suptitle('Four examples from the dataset generated by a 2D GP \n with RBF kernel with σ = 1', fontsize = 16)
    for i, idd in enumerate(idxs):
        row = i // 2; col = i % 2 
        f_x = x[idd, :][np.where(em_2[idd, :] == 1)[0]]
        f_y = y[idd, :][np.where(em_2[idd, :] == 1)[0]]
        axs[row, col].plot(np.sort(f_x), f_y[np.argsort(f_x)])
        s_x = x[idd, :][np.where(em_2[idd, :] == 0)[0]]
        s_y = y[idd, :][np.where(em_2[idd, :] == 0)[0]]
        axs[row, col].plot(np.sort(s_x), s_y[np.argsort(s_x)])
    for ax in axs.flat:
        ax.label_outer()


    plt.show()


def infer_plot(model, em_te, x_te, y_te, num_steps, sample_num, samples=10, mean=True, context_p=50):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    custom_xlim = (4, 16)
    custom_ylim = (-10, 10)
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)

    maxi = num_steps + context_p
    sorted_idx = np.array(pd.DataFrame(x_te[sample_num, :maxi]).apply(
        lambda x: np.argsort(x), axis=1))
    pos_inf = (x_te[sample_num, :]).reshape(-1)

    # true graph
    axs.plot(x_te[sample_num, :maxi][0][sorted_idx][0], y_te[sample_num,
                                                             :maxi][0][sorted_idx][0], c='black', zorder=1, linewidth=3)
    # context points:
    axs.scatter(x_te[sample_num, :context_p][0],
                y_te[sample_num, :context_p][0], c='red')

    for inf in range(samples):
        _, _, tar_inf = infer.inference(model, em_te[sample_num, :maxi].reshape(1, -1), 
            y_te[sample_num, :context_p].reshape(1, -1), num_steps=num_steps)
        axs.plot(pos_inf[sorted_idx].reshape(-1),
                 tar_inf.numpy().reshape(-1)[sorted_idx][0], c='lightskyblue')

    if mean:
        _, _, tar_inf = infer.inference(model, em_te[sample_num, :maxi].reshape(1, -1), 
            y_te[sample_num, :context_p].reshape(1, -1), num_steps=num_steps, sample=False)
        axs.plot(pos_inf[sorted_idx].reshape(-1), tar_inf.numpy().reshape(-1)[sorted_idx][0], c='goldenrod', linewidth=2)

    plt.show()


def choose_random_ex_n_sort(x, num_samples):

    idx = np.random.choice(np.arange(0, len(x)), num_samples, replace=False)
    samples = np.zeros((num_samples, x.shape[1]))
    sorted_idx_samples = pd.DataFrame(x[idx, :]).apply(
        lambda x: np.argsort(x), axis=1)
    # print(np.array(sorted_idx_samples)[0, :])
    return idx, samples, sorted_idx_samples


def assign_context_points_to_preds(idx, samples, y, pred, num_context):
    samples[:, :num_context] = y[idx, :num_context]
    samples[:, num_context:] = pred.numpy()[idx, (num_context - 1):]
    return samples


def follow_training_plot(x_tr, y_tr, pred,
                         x_te, y_te, pred_te, num_context=50, num_samples=2):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    custom_xlim = (4, 16)
    custom_ylim = (-4, 4)

# Setting the values for all axes.
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    idx_tr, samples_train, sorted_idx_tr = choose_random_ex_n_sort(x_tr, 2)
    idx_te, samples_test, sorted_idx_te = choose_random_ex_n_sort(x_te, 2)
    samples_tr = assign_context_points_to_preds(
        idx_tr, samples_train, y_tr, pred, num_context)
    samples_te = assign_context_points_to_preds(
        idx_te, samples_test, y_te, pred_te, num_context)
    axs = plot_subplot_training(axs, x_tr, x_te, y_tr, y_te, samples_tr, samples_te, idx_tr, idx_te, np.array(
        sorted_idx_tr), np.array(sorted_idx_te), num_context)
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def plot_subplot_training(params, x, x_te, y, y_te, pred_y, pred_y_te, tr_idx, te_idx, sorted_idx_tr, sorted_idx_te, num_context):

    for row in range(params.shape[0]):
        if (row == 1):
            x = x_te
            y = y_te
            pred_y = pred_y_te
            tr_idx = te_idx
            sorted_idx_tr = sorted_idx_te
        for col in range(params.shape[1]):

            params[row, col].plot(x[tr_idx[col], sorted_idx_tr[col, :]],
                                  y[tr_idx[col], sorted_idx_tr[col, :]], c='black', label='true function')
            params[row, col].scatter(x[tr_idx[col], :num_context], y[tr_idx[col], :num_context],
                                     c='black', marker="o", zorder=1, s=25, label='context points')
            params[row, col].plot(x[tr_idx[col], sorted_idx_tr[col, :]], pred_y[col,
                                                                                sorted_idx_tr[col, :]], c='lightskyblue', label='reconstructed function')
            if (row == 0):
                params[row, col].set_title('Training ex. {}'.format(col + 1))
            else:
                params[row, col].set_title('Test ex. {}'.format(col + 1))
            if ((row == 0) and (col == 1)):
                params[row, col].legend()
    return params



def follow_training_plot2d(x_tr, y_tr, em_2_tr, pred,
                         x_te, y_te, em_2_te , pred_te, num_context=50):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    custom_xlim = (4, 16)
    custom_ylim = (-8, 8)

# Setting the values for all axes.
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    idx_tr, samples_train, sorted_idx_tr = choose_random_ex_n_sort(x_tr, 1)
    idx_te, samples_test, sorted_idx_te = choose_random_ex_n_sort(x_te, 1)
    samples_tr = assign_context_points_to_preds(
        idx_tr, samples_train, y_tr, pred, num_context)
    samples_te = assign_context_points_to_preds(
        idx_te, samples_test, y_te, pred_te, num_context)
    
    sorted_em = (em_2_tr[idx_tr]).reshape(-1)[np.array(sorted_idx_tr).reshape(-1)]
    sorted_em_te = (em_2_te[:500][idx_te]).reshape(-1)[np.array(sorted_idx_te).reshape(-1)]
    
    idx_f1 = sorted_idx_tr[np.where(sorted_em == 1)[0]]
    idx_f2 = sorted_idx_tr[np.where(sorted_em == 0)[0]]
    idx_f1_te = sorted_idx_te[np.where(sorted_em_te == 1)[0]]
    idx_f2_te = sorted_idx_te[np.where(sorted_em_te == 0)[0]]
    
    samples_tr1 = samples_tr.reshape(-1)[np.array(idx_f1).reshape(-1)]
    samples_tr2 = samples_tr.reshape(-1)[np.array(idx_f2).reshape(-1)]
    
    samples_te1 = samples_te.reshape(-1)[np.array(idx_f1_te).reshape(-1)]
    samples_te2 = samples_te.reshape(-1)[np.array(idx_f2_te).reshape(-1)]
    
    axs = plot_subplot_training2d(axs, x_tr, x_te, y_tr, y_te, samples_tr1, samples_tr2, samples_te1, samples_te2, idx_tr, idx_te, idx_f1, idx_f2, idx_f1_te, idx_f2_te, num_context)
    for ax in axs.flat:
        ax.label_outer()

    plt.show()



def plot_subplot_training2d(params, x, x_te, y, y_te, pred_y, pred_y_2, pred_y_te, pred_y_te_2, idx_tr, idx_te, idx_f1, idx_f2, idx_f1_te, idx_f2_te, num_context):

    for row in range(params.shape[0]):
        if (row == 1):
            x = x_te
            y = y_te
            pred_y = pred_y_te
            pred_y_2 = pred_y_te_2
            idx_tr = idx_te
            idx_f1 = idx_f1_te
            idx_f2 = idx_f2_te
            
        params[row].plot(x[idx_tr[0], np.array(idx_f1)[0]],
                                          y[idx_tr[0], np.array(idx_f1)[0]], c='black', label='true function')

        params[row].plot(x[idx_tr[0], np.array(idx_f2)[0]],
                                          y[idx_tr[0], np.array(idx_f2)[0]], c='blue', label='true function II')

        params[row].scatter(x[idx_tr[0], :50], y[idx_tr[0], :50],
                                             c='black', marker="o", zorder=1, s=25, label='context points')


        params[row].plot(x[idx_tr[0], np.array(idx_f1)[0]], pred_y.reshape(-1), c='gray', label='reconstructed function')

        params[row].plot(x[idx_tr[0], np.array(idx_f2)[0]], pred_y_2.reshape(-1), c='lightskyblue', label='reconstructed function')




        if (row == 0):
            params[row].set_title('Training ex. I')
            leg = params[row].legend()
            bb = leg.get_bbox_to_anchor().inverse_transformed(params[row].transAxes)

            # Change to location of the legend. 
            xOffset = .5
            bb.x0 += xOffset
            bb.x1 += xOffset
            leg.set_bbox_to_anchor(bb, transform = params[row].transAxes)
            
        else:
            params[row].set_title('Test ex. I')
            
    return params
