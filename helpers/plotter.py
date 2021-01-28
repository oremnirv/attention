import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf;
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF

from data import loader
from helpers import metrics, helpers
from inference import infer
from model import experimental_model, experimental2d_model, grapher

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
    fig.suptitle('Four examples from the dataset generated by a 2D GP \n with RBF kernel with σ = 1', fontsize=16)
    for i, idd in enumerate(idxs):
        row = i // 2;
        col = i % 2
        f_x = x[idd, :][np.where(em_2[idd, :] == 1)[0]]
        f_y = y[idd, :][np.where(em_2[idd, :] == 1)[0]]
        axs[row, col].plot(np.sort(f_x), f_y[np.argsort(f_x)])
        s_x = x[idd, :][np.where(em_2[idd, :] == 0)[0]]
        s_y = y[idd, :][np.where(em_2[idd, :] == 0)[0]]
        axs[row, col].plot(np.sort(s_x), s_y[np.argsort(s_x)])
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def infer_plot(model, em, x, y, num_steps, samples=10, mean=True, context_p=50, order=False, axs=None, ins=False,
               consec=True):
    if axs:
        pass
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        custom_xlim = (4, 16)
    # custom_ylim = (-3, 3)
    # plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)

    maxi = num_steps + context_p
    sorted_idx = np.argsort(x)

    if order:
        if consec:
            x = x[sorted_idx]
            y = y[sorted_idx]
            em = em[sorted_idx]
            sorted_idx = np.argsort(x)
        else:
            non_cosec_idx = np.concatenate(
                (np.sort(np.random.choice(sorted_idx[:120], context_p, replace=False)), sorted_idx[120:]))
            x = x[non_cosec_idx]
            y = y[non_cosec_idx]
            em = em[non_cosec_idx]
            sorted_idx = np.argsort(x)
            num_steps = min(len(non_cosec_idx) - context_p, num_steps)

    # true graph
    axs.plot(x[:maxi][sorted_idx], y[:maxi][sorted_idx], c='black', zorder=1, linewidth=3, label='obs. function')

    # context points:
    axs.scatter(x[:context_p],
                y[:context_p], c='red', label='context points')

    for i, inf in enumerate(range(samples)):
        _, _, tar_inf = infer.inference(model, em[:maxi].reshape(1, -1), y[:context_p].reshape(1, -1),
                                        num_steps=num_steps)
        mse_model = metrics.mse(y[context_p: maxi], tar_inf.numpy()[:, context_p: maxi])
        n = num_steps
        y_mean = np.repeat(np.mean(y), n).reshape(1, -1)
        print('sample # {}, r squared: {}'.format(i, 1 - (mse_model / metrics.mse(y[context_p: maxi], y_mean))))
        if i == 0:
            axs.plot(x[sorted_idx], tar_inf.numpy().reshape(-1)[sorted_idx], c='lightskyblue', label='samples')
        else:
            axs.plot(x[sorted_idx], tar_inf.numpy().reshape(-1)[sorted_idx], c='lightskyblue')

    if mean:
        _, _, tar_inf = infer.inference(model, em[:maxi].reshape(1, -1), y[:context_p].reshape(1, -1),
                                        num_steps=num_steps, sample=False)

        axs.plot(x[sorted_idx], tar_inf.numpy().reshape(-1)[sorted_idx], c='goldenrod', label='mean sample')

    if ins:
        return axs
    else:
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


def plot_subplot_training(params, x, x_te, y, y_te, pred_y, pred_y_te, tr_idx, te_idx, sorted_idx_tr, sorted_idx_te,
                          num_context):
    for row in range(params.shape[0]):
        if (row == 1):
            x = x_te
            y = y_te
            pred_y = pred_y_te
            tr_idx = te_idx
            sorted_idx_tr = sorted_idx_te
        for col in range(params.shape[1]):

            params[row, col].plot(x[tr_idx[col], sorted_idx_tr[col, :]],
                                  y[tr_idx[col], sorted_idx_tr[col, :]], c='black', label='obs. function')
            params[row, col].scatter(x[tr_idx[col], :num_context], y[tr_idx[col], :num_context],
                                     c='black', marker="o", zorder=1, s=25, label='context points')
            params[row, col].plot(x[tr_idx[col], sorted_idx_tr[col, :]], pred_y[col,
                                                                                sorted_idx_tr[col, :]],
                                  c='lightskyblue', label='reconstructed function')
            if (row == 0):
                params[row, col].set_title('Training ex. {}'.format(col + 1))
            else:
                params[row, col].set_title('Test ex. {}'.format(col + 1))
            if ((row == 0) and (col == 1)):
                params[row, col].legend()
    return params


def follow_training_plot2d(x_tr, y_tr, em_2_tr, pred,
                           x_te, y_te, em_2_te, pred_te, num_context=50):
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

    axs = plot_subplot_training2d(axs, x_tr, x_te, y_tr, y_te, samples_tr1, samples_tr2, samples_te1, samples_te2,
                                  idx_tr, idx_te, idx_f1, idx_f2, idx_f1_te, idx_f2_te, num_context)
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def plot_subplot_training2d(params, x, x_te, y, y_te, pred_y, pred_y_2, pred_y_te, pred_y_te_2, idx_tr, idx_te, idx_f1,
                            idx_f2, idx_f1_te, idx_f2_te, num_context):
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
                         y[idx_tr[0], np.array(idx_f1)[0]], c='black', label='obs. function')

        params[row].plot(x[idx_tr[0], np.array(idx_f2)[0]],
                         y[idx_tr[0], np.array(idx_f2)[0]], c='blue', label='obs. function II')

        params[row].scatter(x[idx_tr[0], :num_context], y[idx_tr[0], :num_context],
                            c='black', marker="o", zorder=1, s=25, label='context points')

        params[row].plot(x[idx_tr[0], np.array(idx_f1)[0]], pred_y.reshape(-1), c='gray',
                         label='reconstructed function')

        params[row].plot(x[idx_tr[0], np.array(idx_f2)[0]], pred_y_2.reshape(-1), c='lightskyblue',
                         label='reconstructed function II')

        if (row == 0):
            params[row].set_title('Training ex. I')
            leg = params[row].legend()
            bb = leg.get_bbox_to_anchor().inverse_transformed(params[row].transAxes)

            # Change to location of the legend. 
            xOffset = .5
            bb.x0 += xOffset
            bb.x1 += xOffset
            leg.set_bbox_to_anchor(bb, transform=params[row].transAxes)

        else:
            params[row].set_title('Test ex. I')

    return params


def concat_n_rearange(x, y, em, em_2, cond_arr, context_p, num_steps, series=1):
    cond_arr_pre = cond_arr[:context_p]
    cond_arr_pos = cond_arr[context_p:]
    cond = []
    cond.append(np.where([cond_arr_pre == series])[1])
    cond.append(np.where([cond_arr_pos == series])[1])
    cond.append(np.where(~(cond_arr_pre == series)))
    cond.append(np.where(~ (cond_arr_pos == series)))

    # create the two series that will be used as context points
    y_pre = y[:context_p];
    y_post = y[context_p:]
    tar_1 = np.concatenate((y_pre[cond[0]], y_post[cond[1]])).reshape(1, -1)
    tar_0_par = y_pre[cond[2]]
    tar_0 = np.concatenate((y_pre[cond[2]], y_post[cond[3]])).reshape(1, -1)
    # generate rearanged target for inference
    y_infer = np.concatenate((y_pre, y_post[cond[1]])).reshape(1, -1)
    maxi = num_steps + y_infer.shape[1]

    # generate rearanged input for infer
    em_pre = em[:context_p];
    em_post = em[context_p:]
    em2_pre = em_2[:context_p];
    em2_post = em_2[context_p:]

    em_infer = np.concatenate((em_pre, em_post[cond[1]]))
    em_infer = np.concatenate((em_infer, em_post[cond[3]]))
    em_infer = em_infer[:maxi]

    em2_infer = np.concatenate((em2_pre, em2_post[cond[1]]))
    em2_infer = np.concatenate((em2_infer, em2_post[cond[3]]))
    em2_infer = em2_infer[:maxi]

    # rearange x
    x_pre = x[:context_p];
    x_post = x[context_p:]
    x_1 = np.concatenate((x_pre[cond[0]], x_post[cond[1]]))
    x_0 = np.concatenate((x_pre[cond[2]], x_post[cond[3]]))
    x_0_part = x_pre[cond[2]]
    xx_0 = np.concatenate((x_pre, x_post[cond[1]]))
    xx_0 = np.concatenate((xx_0, x_post[cond[3]]))
    yy_0 = np.concatenate((y_pre, y_post[cond[1]]))
    yy_0 = np.concatenate((yy_0, y_post[cond[3]]))
    xx_0 = xx_0[:maxi]

    sorted_x_1 = np.argsort(x_1)
    sorted_x_0 = np.argsort(x_0)
    sorted_x_0_p = np.argsort(x_0_part)
    sorted_xx_0 = np.argsort(xx_0)
    sorted_x = np.argsort(x)
    sorted_x = sorted_x[np.where(sorted_x < maxi)]

    # sort 1's and 0's according to sorted x's
    sorted_em = em2_infer.reshape(-1)[sorted_xx_0.reshape(-1)]

    return sorted_xx_0[np.where(sorted_em == 0)], xx_0[sorted_xx_0[np.where(sorted_em == 0)]], x_0[
        sorted_x_0.reshape(-1)], tar_0[0][sorted_x_0.reshape(-1)], x_1[sorted_x_1.reshape(-1)], tar_1[0][
               sorted_x_1.reshape(-1)], x_0_part[sorted_x_0_p.reshape(-1)], tar_0_par[
               sorted_x_0_p.reshape(-1)], em_infer, em2_infer, y_infer, yy_0


def infer_plot2D(decoder, x, y, em, em_2, num_steps=100, samples=10, order=True, context_p=50, mean=True, consec=False,
                 axs=None, ins=False):
    if axs:
        pass
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        custom_xlim = (4, 16)

    if order:
        if consec:
            sorted_idx = np.argsort(x).reshape(-1)
            x = x.reshape(-1)[sorted_idx]
            y = y.reshape(-1)[sorted_idx]
            em = em.reshape(-1)[sorted_idx]
            em_2 = em_2.reshape(-1)[sorted_idx]
        else:
            sorted_idx = np.argsort(x).reshape(-1)
            non_cosec_idx = np.concatenate(
                (np.sort(np.random.choice(sorted_idx[:120], context_p, replace=False)), sorted_idx[120:])).reshape(-1)
            x = x.reshape(-1)[non_cosec_idx]
            y = y.reshape(-1)[non_cosec_idx]
            em = em.reshape(-1)[non_cosec_idx]
            em_2 = em_2.reshape(-1)[non_cosec_idx]
            num_steps = min(len(non_cosec_idx) - context_p, num_steps)

    sorted_infer, x_infer, x_0, tar_0, x_1, tar_1, x_0_part, tar_0_part, em_infer, em2_infer, y_infer, yy_0 = concat_n_rearange(
        x.reshape(-1), y.reshape(-1), em.reshape(-1), em_2.reshape(-1), em_2.reshape(-1), context_p * 2,
        num_steps=num_steps)
    # if not num_steps:
    #     num_steps = 400 - y_infer.shape[1]
    axs.scatter(x_0_part, tar_0_part, c='red')
    axs.plot(x_0, tar_0, c='lightcoral')
    axs.plot(x_1, tar_1, c='black')
    for i, inf in enumerate(range(samples)):
        _, _, tar_inf = infer.inference(decoder, em_te=em_infer.reshape(1, -1), y=y_infer, num_steps=num_steps,
                                        sample=True, d=True, em_te_2=em2_infer.reshape(1, -1), series=1)
        axs.plot(x_infer, tar_inf.numpy().reshape(-1)[sorted_infer], c='lightskyblue')
        mse_model = metrics.mse(yy_0.reshape(-1)[sorted_infer].reshape(1, -1),
                                tar_inf.numpy().reshape(-1)[sorted_infer].reshape(1, -1))
        n = min(len(em_infer) - y_infer.shape[1], num_steps)
        y_mean = np.repeat(np.mean(yy_0.reshape(-1)[sorted_infer]), n).reshape(1, -1)
        print('sample # {}, r squared: {}'.format(i, 1 - (mse_model / metrics.mse(yy_0[-n:].reshape(1, -1), y_mean))))
    if mean:
        _, _, tar_inf = infer.inference(decoder, em_te=em_infer.reshape(1, -1), tar=y_infer, num_steps=num_steps,
                                        sample=False, d=True, em_te_2=em2_infer.reshape(1, -1), series=1)
        axs.plot(x_infer, tar_inf.numpy().reshape(-1)[sorted_infer], c='goldenrod')

    if ins:
        return axs
    else:
        plt.show()


def GP_compare_1D(x, y, kernel, noise=False, context_p=50, order=True, consec=True, axs=None, ins=False):
    if axs:
        pass
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        custom_xlim = (4, 16)

    sorted_idx = np.argsort(x)

    if order:
        if consec:
            x = x[sorted_idx]
            y = y[sorted_idx]
            sorted_idx = np.argsort(x)
        else:
            non_cosec_idx = np.concatenate(
                (np.sort(np.random.choice(sorted_idx[:120], context_p, replace=False)), sorted_idx[120:]))
            x = x[non_cosec_idx]
            y = y[non_cosec_idx]
            sorted_idx = np.argsort(x)

    if (kernel == 'rbf'):
        k = RBF(length_scale=1)
        print(k)

    else:
        k = ExpSineSquared(length_scale=1, periodicity=1, length_scale_bounds=(1, 10.0), periodicity_bounds=(1, 10.0))
        print(k)

    if noise:
        k = k + WhiteKernel(noise_level=0.5)
        print(k)

    model = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=1, alpha=0.0001)
    model.fit(x[:context_p].reshape(-1, 1), y[:context_p].reshape(-1, 1))
    μ, σ = model.predict(x[context_p:].reshape(-1, 1), return_std=True)
    μ = np.concatenate((y[:context_p].squeeze(), μ.squeeze()))
    σ = np.concatenate((np.zeros(context_p).squeeze(), σ.squeeze()))
    p_sort = np.argsort(x[context_p:])
    y_samples = model.sample_y(x[context_p:][p_sort].reshape(-1, 1), 10).squeeze()
    x_samples = np.tile(x[context_p:][p_sort].reshape(-1, 1), 10).reshape(len(x[context_p:]), -1)

    # print(y_samples.shape)
    # print(x_samples.shape)
    # print(kernel, σ)

    # axs.fill_between(x.reshape(-1)[sorted_idx], μ.squeeze()[sorted_idx] -2 * σ[sorted_idx], μ.squeeze()[sorted_idx] + 2 * σ[sorted_idx], alpha=.4, color = 'lightskyblue')
    axs.plot(x_samples, y_samples, color="lightskyblue")

    axs.plot(x.reshape(-1)[sorted_idx], μ.squeeze()[sorted_idx], color="goldenrod")

    axs.plot(x[sorted_idx], y[sorted_idx], color='black')

    axs.scatter(x[:context_p].reshape(-1, 1), y[:context_p].reshape(-1, 1), color='red')
    print(model.kernel_)
    if ins:
        return axs
    else:
        plt.show()


def GP_infer1D():
    GPT_files = glob.glob('/Users/omernivron/Downloads/GPT_*')
    GPT_files = [f for f in GPT_files if f.split('_')[-1] != '2D']
    fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)

    for row, big_ax in enumerate(big_axes, start=1):
        title = GPT_files[row - 1].split('GPT_')[-1].split('_')
        if len(title) > 1:
            big_ax.set_title("{} {} kernel".format(title[-2].upper(), title[-1]), fontsize=16)
        else:
            big_ax.set_title("{} kernel".format(title[-1].upper()), fontsize=16)
        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.7, wspace=0.4)
    custom_xlim = (4, 16)
    custom_ylim = (-5, 5)

    for i, f in enumerate(GPT_files):
        kernel = f.split('GPT_')[-1]
        if (kernel.split('_')[-1] == 'noise'):
            noise = True
        else:
            noise = False
        k = kernel.split('_')[0]
        idx = np.random.choice(range(30000), 1)
        data = loader.load_data(kernel, size=1, rewrite='False')
        for j, order in enumerate([True, False]):
            ax = fig.add_subplot(len(GPT_files), 2, i * 2 + j + 1)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            GP_compare_1D(x=data[1][idx, :].reshape(-1), y=data[-1][idx, :].reshape(-1), kernel=k, noise=noise,
                          order=order, axs=ax, ins=True)


def all_inference(consec=True):
    GPT_files = glob.glob('/Users/omernivron/Downloads/GPT_*')
    fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=5, ncols=1, sharey=True)

    for row, big_ax in enumerate(big_axes, start=1):
        title = GPT_files[row - 1].split('GPT_')[-1].split('_')
        if len(title) > 1:
            big_ax.set_title("{} {} kernel".format(title[-2].upper(), title[-1]), fontsize=16)
        else:
            big_ax.set_title("{} kernel".format(title[-1].upper()), fontsize=16)
        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.7, wspace=0.4)
    custom_xlim = (4, 16)
    custom_ylim = (-5, 5)

    for i, f in enumerate(GPT_files):
        d = False
        kernel = f.split('GPT_')[-1]
        idx = np.random.choice(range(30000), 1)
        data = loader.load_data(kernel, size=1, rewrite='False')
        for j, order in enumerate([True, False]):
            ax = fig.add_subplot(len(GPT_files), 2, i * 2 + j + 1)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            folder = f + '/ckpt/check_run_1'
            train_step, test_step, loss_object, train_loss, test_loss, m_tr, m_te = grapher.build_graph()
            ℯ = 512;
            l = [256, 256, 64, 32];
            heads = 32;
            context = 50;
            if f.split('_')[-1] == '2D':
                d = True
            ℯ, l1, _, l2, l3 = helpers.load_spec(folder, ℯ, l, context_p=context, d=d)
            if f.split('_')[-1] == '2D':
                decoder = experimental2d_model.Decoder(ℯ, l1, l2, l3, num_heads=heads)
            else:
                vocab = 400 if kernel == 'rbf' else 200
                decoder = experimental_model.Decoder(ℯ, l1, l2, l3, num_heads=heads, input_vocab_size=vocab);
            optimizer_c = tf.keras.optimizers.Adam(3e-4)
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_c, net=decoder)
            manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=3)
            ckpt.restore(manager.latest_checkpoint)
            if f.split('_')[-1] == '2D':
                print('context: ', context)
                infer_plot2D(decoder, data[2][idx, :], data[6][idx, :], data[3][idx, :], data[0][idx, :], samples=10,
                             num_steps=999, consec=consec, order=order, axs=ax, ins=True, context_p=context * 2)
            else:
                infer_plot(decoder, em=data[2][idx, :].reshape(-1), x=data[1][idx, :].reshape(-1),
                           y=data[-1][idx, :].reshape(-1), num_steps=150, samples=10, context_p=context, order=order,
                           axs=ax, ins=True, consec=consec)
                if (((i * 2 + j + 1) % 2) == 1):
                    leg = ax.legend()
