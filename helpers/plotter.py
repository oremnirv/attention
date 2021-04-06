import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF
from data import loader
from helpers import metrics, helpers
from inference import infer
from model import experimental_model, experimental2d_model, grapher
plt.style.use('ggplot')


def plot_examples(x, y):
    """
    Show few graphs of how the data looks like
    :param x: (np.array)
    :param y: (np.array)
    """
    idx = np.random.choice(np.arange(0, len(x)), 5, replace=False)
    for i in idx:
        sorted_idx = np.argsort(x[i, :])
        plt.plot(x[i, sorted_idx], y[i, sorted_idx])
    plt.title(
        'Five examples from the dataset generated by a GP \n with RBF kernel with σ = 1')


def plot_2d_examples(x, y, em_2):
    """
    Show few graphs of how the data looks like

    :param x: (np.array)
    :param y: (np.array)
    :param em_2: (np.array) with 0/1 values indicating the sequence member
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    custom_xlim = (4, 16)
    custom_ylim = (-8, 8)
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    idxs = np.random.choice(x.shape[0], 4)
    fig.suptitle('Four examples from the dataset generated by a 2D GP \n with RBF kernel with σ = 1', fontsize=16)
    for i, idd in enumerate(idxs):
        row = i // 2;
        col = i % 2
        # choose one example from sequence member 1
        f_x = x[idd, :][np.where(em_2[idd, :] == 1)[0]]
        f_y = y[idd, :][np.where(em_2[idd, :] == 1)[0]]
        axs[row, col].plot(np.sort(f_x), f_y[np.argsort(f_x)])
        # choose one example from sequence member 0
        s_x = x[idd, :][np.where(em_2[idd, :] == 0)[0]]
        s_y = y[idd, :][np.where(em_2[idd, :] == 0)[0]]
        axs[row, col].plot(np.sort(s_x), s_y[np.argsort(s_x)])
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def plot_subplot_training(params, x, x_te, y, y_te, pred_y, pred_y_te, tr_idx, te_idx, sorted_idx_tr, sorted_idx_te,
                          num_context):
    """

    :param params:
    :param x:
    :param x_te:
    :param y:
    :param y_te:
    :param pred_y:
    :param pred_y_te:
    :param tr_idx:
    :param te_idx:
    :param sorted_idx_tr:
    :param sorted_idx_te:
    :param num_context: (int) how many context points have been used
    :return:
    """
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


def create_condition_list(cond_arr, s=0):
    """
    :param cond_arr: (np.array) with 0/1 values
    :return:
    list with two np.array elements. The first indicating
    the indices of values from sequence member 0, the second the indices of values
    of sequence member 1
    """
    cond = []
    cond.append(np.where(cond_arr == s))
    cond.append(np.where((cond_arr != s)))
    return cond


def infer_plot2D(decoder, x, y, em, em_2, num_steps=100, samples=10, order=True, context_p=50, mean=True, consec=False,
                 axs=None, ins=False, s=0):
    """
    This is a wrapper function for making inferences for pairs of sequences
    and plotting the result, including the attention mechanism.

    :param decoder: tf.Keras.Model object. Already trained.
    :param x: (np.array)
    :param y: (np.array)
    :param em: (np.array) indices associated with x-vals -- to be used for embedding
    :param em_2: (np.array) 0/1 values indicating pair sequence member
    :param num_steps: (int) how many infernce steps to make
    :param samples: (int) how many inference trajectories to draw
    :param order: (bool)
    :param context_p: (int) has to be smaller than 120
    :param mean: (bool) If True infer just the mean without sampling from N(mu, sigma)
    :param consec: (bool) True when we want context points to be taken consecutively
    :param axs:
    :param ins:
    :return:
    """

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

    np.random.seed(443)
    cond = create_condition_list(em_2, s=s)
    x0 = x[cond[0]]; x1 = x[cond[1]]
    y0 = y[cond[0]]; y1 = y[cond[1]]
    em0 = em[cond[0]]; em1 = em[cond[1]]
    em_2_0 = em_2[cond[0]]; em_2_1 = em_2[cond[1]]
    print('em_2_0: ', em_2_0)
    print('em_2_1: ', em_2_1)

    y_infer = np.concatenate((y0, y1[:context_p])).reshape(1, -1)
    s_step = y_infer.shape[1]
    yy = np.concatenate((y0, y1)).reshape(1, -1)
    x_infer = np.concatenate((x0, x1)).reshape(1, -1)
    m_step = x_infer.shape[1] if num_steps == 999 else y_infer.shape[1] + num_steps
    em_infer = np.concatenate((em0, em1)).reshape(1, -1)
    em2_infer = np.concatenate((em_2_0, em_2_1)).reshape(1, -1)

    axs.scatter(x1[:context_p], y1[:context_p], c='red')
    print('last_x: ', x1[context_p-1])
    axs.plot(x0, y0, c='lightcoral', label = 's={}'.format(s))
    axs.plot(x1, y1, c='black', label = 's={}'.format(1- s))
    for i, inf in enumerate(range(samples)):
        _, _, y_inf, _ = infer.inference(decoder, x=em_infer, y=y_infer, num_steps=num_steps,
                                      sample=True, d=True, x_2=em2_infer, infer=True, xx = x_infer.reshape(-1), yy=yy.reshape(-1), x0=x0, y0=y0, x1=x1, y1=y1)

        axs.scatter(x_infer.reshape(-1)[s_step:m_step], y_inf.numpy().reshape(-1)[s_step:], c='lightskyblue')
        print('x_infer: ', x_infer.reshape(-1)[s_step])

    if mean:
        _, _, y_inf_m, _ = infer.inference(decoder, x=em_infer, y=y_infer, num_steps=num_steps,
                                      sample=False, d=True, x_2=em2_infer, infer=False)

        axs.scatter(x_infer.reshape(-1)[s_step:m_step], y_inf_m.numpy().reshape(-1)[s_step:], c='goldenrod')

    if ins:
        return axs
    else:
        axs.legend()
        plt.show()
        return x, y, x1, y1, x_infer, em2_infer, y_inf_m.numpy().reshape(-1),  x0[:context_p], y0[:context_p], x0, y0


def concat_context_to_infer(df, cond, context_p):
    """

    :param df: (np.array)
    :param cond: list of four np.arrays of indices. The first consists of indices from context points
    that are part of pair sequence member s (s is chosen in rearange_tr_2d func), second indicates indices
    of sequence member s that are not context points, third is the same like one but for the second sequence
    and fourth is the same as the second but for the second sequence member.
    :param context_p: (int) how many context points to use
    :return:
    reordered np.array

    To see an example, Run (see also main below):
    df =np.random.normal(0, 1, 15)
    em = np.random.choice([0, 1], 15)
    context_p = 4; s=1
    em_pre = em[:context_p]
    em_pos = em[context_p:]
    cond = [np.where(em_pre == s), np.where(em_pos == s), np.where(~(em_pre == s)),
        np.where(~(em_pos == s))]
    concat_context_to_infer(df, cond, context_p)
    """
    df_pre = df[:context_p]  # this includes context_p points, some from series 0 and some from series 1
    df_post = df[context_p:]  # this includes all points that were not picked as context
    # print('df_post: ', df_post)
    df_infer = np.concatenate((df_pre, df_post[cond[1]]))  # this includes context points and all the rest of series 0/1
    df_infer = np.concatenate((df_infer, df_post[cond[3]]))  # this completes the rest of the series to infer
    return df_infer

# def GP_compare_1D(x, y, kernel, noise=False, context_p=50, order=True, consec=True, axs=None, ins=False):
#     if axs:
#         pass
#     else:
#         fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#         custom_xlim = (4, 16)
#
#     sorted_idx = np.argsort(x)
#
#     if order:
#         if consec:
#             x = x[sorted_idx]
#             y = y[sorted_idx]
#             sorted_idx = np.argsort(x)
#         else:
#             non_cosec_idx = np.concatenate(
#                 (np.sort(np.random.choice(sorted_idx[:120], context_p, replace=False)), sorted_idx[120:]))
#             x = x[non_cosec_idx]
#             y = y[non_cosec_idx]
#             sorted_idx = np.argsort(x)
#
#     if (kernel == 'rbf'):
#         k = RBF(length_scale=1)
#
#     else:
#         k = ExpSineSquared(length_scale=1, periodicity=1, length_scale_bounds=(1, 10.0), periodicity_bounds=(1, 10.0))
#
#     if noise:
#         k = k + WhiteKernel(noise_level=0.5)
#
#     model = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=1, alpha=0.0001)
#     model.fit(x[:context_p].reshape(-1, 1), y[:context_p].reshape(-1, 1))
#     μ, σ = model.predict(x[context_p:].reshape(-1, 1), return_std=True)
#     μ = np.concatenate((y[:context_p].squeeze(), μ.squeeze()))
#     σ = np.concatenate((np.zeros(context_p).squeeze(), σ.squeeze()))
#     p_sort = np.argsort(x[context_p:])
#     y_samples = model.sample_y(x[context_p:][p_sort].reshape(-1, 1), 10).squeeze()
#     x_samples = np.tile(x[context_p:][p_sort].reshape(-1, 1), 10).reshape(len(x[context_p:]), -1)
#     axs.plot(x_samples, y_samples, color="lightskyblue")
#     axs.plot(x.reshape(-1)[sorted_idx], μ.squeeze()[sorted_idx], color="goldenrod")
#     axs.plot(x[sorted_idx], y[sorted_idx], color='black')
#     axs.scatter(x[:context_p].reshape(-1, 1), y[:context_p].reshape(-1, 1), color='red')
#     if ins:
#         return axs
#     else:
#         plt.show()

#
# def GP_infer1D():
#     GPT_files = glob.glob('/Users/omernivron/Downloads/GPT_*')
#     GPT_files = [f for f in GPT_files if f.split('_')[-1] != '2D']
#     fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)
#
#     for row, big_ax in enumerate(big_axes, start=1):
#         title = GPT_files[row - 1].split('GPT_')[-1].split('_')
#         if len(title) > 1:
#             big_ax.set_title("{} {} kernel".format(title[-2].upper(), title[-1]), fontsize=16)
#         else:
#             big_ax.set_title("{} kernel".format(title[-1].upper()), fontsize=16)
#         # Turn off axis lines and ticks of the big subplot
#         # obs alpha is 0 in RGBA string!
#         big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
#         # removes the white frame
#         big_ax._frameon = False
#
#     plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.7, wspace=0.4)
#     custom_xlim = (4, 16)
#     custom_ylim = (-5, 5)
#
#     for i, f in enumerate(GPT_files):
#         kernel = f.split('GPT_')[-1]
#         if (kernel.split('_')[-1] == 'noise'):
#             noise = True
#         else:
#             noise = False
#         k = kernel.split('_')[0]
#         idx = np.random.choice(range(30000), 1)
#         data = loader.load_data(kernel, size=1, rewrite='False')
#         for j, order in enumerate([True, False]):
#             ax = fig.add_subplot(len(GPT_files), 2, i * 2 + j + 1)
#             plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
#             GP_compare_1D(x=data[1][idx, :].reshape(-1), y=data[-1][idx, :].reshape(-1), kernel=k, noise=noise,
#                           order=order, axs=ax, ins=True)

#
# def all_inference(consec=True):
#     GPT_files = glob.glob('/Users/omernivron/Downloads/GPT_*')
#     fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=5, ncols=1, sharey=True)
#
#     for row, big_ax in enumerate(big_axes, start=1):
#         title = GPT_files[row - 1].split('GPT_')[-1].split('_')
#         if len(title) > 1:
#             big_ax.set_title("{} {} kernel".format(title[-2].upper(), title[-1]), fontsize=16)
#         else:
#             big_ax.set_title("{} kernel".format(title[-1].upper()), fontsize=16)
#         # Turn off axis lines and ticks of the big subplot
#         # obs alpha is 0 in RGBA string!
#         big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
#         # removes the white frame
#         big_ax._frameon = False
#
#     plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.7, wspace=0.4)
#     custom_xlim = (4, 16)
#     custom_ylim = (-5, 5)
#
#     for i, f in enumerate(GPT_files):
#         d = False
#         kernel = f.split('GPT_')[-1]
#         idx = np.random.choice(range(30000), 1)
#         data = loader.load_data(kernel, size=1, rewrite='False')
#         for j, order in enumerate([True, False]):
#             ax = fig.add_subplot(len(GPT_files), 2, i * 2 + j + 1)
#             plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
#             folder = f + '/ckpt/check_run_1'
#             train_step, test_step, loss_object, train_loss, test_loss, m_tr, m_te = grapher.build_graph()
#             ℯ = 512;
#             l = [256, 256, 64, 32];
#             heads = 32;
#             context = 50;
#             if f.split('_')[-1] == '2D':
#                 d = True
#             ℯ, l1, _, l2, l3 = helpers.load_spec(folder, ℯ, l, context_p=context, d=d)
#             if f.split('_')[-1] == '2D':
#                 decoder = experimental2d_model.Decoder(ℯ, l1, l2, l3, num_heads=heads)
#             else:
#                 vocab = 400 if kernel == 'rbf' else 200
#                 decoder = experimental_model.Decoder(ℯ, l1, l2, l3, num_heads=heads, input_vocab_size=vocab);
#             optimizer_c = tf.keras.optimizers.Adam(3e-4)
#             ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_c, net=decoder)
#             manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=3)
#             ckpt.restore(manager.latest_checkpoint)
#             if f.split('_')[-1] == '2D':
#                 print('context: ', context)
#                 infer_plot2D(decoder, data[2][idx, :], data[6][idx, :], data[3][idx, :], data[0][idx, :], samples=10,
#                              num_steps=999, consec=consec, order=order, axs=ax, ins=True, context_p=context * 2)
#             else:
#                 infer_plot(decoder, em=data[2][idx, :].reshape(-1), x=data[1][idx, :].reshape(-1),
#                            y=data[-1][idx, :].reshape(-1), num_steps=150, samples=10, context_p=context, order=order,
#                            axs=ax, ins=True, consec=consec)
#                 if (((i * 2 + j + 1) % 2) == 1):
#                     leg = ax.legend()


def plot_subplot_training2d(params, x, x_te, y, y_te, pred_y, pred_y_2, pred_y_te, pred_y_te_2, idx_tr, idx_te, idx_f1,
                            idx_f2, idx_f1_te, idx_f2_te, num_context):
    """

    :param params:
    :param x:
    :param x_te:
    :param y:
    :param y_te:
    :param pred_y:
    :param pred_y_2:
    :param pred_y_te:
    :param pred_y_te_2:
    :param idx_tr:
    :param idx_te:
    :param idx_f1:
    :param idx_f2:
    :param idx_f1_te:
    :param idx_f2_te:
    :param num_context:
    :return:
    """
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


def infer_plot(model, em, em_y,  x, y, num_steps, samples=10, mean=True, context_p=50, order=False, axs=None, ins=False,
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
    samples_arr = np.zeros((samples, 2))

    if order:
        if consec:
            x = x[sorted_idx]
            y = y[sorted_idx]
            em = em[sorted_idx]
            em_y = em_y[sorted_idx]
            sorted_idx = np.argsort(x)
        else:
            non_cosec_idx = np.concatenate(
                (np.sort(np.random.choice(sorted_idx[:120], context_p, replace=False)), sorted_idx[120:]))
            x = x[non_cosec_idx]
            y = y[non_cosec_idx]
            em = em[non_cosec_idx]
            em_y = em_y[non_cosec_idx]
            sorted_idx = np.argsort(x)
            num_steps = min(len(non_cosec_idx) - context_p, num_steps)

    # true graph
    axs.plot(x[:maxi][sorted_idx], y[:maxi][sorted_idx], c='black', zorder=1, linewidth=3, label='obs. function')

    # context points:
    axs.scatter(x[:context_p],
                y[:context_p], c='red', label='context points')

    for i, inf in enumerate(range(samples)):
        _, _, tar_inf, _,  _ = infer.inference(model, em[:maxi].reshape(1, -1),  y[:context_p].reshape(1, -1),  em_y[:context_p].reshape(1, -1),
                                        num_steps=num_steps, sample=True)
        # mse_model = metrics.mse(y[context_p: maxi], tar_inf.numpy()[:, context_p: maxi])
        # n = num_steps
        # y_mean = np.repeat(np.mean(y), n).reshape(1, -1)
        # print('sample # {}, r squared: {}'.format(i, 1 - (mse_model / metrics.mse(y[context_p: maxi], y_mean))))

        samples_arr[i, :] = tar_inf[-2:]

        if i == 0:
            axs.plot(x[sorted_idx], tar_inf.numpy().reshape(-1)[sorted_idx], c='lightskyblue', label='samples')
        else:
            axs.plot(x[sorted_idx], tar_inf.numpy().reshape(-1)[sorted_idx], c='lightskyblue')

    if mean:
        _, _, tar_inf, _, _ = infer.inference(model, em[:maxi].reshape(1, -1), y[:context_p].reshape(1, -1), em_y[:context_p].reshape(1, -1),
                                        num_steps=num_steps, sample=False)

        axs.plot(x[sorted_idx], tar_inf.numpy().reshape(-1)[sorted_idx], c='goldenrod', label='mean sample')

    if ins:
        return axs
    else:
        plt.show()
        return samples_arr


def follow_training_plot(x_tr, y_tr, pred,
                         x_te, y_te, pred_te, num_context=50, num_samples=2):
    """

    :param x_tr:
    :param y_tr:
    :param pred:
    :param x_te:
    :param y_te:
    :param pred_te:
    :param num_context:
    :param num_samples:
    :return:
    """
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

#
# def concat_n_rearange(x, y, em, em_2, context_p, num_steps, series=1):
#     """
#
#     :param x:
#     :param y:
#     :param em:
#     :param em_2:
#     :param context_p:
#     :param num_steps:
#     :param series:
#     :return:
#     """
#     cond = create_condition_list(em_2, context_p)
#     y1, y0 = get_series_separately(y, cond, context_p)
#     y0_p = get_context_points(y, cond, context_p)
#     x0_p = get_context_points(x, cond, context_p)
#     y_infer = y_infer_constructor(y, cond, context_p)
#     em_infer = concat_context_to_infer(em, cond, context_p)
#     em2_infer = concat_context_to_infer(em_2, cond, context_p)
#     yy = concat_context_to_infer(y, cond, context_p)
#     xx = concat_context_to_infer(x, cond, context_p)
#     x1, x0 = get_series_separately(x, cond, context_p)
#
#     maxi = num_steps + y_infer.shape[1]
#     em_infer = em_infer[:maxi]
#     em2_infer = em2_infer[:maxi]
#     xx = xx[:maxi]
#     s_x1, s_x0, s_x0p, s_xx = arg_sorter([x1, x0, x0_p, xx])
#     # sort 1's and 0's according to sorted x's
#     s_vals_em2 = em2_infer[s_xx]
#     series_cond = np.where(s_vals_em2 == series)
#     s_idx_xx_ser0 = s_xx[series_cond]
#     s_vals_xx_ser0 = xx[s_idx_xx_ser0]
#     s_vals_x0 = x0[s_x0]
#     s_vals_y0 = y0[s_x0]
#     s_vals_x1 = x1[s_x1]
#     s_vals_y1 = y1[s_x1]
#     s_context_x = x0_p[s_x0p]
#     s_context_y = y0_p[s_x0p]
#
#     return s_idx_xx_ser0, s_vals_xx_ser0, s_vals_x0 \
#         , s_vals_y0, s_vals_x1, s_vals_y1 \
#         , s_context_x, s_context_y, \
#            em_infer.reshape(1, -1), em2_infer.reshape(1, -1), y_infer, yy, xx, x1, y1, x0_p, y0_p, x0, y0
#
#
# def y_infer_constructor(df, cond, context_p):
#     """
#
#     :param df:
#     :param cond:
#     :return:
#     """
#     df_pre = df[:context_p]
#     df_post = df[context_p:]
#     df_infer = np.concatenate((df_pre, df_post[cond[1]])).reshape(1, -1)
#     return df_infer
#
#
# def arg_sorter(*l):
#     """
#
#     :param l:
#     :return:
#     """
#     s_list = []
#     for idx, element in enumerate(*l):
#         s_list.append(np.argsort(element))
#     return s_list

def choose_random_ex_n_sort(x, num_samples):
    """

    :param x:
    :param num_samples:
    :return:
    """
    idx = np.random.choice(np.arange(0, len(x)), num_samples, replace=False)
    samples = np.zeros((num_samples, x.shape[1]))
    sorted_idx_samples = pd.DataFrame(x[idx, :]).apply(
        lambda x: np.argsort(x), axis=1)
    return idx, samples, sorted_idx_samples

def assign_context_points_to_preds(idx, samples, y, pred, num_context):
    """
    This function is used to ensure that context points
    receive the observed y-vals and not predictions.

    :param idx: (int)
    :param samples:
    :param y: (np.array)
    :param pred: (tf.tensor)
    :param num_context: (int) how many context points to use.
    :return:
    (np.array)
    """
    samples[:, :num_context] = y[idx, :num_context]
    samples[:, num_context:] = pred.numpy()[idx, (num_context - 1):]  # the prediction at t is assocaited with y_(t+1)
    return samples


# def get_series_separately(df, cond, context_p):
#     """
#
#     :param df:
#     :param cond:
#     :param context_p:
#     :return:
#     """
#     df_pre = df[:context_p]
#     df_post = df[context_p:]
#     df_1 = np.concatenate((df_pre[cond[0]], df_post[cond[1]]))
#     df_0 = np.concatenate((df_pre[cond[2]], df_post[cond[3]]))
#     return df_1, df_0

#
# def get_context_points(df, cond, context_p):
#     """
#
#     :param df:
#     :param cond:
#     :return:
#     """
#     df_pre = df[:context_p];
#     context_points = df_pre[cond[2]]
#     return context_points

def follow_training_plot2d(x_tr, y_tr, em_2_tr, pred,
                           x_te, y_te, em_2_te, pred_te, num_context=50):
    """
    :param x_tr: (np.array)
    :param y_tr: (np.array)
    :param em_2_tr: (np.array)
    :param pred: (tf.tensor)
    :param x_te: (np.array)
    :param y_te: (np.array)
    :param em_2_te: (np.array)
    :param pred_te: (tf.tensor)
    :param num_context: (int)
    """
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
    sorted_em_te = (em_2_te[idx_te]).reshape(-1)[np.array(sorted_idx_te).reshape(-1)]

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


def main():
    df = np.random.normal(0, 1, 15)
    print('df: ', df)
    em = np.random.choice([0, 1], 15)
    print('em: ', em)
    context_p = 4;
    s = 1
    em_pre = em[:context_p]
    em_pos = em[context_p:]
    print('condition list: ')
    cond = [np.where(em_pre == s), np.where(em_pos == s), np.where(~(em_pre == s)),
            np.where(~(em_pos == s))]
    print(cond)
    print('output: ')
    print(concat_context_to_infer(df, cond, context_p))



if __name__ == '__main__':
    main()