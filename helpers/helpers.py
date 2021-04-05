import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model import old_model_1D, experimental_model, experimental2d_model


def mkdir(folder):
    """
    Checks if a folder exist and create a new one if it does not.
    :param folder: (str) path to folder
    """
    if os.path.exists(folder):
        print('Already exists')
        pass
    else:
        os.mkdir(folder)
        print('New folder {}'.format(folder))


def tf_summaries(run, string, train_loss_r, test_loss_r, tr_metric, te_metric, weights, names):
    """
    These are summary stats used for logging things on TensorBoard

    """
    tf.summary.scalar("training loss run {}".format(run), train_loss_r, step=string)
    tf.summary.scalar("test loss run {}".format(run), test_loss_r, step=string)
    tf.summary.scalar('train metric', tr_metric, step=string)
    tf.summary.scalar('test metric', te_metric, step=string)
    for idx, var in enumerate(weights):
        tf.summary.histogram(names[idx].numpy().decode('utf-8'), var, step=string)


def print_progress(epoch, batch_n, train_loss_r, test_loss_r, tr_metric, te_metric):
    """

    """
    print(
        'Epoch {} batch {} train Loss {:.4f} test Loss {:.4f} with training MSE metric {:.4f} and testing MSE metric '
        '{:.4f}'.format(
            epoch, batch_n,
            train_loss_r, test_loss_r, tr_metric, te_metric))


def tensorboard_embeddings(model, layer_num, meta_data, logdir):
    """
    This function saves a file to logdir that can be used for viewing
    T-SNE or PCA on any layer in the NN on TensorBoard

    :param model: trained tensorflow model object
    :param layer_num: (int) which layer would you like to present? 0 means look at embedding layer
    :param meta_data: (strings or np.array) used to identify each point in your layer. For example in emebeddings' layer
    if we have 2000 embeddings then we can identify them by the array np.arange(0, 2000, 1)
    :param logdir: (string) path where all tensorboard event files are saved
    """
    from tensorboard.plugins import projector
    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    subwords = meta_data
    with open(os.path.join(logdir, 'metadata.tsv'), "w") as f:
        for subword in subwords:
            f.write("learnt {}\n".format(subword))
        for unknown in np.arange(0, 2000)[~np.isin(range(0, 2000), subwords)]:
            f.write("unknown #{}\n".format(unknown))
    weights = tf.Variable(model.layers[layer_num].get_weights()[0])
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(logdir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(logdir, config)


def write_speci(folder, names, shapes, context_p, heads):
    """
    This is used in order to create a csv with the
    information about which layers were used in this specific run
    of the network --> so that one can replicate easily results

    :param folder: (str) the path were this file should be saved
    :param names:
    :param shapes:
    :param context_p: (int) how many context points were taken
    :param heads: (int) how many heads were used in the MultiHead attention
    """
    with open(os.path.expanduser(folder + '_context_' + str(context_p) + '_speci.csv'), "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['heads', str(heads)])
        for name, shape in zip(names, shapes):
            writer.writerow([str(name.numpy()).split('/')[-2], str(shape.numpy())])


def load_spec(path, e, l, heads, context_p, d=False):
    """
    This function currently only fit a network with A1-A6 layers.
    :param path:
    :param e:
    :param l:
    :param heads:
    :param context_p:
    :return:

    """
    if not os.path.exists(path + '_context_' + str(context_p) + '_speci.csv'):
        print('Does not exist')
        return (e, *l, heads)
    else:
        df = pd.read_csv(path + '_context_' + str(context_p) + '_speci.csv', header=None)
        try:
            heads = int(df.loc[df.iloc[:, 0] == 'heads', 1][0])
        except:
            heads = 8
        e = int(list(df.loc[df.iloc[:, 0] == 'embedding_x', 1])[0].split(' ')[-1][:-1])
        l1 = int(list(df.loc[df.iloc[:, 0] == 'A1', 1])[1][1:-1])
        if d:
            l2 = int(list(df.loc[df.iloc[:, 0] == 'A4', 1])[1][1:-1])
            l3 = int(list(df.loc[df.iloc[:, 0] == 'A5', 1])[1][1:-1])
        else:
            l2 = int(list(df.loc[df.iloc[:, 0] == 'A3', 1])[1][1:-1])
            l3 = int(list(df.loc[df.iloc[:, 0] == 'A4', 1])[1][1:-1])

        l = [l1, l2, l3]

        return (e, *l, heads)


def pre_trained_loader(x, save_dir, e, l, d=True, batch_s=64, context=50, heads=1, run=9999, old =False):
    name_comp = 'run_' + str(run)
    logdir = save_dir + '/logs/' + name_comp
    writer = tf.summary.create_file_writer(logdir)
    folder = save_dir + '/ckpt/check_' + name_comp
    optimizer_c = tf.keras.optimizers.Adam(3e-4)
    e, l1, l2, l3, heads = load_spec(folder, e, l, heads, context, d=d)
    mkdir(folder)
    if d:
        decoder = experimental2d_model.Decoder(e, l1, l2, l3, num_heads=heads)
    elif old:
        decoder = old_model_1D.Decoder(e, l1, l2, l3, num_heads=heads)
    else:
        decoder = experimental_model.Decoder(e, l1, l2, l3, num_heads=heads)
    num_batches = int(x.shape[0] / batch_s)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_c, net=decoder)
    manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    return decoder, optimizer_c, ckpt, manager, num_batches, writer, folder


def gather_idx(c, l=400, b=64):
    """
    This function is used to indicate the network which points are context in each row.
    During training we call this function and then use its output as an index
    to an array of zeros to assign the value 1 in each (row, column) combination we are interested in
    making a prediction.

    :param c: (list of ints) indicating the number of context points for each row
    :param l: (int) the max sequence length in dataset
    :return:
    (np.array) a 2-d array that can be used for fancy indexing.
    The first column represents the row number and the second column represents the column number.

    In order to see an example output, run in main:
    gather_idx([396, 391])

    Output:
    [[  0 396]
    [  0 397]
    [  0 398]
    [  0 399]
    [  1 391]
    [  1 392]
    [  1 393]
    [  1 394]
    [  1 395]
    [  1 396]
    [  1 397]
    [  1 398]
    [  1 399]]
    """
    if type(c) is list:
        cols = [np.arange(c[i], l, 1) for i in range(b)]
        cc = np.concatenate(cols, axis=0)
        rows = [np.repeat(i, len(m)) for i, m in enumerate(cols)]
        r = np.concatenate(rows, axis=0)
        to_gather = np.concatenate((r.reshape(-1, 1), cc.reshape(-1, 1)), 1)
    else:
        cols = [np.arange(c, l, 1) for i in range(b)]
        cc = np.concatenate(cols, axis=0)
        rows = [np.repeat(i, len(m)) for i, m in enumerate(cols)]
        r = np.concatenate(rows, axis=0)
        to_gather = np.concatenate((r.reshape(-1, 1), cc.reshape(-1, 1)), 1)
    return to_gather



def main():
    print(gather_idx([396, 391]))

if __name__ == '__main__':
    main()